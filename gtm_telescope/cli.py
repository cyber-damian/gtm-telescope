# Initial setup + checks: ./scripts/bootstrap.sh  (--check-only | --force-install)

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows / minimal builds
    fcntl = None  # type: ignore[misc, assignment]

from dotenv import load_dotenv

from gtm_telescope import corpus, evaluation, retrieval, vector_cache
from gtm_telescope import llm as llm_mod
from gtm_telescope.batch import BatchError, run_batch
from gtm_telescope.calibrate import calibrate_corpus
from gtm_telescope.config import (
    TelescopeSettings,
    load_telescope_settings,
    parse_yaml_block,
)
from gtm_telescope.prompts import DEFAULT_MODE, MODES, system_prompt
from gtm_telescope.render import RenderResult, format_sales_display, render_from_hits
from gtm_telescope.retrieval import Hit

_PYPROJECT_NAME_RE = re.compile(
    r"^\s*name\s*=\s*[\"']gtm-telescope[\"']",
    re.MULTILINE,
)

_dotenv_loaded = False


def _pyproject_names_gtm_telescope(ppt: Path) -> bool:
    try:
        text = ppt.read_text(encoding="utf-8")
    except OSError:
        return False
    return bool(_PYPROJECT_NAME_RE.search(text))


def _discover_project_root_candidate() -> Path | None:
    """First directory (walking up from start) with config.yaml + gtm-telescope pyproject.toml."""
    for start in (Path.cwd().resolve(), Path(sys.argv[0]).resolve().parent):
        for d in [start, *start.parents]:
            cfg = d / "config.yaml"
            ppt = d / "pyproject.toml"
            if cfg.is_file() and ppt.is_file() and _pyproject_names_gtm_telescope(ppt):
                return d
    return None


@lru_cache(maxsize=1)
def project_root() -> Path:
    """Resolve repo root: cwd walk, then launcher path walk; optional ``telescope.root`` in YAML.

    Result is memoized for the life of the process.
    """
    discovered = _discover_project_root_candidate()
    if discovered is None:
        sys.stderr.write(
            "error: could not find gtm-telescope project root (config.yaml + pyproject name "
            "gtm-telescope). Run from inside the repo, use bin/lens from a checkout, or set "
            "telescope.root in config.yaml.\n"
        )
        sys.exit(1)
    cfg_path = discovered / "config.yaml"
    try:
        text = cfg_path.read_text(encoding="utf-8")
    except OSError:
        return discovered
    tel = parse_yaml_block(text, "telescope")
    raw = (tel.get("root") or "").strip()
    if not raw:
        return discovered
    p = Path(raw).expanduser()
    p = (discovered / p).resolve() if not p.is_absolute() else p.resolve()
    if not p.is_dir():
        sys.stderr.write(f"error: telescope.root is not a directory: {raw}\n")
        sys.exit(1)
    inner_cfg = p / "config.yaml"
    inner_ppt = p / "pyproject.toml"
    if not inner_cfg.is_file() or not inner_ppt.is_file() or not _pyproject_names_gtm_telescope(
        inner_ppt
    ):
        sys.stderr.write(
            f"error: telescope.root must point to a gtm-telescope checkout "
            f"(config.yaml + pyproject): {p}\n"
        )
        sys.exit(1)
    return p


def ensure_dotenv_loaded() -> None:
    """Load optional repo ``llm.env_file`` (default ``.env``) for secrets such as API keys."""
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    _dotenv_loaded = True
    root = project_root()
    cfg = root / "config.yaml"
    if not cfg.is_file():
        return
    try:
        text = cfg.read_text(encoding="utf-8")
    except OSError:
        return
    llm = parse_yaml_block(text, "llm")
    rel = (llm.get("env_file") or ".env").strip() or ".env"
    path = (root / rel).resolve()
    if path.is_file():
        load_dotenv(path, override=False)


def config_path() -> Path:
    return project_root() / "config.yaml"


def load_config(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Load llm and storage sections from config.yaml."""
    if not path.is_file():
        sys.stderr.write(f"error: config not found: {path}\n")
        sys.exit(1)
    text = path.read_text(encoding="utf-8")
    llm = parse_yaml_block(text, "llm")
    storage = parse_yaml_block(text, "storage")
    if "data_dir" not in storage:
        storage["data_dir"] = "data-storage"
    return llm, storage


def _config_yaml_fingerprint() -> str:
    p = config_path()
    try:
        st = p.stat()
        return f"{st.st_mtime_ns}:{st.st_size}"
    except OSError:
        return "missing"


@lru_cache(maxsize=4)
def _load_telescope_settings_cached(_fingerprint: str) -> TelescopeSettings:
    p = config_path()
    text = p.read_text(encoding="utf-8") if p.is_file() else ""
    return load_telescope_settings(text)


def _ts() -> TelescopeSettings:
    """Defaults merged with config.yaml: embedding, retrieval, process, paths (cached per mtime)."""
    return _load_telescope_settings_cached(_config_yaml_fingerprint())


def clear_telescope_settings_cache() -> None:
    """Clear cached settings (for tests after editing config.yaml)."""
    _load_telescope_settings_cached.cache_clear()


def jina_grep_executable() -> str:
    cfg_bin = _ts().jina_grep_bin
    if cfg_bin:
        p = Path(cfg_bin).expanduser().resolve()
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
        sys.stderr.write(f"error: paths.jina_grep_bin is not an executable file: {cfg_bin}\n")
        sys.exit(1)
    venv_bin = project_root() / ".venv" / "bin" / "jina-grep"
    if venv_bin.is_file() and os.access(venv_bin, os.X_OK):
        return str(venv_bin.resolve())
    found = shutil.which("jina-grep")
    if found:
        return found
    sys.stderr.write(
        "error: jina-grep not found. Run ./scripts/bootstrap.sh or "
        "uv sync --extra calibrate, or set paths.jina_grep_bin in config.yaml.\n"
    )
    sys.exit(1)


def rg_executable() -> str:
    cfg_bin = _ts().rg_bin
    if cfg_bin:
        p = Path(cfg_bin).expanduser().resolve()
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
        sys.stderr.write(f"error: paths.rg_bin is not an executable file: {cfg_bin}\n")
        sys.exit(1)
    bundled = project_root() / "resources" / "rg"
    if bundled.is_file() and os.access(bundled, os.X_OK):
        return str(bundled.resolve())
    found = shutil.which("rg")
    if found:
        return found
    sys.stderr.write(
        "error: ripgrep (rg) not found. Run ./scripts/bootstrap.sh to install "
        f"the bundled copy at {bundled}, set paths.rg_bin in config.yaml, or install rg on PATH.\n"
    )
    sys.exit(1)


def data_storage_dir(storage: dict[str, str] | None = None) -> Path:
    root = project_root()
    cfg = storage
    if cfg is None:
        _, cfg = load_config(config_path())
    raw = (cfg.get("data_dir") or "data-storage").strip()
    p = Path(raw)
    if p.is_absolute():
        return p
    return (root / p).resolve()


# Corpus files for lens focus/render: Markdown or UTF-8 plain text only (see lens calibrate).
_MARKDOWN_OR_TEXT_SUFFIXES = corpus.CORPUS_SUFFIXES
_SKIP_CORPUS_NOISE_NAMES_LOWER = frozenset(
    {".ds_store", "thumbs.db", ".gtm_telescope_bootstrap_prefetch_ok"}
)
# Tool-managed directories to ignore during corpus validation (e.g. vector cache DB).
_SKIP_CORPUS_DIRS = corpus.SKIP_CORPUS_DIRS


def _first_non_markdown_or_txt_file_under(data: Path) -> Path | None:
    """First regular file whose suffix is not .md, .markdown, or .txt (case-insensitive), else None.

    Iterative os.scandir; stops on first mismatch. Skips .DS_Store and Thumbs.db. No symlink follow.
    """
    if not data.is_dir():
        return None
    stack: list[Path] = [data.resolve()]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if entry.name in _SKIP_CORPUS_DIRS:
                                continue
                            stack.append(Path(entry.path))
                            continue
                    except OSError:
                        continue
                    try:
                        if not entry.is_file(follow_symlinks=False):
                            continue
                    except OSError:
                        continue
                    if entry.name.lower() in _SKIP_CORPUS_NOISE_NAMES_LOWER:
                        continue
                    suf = Path(entry.name).suffix.lower()
                    if suf in corpus.CONVERTIBLE_SUFFIXES and Path(entry.path).with_suffix(
                        ".md"
                    ).is_file():
                        # calibrate.originals: keep leaves the source next to its converted .md
                        continue
                    if suf not in _MARKDOWN_OR_TEXT_SUFFIXES:
                        return Path(entry.path)
        except OSError:
            continue
    return None


def _require_markdown_or_text_corpus(data: Path) -> None:
    bad = _first_non_markdown_or_txt_file_under(data)
    if bad is None:
        return
    repo_root = project_root().resolve()
    try:
        display = str(bad.resolve().relative_to(repo_root))
    except ValueError:
        display = str(bad.resolve())
    sys.stderr.write(
        f"error: unsupported file type in data storage ({display}). "
        "Only Markdown (.md, .markdown) and plain text (.txt) files are allowed.\n"
        "Run `lens calibrate` first to automatically convert all documents.\n"
    )
    sys.exit(1)


def _require_calibrated_vector_cache(data: Path) -> None:
    """Exit unless the vector cache exists, has chunks, and matches the corpus (focus/render)."""
    if not vector_cache.corpus_has_indexable_files(data):
        return
    info = vector_cache.cache_info(data)
    if not info.get("exists"):
        sys.stderr.write(
            "error: no vector cache. Run `lens calibrate` to convert documents and build "
            "the index, then run this command again.\n"
        )
        sys.exit(1)
    if not info.get("schema_ok", True):
        sys.stderr.write(
            "error: the vector cache was built by an older version (schema "
            f"v{(info.get('meta') or {}).get('schema_version', '?')}). "
            "Run `lens calibrate` to rebuild the index, then run this command again.\n"
        )
        sys.exit(1)
    chunks = int(info.get("chunk_rows") or 0)
    stale = int(info.get("stale_or_new_files") or 0)
    if chunks == 0:
        sys.stderr.write(
            "error: vector cache is empty. Run `lens calibrate` to build the index, "
            "then run this command again.\n"
        )
        sys.exit(1)
    if stale > 0:
        sys.stderr.write(
            f"error: {stale} corpus file(s) changed since the last index. "
            "Run `lens calibrate` to refresh the cache, then run this command again.\n"
        )
        sys.exit(1)


@contextmanager
def _embedding_server_file_lock(data: Path):
    """Serialize embedding-server autostart across concurrent ``lens`` processes (Unix)."""
    if fcntl is None:
        yield
        return
    lock_path = data.resolve() / ".lens-cache" / "embed-server.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            with suppress(OSError):
                fcntl.flock(lock_fd, fcntl.LOCK_UN)


def _embedding_url_port_hint(url: str) -> str:
    p = urllib.parse.urlparse(url)
    if p.port is not None:
        return str(p.port)
    if p.scheme == "https":
        return "443 (default for https)"
    if p.scheme == "http":
        return "80 (default for http)"
    return "unknown"


def _ensure_embedding_server_ready(server_url: str, data: Path) -> None:
    """Start jina-grep serve in the background if needed; poll /health; warm up MLX once."""
    url = server_url.rstrip("/")
    if vector_cache.health_check(url, timeout=2.0):
        return

    with _embedding_server_file_lock(data):
        if vector_cache.health_check(url, timeout=2.0):
            return

        sys.stderr.write("vector cache: embedding server not running; starting it…\n")
        sys.stderr.flush()
        try:
            subprocess.Popen(
                [jina_grep_executable(), "serve", "start"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as e:
            sys.stderr.write(f"error: could not start embedding server: {e}\n")
            sys.exit(1)
        s = _ts()
        deadline = time.monotonic() + s.embed_server_ready_sec
        while time.monotonic() < deadline:
            if vector_cache.health_check(url, timeout=1.0):
                break
            time.sleep(0.25)
        else:
            sys.stderr.write(
                "error: embedding server failed to become healthy in time; "
                "run `lens start` manually\n"
            )
            sys.exit(1)
        tw = s.embed_http_timeout
        try:
            with vector_cache.EmbeddingHttpClient(url, max(tw, 30.0)) as http_client:
                vector_cache.embed_batch(
                    ["warmup"],
                    prompt_name="document",
                    model=s.embedding_model,
                    task=vector_cache.DEFAULT_TASK,
                    dim=s.embedding_truncate_dim,
                    server=url,
                    timeout=max(tw, 30.0),
                    http_client=http_client,
                )
        except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError, KeyError) as e:
            sys.stderr.write(f"error: embedding server warmup failed: {e}\n")
            sys.exit(1)


def usage() -> None:
    prog = Path(sys.argv[0]).name
    sys.stderr.write(
        f"usage:\n"
        f"  {prog} start [extra args for: jina-grep serve start ...]\n"
        f"  {prog} stop\n"
        f"  {prog} status\n"
        f"  {prog} focus <natural language query>\n"
        f"  {prog} render [--short|--story|--explain] [--json] <natural language query>\n"
        f"  {prog} render [--short|--story|--explain] --batch FILE.csv|FILE.xlsx "
        "[--question-col NAME] [--out FILE] [--sheet NAME]\n"
        f"  {prog} calibrate\n"
        f"  {prog} eval [--file eval/questions.yaml] [--k 3,10] [--json]\n"
        f"  {prog} cache clear\n"
        f"  {prog} cache info\n"
        "\n"
        "Tunables live in config.yaml (embedding, retrieval, process, paths); see README.\n"
    )


def _snippet_block_lines(
    rank: int,
    file_display: str,
    line: int,
    raw: str,
    wrap: int,
    *,
    end_line: int | None = None,
    heading: str | None = None,
    note: str = "",
) -> list[str]:
    """Lines for one hit: header then wrapped body (no table markup), easy to copy."""
    para = " ".join(raw.split())
    loc = f"{line}-{end_line}" if end_line and end_line != line else str(line)
    header = f"[{rank}] {file_display}:{loc}"
    if heading:
        header += f" · {heading.replace(' > ', ' › ')}"
    if note:
        header += f"  {note}"
    out: list[str] = [header]
    if not para:
        out.append("(empty)")
    else:
        out.extend(
            textwrap.wrap(
                para,
                width=wrap,
                break_long_words=True,
                break_on_hyphens=False,
            )
        )
    out.append("")
    return out


def _format_total_search_time(elapsed_s: float) -> str:
    return f"\nTotal search time: {elapsed_s:.2f}s\n\n"


def retrieve_hits(query: str, storage: dict[str, str] | None = None) -> tuple[list[Hit], float]:
    """Run hybrid retrieval over the calibrated cache. Returns (hits, wall-clock seconds)."""
    s = _ts()
    data = data_storage_dir(storage)
    if not data.is_dir():
        sys.stderr.write(f"error: data storage path not a directory: {data}\n")
        sys.exit(1)
    t0 = time.perf_counter()
    _ensure_embedding_server_ready(s.embedding_server_url, data)
    try:
        hits = retrieval.retrieve(query, data, s, rg_resolver=rg_executable)
    except retrieval.RetrievalError as e:
        sys.stderr.write(f"error: {e}\n")
        sys.exit(1)
    except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError, KeyError) as e:
        sys.stderr.write(f"error: vector cache search failed: {e}\n")
        sys.exit(1)
    return hits, time.perf_counter() - t0


def format_hits_report(hits: list[Hit]) -> str:
    """Rank table plus copy-friendly snippet blocks (``(no hits)`` when empty)."""
    s = _ts()
    if not hits:
        return "(no hits)\n"
    headers = ("Rank", "File", "Lines", "Source", "RRF", "Cos", "Date")
    n_col = len(headers)
    table_rows: list[tuple[str, ...]] = []
    for i, h in enumerate(hits, start=1):
        lines = f"{h.line_no}-{h.end_line}" if h.end_line != h.line_no else str(h.line_no)
        cos = f"{h.cos:.3f}" if h.cos is not None else "—"
        date_s = (h.doc_date or "—") + (" ⚠ stale" if h.stale else "")
        table_rows.append((str(i), h.path, lines, h.source, f"{h.rrf_score:.4f}", cos, date_s))

    widths = [len(headers[j]) for j in range(n_col)]
    for tr in table_rows:
        for c in range(n_col):
            widths[c] = max(widths[c], len(tr[c]))

    def fmt_row(cells: tuple[str, ...]) -> str:
        return " | ".join(cells[j].ljust(widths[j]) for j in range(n_col))

    sep = "-+-".join("-" * widths[j] for j in range(n_col))
    lines_out: list[str] = [fmt_row(headers), sep]
    lines_out.extend(fmt_row(tr) for tr in table_rows)
    lines_out.append("")
    lines_out.append(
        f"Snippets (top {s.hybrid_report_snippet_blocks} by rank; plain text; copy blocks below)"
    )
    lines_out.append("-" * 72)
    for i, h in enumerate(hits[: s.hybrid_report_snippet_blocks], start=1):
        lines_out.extend(
            _snippet_block_lines(
                i,
                h.path,
                h.line_no,
                h.text,
                s.hybrid_report_snippet_wrap,
                end_line=h.end_line,
                heading=h.heading,
                note="⚠ stale" if h.stale else "",
            )
        )
    return "\n".join(lines_out)


def hybrid_search_report(query: str, storage: dict[str, str] | None = None) -> tuple[str, float]:
    """Retrieve and format as a table report. Returns (report, wall-clock seconds)."""
    hits, elapsed = retrieve_hits(query, storage)
    return format_hits_report(hits), elapsed


def cmd_start(argv: list[str]) -> None:
    s = _ts()
    cmd = [jina_grep_executable(), "serve", "start", *argv]
    try:
        r = subprocess.run(cmd, timeout=s.subprocess_timeout)
    except FileNotFoundError:
        sys.stderr.write("error: failed to execute jina-grep (missing binary?)\n")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        sys.stderr.write(
            f"error: jina-grep serve start timed out after {s.subprocess_timeout}s; "
            "raise process.subprocess_timeout in config.yaml if startup is genuinely slow\n"
        )
        sys.exit(1)
    sys.exit(r.returncode)


def cmd_stop() -> None:
    s = _ts()
    try:
        r = subprocess.run(
            [jina_grep_executable(), "serve", "stop"], timeout=s.subprocess_timeout
        )
    except FileNotFoundError:
        sys.stderr.write("error: failed to execute jina-grep (missing binary?)\n")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        sys.stderr.write(
            f"error: jina-grep serve stop timed out after {s.subprocess_timeout}s\n"
        )
        sys.exit(1)
    sys.exit(r.returncode)


def cmd_status() -> None:
    """Print jina-grep embedding server status and configured URL/port."""
    s = _ts()
    try:
        r = subprocess.run(
            [jina_grep_executable(), "serve", "status"],
            capture_output=True,
            text=True,
            timeout=s.subprocess_timeout,
        )
    except FileNotFoundError:
        sys.stderr.write("error: failed to execute jina-grep (missing binary?)\n")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        sys.stderr.write(
            f"error: jina-grep serve status timed out after {s.subprocess_timeout}s\n"
        )
        sys.exit(1)
    line = (r.stdout or r.stderr or "").strip()
    if line:
        sys.stdout.write(line + "\n")
    else:
        sys.stdout.write("(no status output from jina-grep)\n")
    url = s.embedding_server_url.rstrip("/")
    sys.stdout.write(f"Configured embedding server: {url}\n")
    sys.stdout.write(f"Port (from URL): {_embedding_url_port_hint(url)}\n")


def cmd_focus(argv: list[str]) -> None:
    q = " ".join(argv).strip()
    if not q:
        usage()
        sys.exit(2)
    _, storage = load_config(config_path())
    data = data_storage_dir(storage)
    _require_calibrated_vector_cache(data)
    _require_markdown_or_text_corpus(data)
    report, search_s = hybrid_search_report(q, storage)
    sys.stdout.write(report)
    sys.stdout.write(_format_total_search_time(search_s))


def _parse_eval_args(argv: list[str]) -> tuple[Path, list[int], bool]:
    file_arg: str | None = None
    ks = list(evaluation.DEFAULT_KS)
    as_json = False
    i = 0
    try:
        while i < len(argv):
            a = argv[i]
            if a == "--json":
                as_json = True
            elif a in ("--file", "--k"):
                if i + 1 >= len(argv):
                    raise ValueError(f"{a} requires a value")
                i += 1
                if a == "--file":
                    file_arg = argv[i]
                else:
                    ks = evaluation.parse_ks(argv[i])
            elif a.startswith("--file="):
                file_arg = a.split("=", 1)[1]
            elif a.startswith("--k="):
                ks = evaluation.parse_ks(a.split("=", 1)[1])
            else:
                raise ValueError(f"unknown eval argument {a!r}")
            i += 1
    except ValueError as e:
        sys.stderr.write(f"error: {e}\n")
        usage()
        sys.exit(2)
    if file_arg is None:
        path = project_root() / "eval" / "questions.yaml"
    else:
        path = Path(file_arg).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
    return path, ks, as_json


def cmd_eval(argv: list[str]) -> None:
    """Run golden questions through the same retrieval path as ``focus`` (no LLM)."""
    path, ks, as_json = _parse_eval_args(argv)
    try:
        questions = evaluation.load_questions(path)
    except ValueError as e:
        sys.stderr.write(f"error: {e}\n")
        sys.exit(1)
    _, storage = load_config(config_path())
    data = data_storage_dir(storage)
    _require_calibrated_vector_cache(data)
    _require_markdown_or_text_corpus(data)

    def search(q: str) -> list[str]:
        hits, _ = retrieve_hits(q, storage)
        return [h.path for h in hits]

    results = evaluation.evaluate(questions, search)
    summary = evaluation.summarize(results, ks)
    if as_json:
        sys.stdout.write(evaluation.to_json(results, summary))
    else:
        sys.stdout.write(evaluation.format_report(results, summary))
    # Measurement tool, not a gate: exit code is always 0.


def chat_completion(
    provider: llm_mod.Provider, system: str, user_content: str, *, max_tokens: int
) -> str:
    """Thin CLI wrapper over :func:`llm.complete` (errors -> stderr + exit 1)."""
    try:
        return llm_mod.complete(
            provider, system, user_content, timeout=_ts().http_timeout, max_tokens=max_tokens
        )
    except llm_mod.LLMError as e:
        sys.stderr.write(f"error: {provider.name}: {e}\n")
        sys.exit(1)


def _llm_max_tokens(llm: dict[str, str]) -> int:
    try:
        v = int(llm.get("max_tokens", "").strip())
    except ValueError:
        return llm_mod.DEFAULT_MAX_TOKENS
    return v if v > 0 else llm_mod.DEFAULT_MAX_TOKENS


def cmd_calibrate(argv: list[str]) -> None:
    if argv:
        sys.stderr.write("error: calibrate takes no arguments\n")
        usage()
        sys.exit(2)
    _, storage = load_config(config_path())
    data = data_storage_dir(storage)
    root = project_root()
    rc = calibrate_corpus(data, repo_root=root, originals=_ts().calibrate_originals)
    if rc == 0:
        sys.stderr.write(
            "lens calibrate: building / refreshing vector cache — this can take several minutes…\n"
        )
        sys.stderr.flush()
        try:
            s = _ts()
            _ensure_embedding_server_ready(s.embedding_server_url, data)
            files, chunks = vector_cache.refresh_index(
                data,
                server_url=s.embedding_server_url,
                model=s.embedding_model,
                dim=s.embedding_truncate_dim,
                granularity=s.semantic_granularity,
                timeout=s.embed_http_timeout,
                max_file_bytes=s.max_file_bytes,
                chunk_params=s.chunk_params,
            )
            sys.stderr.write(f"cache: refreshed {files} file(s), {chunks} chunk(s)\n")
        except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError, KeyError) as e:
            sys.stderr.write(f"error: cache refresh failed: {e}\n")
            sys.exit(1)
    sys.exit(rc)


def cmd_cache_clear(argv: list[str]) -> None:
    if argv:
        sys.stderr.write("error: cache clear takes no arguments\n")
        usage()
        sys.exit(2)
    _, storage = load_config(config_path())
    data = data_storage_dir(storage)
    removed, path, errs = vector_cache.clear_cache(data)
    for e in errs:
        sys.stderr.write(f"warning: cache clear: {e}\n")
    if removed and not errs:
        sys.stdout.write(f"Cleared vector cache at {path}\n")
    elif not path.is_dir():
        sys.stdout.write(f"No vector cache directory at {path}\n")
    else:
        sys.stderr.write(f"warning: vector cache at {path} may be only partially removed\n")


def cmd_cache_info(argv: list[str]) -> None:
    if argv:
        sys.stderr.write("error: cache info takes no arguments\n")
        usage()
        sys.exit(2)
    _, storage = load_config(config_path())
    data = data_storage_dir(storage)
    info = vector_cache.cache_info(data)
    if not info.get("exists"):
        sys.stdout.write(
            "No vector cache found. Run `lens calibrate` to convert documents and "
            "build the index.\n"
        )
        return
    meta = info.get("meta") or {}
    kb = info.get("db_size_bytes", 0) / 1024.0
    sys.stdout.write(f"Cache path:        {info['db_path']}\n")
    sys.stdout.write(f"On-disk size:      {kb:.1f} KiB\n")
    sys.stdout.write(f"Schema version:    {meta.get('schema_version', '—')}\n")
    sys.stdout.write(f"Model:             {meta.get('model', '—')}\n")
    sys.stdout.write(f"Task:              {meta.get('task', '—')}\n")
    sys.stdout.write(f"Granularity:       {meta.get('granularity', '—')}\n")
    sys.stdout.write(f"Dim:               {meta.get('dim', '—')}\n")
    sys.stdout.write(
        f"Lexical index:     {'SQLite FTS5' if info.get('fts5') else 'ripgrep fallback'}\n"
    )
    sys.stdout.write(f"Indexed files:     {info.get('file_rows', 0)}\n")
    sys.stdout.write(f"Chunks:            {info.get('chunk_rows', 0)}\n")
    sys.stdout.write(f"Vector blob bytes: {info.get('vector_bytes', 0)}\n")
    mns = info.get("max_mtime_ns")
    if mns is not None:
        dt = datetime.fromtimestamp(mns / 1e9, tz=timezone.utc)
        sys.stdout.write(f"Latest file mtime: {dt.isoformat()}\n")
    stale = info.get("stale_or_new_files", 0)
    sys.stdout.write(f"Stale/new files:   {stale}")
    if stale:
        sys.stdout.write(" (run `lens calibrate` to refresh)\n")
    else:
        sys.stdout.write("\n")


@dataclass(frozen=True)
class RenderArgs:
    question: str
    json_out: bool = False
    mode: str = DEFAULT_MODE
    batch: str | None = None
    question_col: str | None = None
    out: str | None = None
    sheet: str | None = None


def _parse_render_args(argv: list[str]) -> RenderArgs:
    """Parse ``render`` arguments; raises ``ValueError`` with a user-facing message."""
    words: list[str] = []
    opts: dict[str, str] = {}
    json_out = False
    modes: list[str] = []
    mode_flags = {f"--{m}": m for m in MODES}
    value_flags = {"--batch", "--question-col", "--out", "--sheet"}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--json":
            json_out = True
        elif a in mode_flags:
            if mode_flags[a] not in modes:
                modes.append(mode_flags[a])
        elif a in value_flags:
            if i + 1 >= len(argv):
                raise ValueError(f"{a} requires a value")
            opts[a] = argv[i + 1]
            i += 1
        elif a.startswith("--") and len(a) > 2:
            raise ValueError(f"unknown option {a!r} for render")
        else:
            words.append(a)
        i += 1
    if len(modes) > 1:
        raise ValueError(
            "use only one of " + ", ".join(f"--{m}" for m in MODES) + f" (got {len(modes)})"
        )
    ra = RenderArgs(
        question=" ".join(words).strip(),
        json_out=json_out,
        mode=modes[0] if modes else DEFAULT_MODE,
        batch=opts.get("--batch"),
        question_col=opts.get("--question-col"),
        out=opts.get("--out"),
        sheet=opts.get("--sheet"),
    )
    if ra.batch is None:
        if not ra.question:
            raise ValueError("render needs a question (or --batch FILE)")
        if ra.question_col or ra.out or ra.sheet:
            raise ValueError("--question-col, --out and --sheet require --batch")
    else:
        if ra.question:
            raise ValueError("--batch cannot be combined with a question argument")
        if ra.json_out:
            raise ValueError("--json is not supported with --batch (output is CSV/XLSX)")
    return ra


def _render_one(
    question: str,
    provider: llm_mod.Provider,
    mode: str,
    storage: dict[str, str],
    *,
    max_tokens: int,
) -> RenderResult:
    """Retrieve, ask the LLM, post-process. Exits via ``chat_completion`` on LLM errors."""
    ts = _ts()
    system = system_prompt(mode)
    t0 = time.perf_counter()
    hit_list, _ = retrieve_hits(question, storage)
    result = render_from_hits(
        question,
        hit_list,
        chat=lambda prompt: chat_completion(provider, system, prompt, max_tokens=max_tokens),
        max_chunks=ts.render_max_chunks,
        semantic_threshold=ts.semantic_similarity_threshold,
        batch_low_rrf=ts.batch_low_rrf,
    )
    result.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return result


def cmd_render(argv: list[str]) -> None:
    try:
        ra = _parse_render_args(argv)
    except ValueError as e:
        sys.stderr.write(f"error: {e}\n")
        usage()
        sys.exit(2)
    llm, storage = load_config(config_path())
    data = data_storage_dir(storage)
    _require_calibrated_vector_cache(data)
    _require_markdown_or_text_corpus(data)
    models = {p.name: llm.get(f"{p.name}_model", "") for p in llm_mod.PROVIDERS}
    try:
        provider = llm_mod.select_provider(os.environ, models)
    except llm_mod.LLMError as e:
        sys.stderr.write(f"error: {e}\n")
        sys.exit(1)
    max_tokens = _llm_max_tokens(llm)

    def answer(q: str) -> RenderResult:
        return _render_one(q, provider, ra.mode, storage, max_tokens=max_tokens)

    if ra.batch is not None:
        try:
            out = run_batch(
                Path(ra.batch),
                answer=answer,
                question_col=ra.question_col,
                out_path=Path(ra.out) if ra.out else None,
                sheet=ra.sheet,
            )
        except BatchError as e:
            sys.stderr.write(f"error: {e}\n")
            sys.exit(1)
        sys.stdout.write(f"Wrote {out}\n")
        return

    result = answer(ra.question)
    if ra.json_out:
        sys.stdout.write(result.to_json())
        return
    sys.stdout.write(format_sales_display(result, question=ra.question, mode=ra.mode))
    sys.stdout.write(_format_total_search_time(result.elapsed_ms / 1000.0))


def main() -> None:
    if len(sys.argv) < 2:
        usage()
        sys.exit(2)
    ensure_dotenv_loaded()
    try:
        _ = _ts()
    except ValueError as e:
        sys.stderr.write(f"error: {e}\n")
        sys.exit(1)
    sub = sys.argv[1]
    rest = sys.argv[2:]
    if sub == "start":
        cmd_start(rest)
    elif sub == "stop":
        if rest:
            sys.stderr.write("error: stop takes no arguments\n")
            usage()
            sys.exit(2)
        cmd_stop()
    elif sub == "status":
        if rest:
            sys.stderr.write("error: status takes no arguments\n")
            usage()
            sys.exit(2)
        cmd_status()
    elif sub == "focus":
        cmd_focus(rest)
    elif sub == "render":
        cmd_render(rest)
    elif sub == "eval":
        cmd_eval(rest)
    elif sub == "calibrate":
        cmd_calibrate(rest)
    elif sub == "cache":
        if not rest:
            sys.stderr.write("error: cache requires a subcommand (clear | info)\n")
            usage()
            sys.exit(2)
        sub2, tail = rest[0], rest[1:]
        if sub2 == "clear":
            cmd_cache_clear(tail)
        elif sub2 == "info":
            cmd_cache_info(tail)
        else:
            sys.stderr.write(f"error: unknown cache subcommand {sub2!r}\n")
            usage()
            sys.exit(2)
    else:
        sys.stderr.write(f"error: unknown command {sub!r}\n")
        usage()
        sys.exit(2)


if __name__ == "__main__":
    main()

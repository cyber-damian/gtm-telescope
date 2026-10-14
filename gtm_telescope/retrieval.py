"""Hybrid retrieval: FTS5 (or ripgrep) lexical + cosine semantic, fused per chunk with RRF.

``retrieve(query, data, settings)`` is the single entry point used by ``focus``, ``render`` and
``eval``. It only reads the cache (``lens calibrate`` owns indexing) and returns ranked
:class:`Hit` objects; presentation lives elsewhere.
"""

from __future__ import annotations

import re
import shutil
import sqlite3
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from gtm_telescope import vector_cache
from gtm_telescope.config import TelescopeSettings
from gtm_telescope.query import ParsedQuery, parse_query, to_fts5
from gtm_telescope.vector_cache import ChunkKey, ChunkRow

SOURCE_DUAL = "dual"
SOURCE_LEXICAL = "lexical"
SOURCE_SEMANTIC = "semantic"

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# path:lineno:rest (non-greedy path for typical POSIX corpus paths without ':' in names)
_GREP_RECORD_RE = re.compile(r"^(?P<file>.+?):(?P<lineno>\d+):(?P<content>.*)$")


class RetrievalError(RuntimeError):
    """Retrieval failed in a way the user should see (the CLI prints it and exits 1)."""


@dataclass
class Hit:
    path: str  # corpus-relative, POSIX separators
    line_no: int
    end_line: int
    heading: str | None
    text: str
    rrf_score: float
    cos: float | None
    bm25: float | None
    source: str  # dual | lexical | semantic
    doc_date: str | None
    doc_type: str | None
    stale: bool = False
    idx: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------------------
# Filtering helpers (moved from cli.py; applied after fusion)
# --------------------------------------------------------------------------------------

_BOILERPLATE_NEEDLES = (
    "all rights reserved",
    "privacy policy",
    "cookie policy",
    "terms of use",
    "terms of service",
    "follow us on linkedin",
    "follow us on twitter",
    "subscribe to our newsletter",
    "unsubscribe",
    "do not sell my personal information",
    "trademarks are the property",
    "patent pending",
    "© ",
    "(c) 19",
    "(c) 20",
    "linkedin.com/company",
    "twitter.com/",
    "facebook.com/",
    "share on facebook",
    "share this article",
    "for more information, visit",
    "schedule a demo",
    "contact sales",
)


def is_boilerplate_snippet(text: str) -> bool:
    """Heuristic drop for legal/marketing footers and link dumps repeated across many files."""
    t = text.casefold()
    if len(t.strip()) < 8:
        return True
    if any(n in t for n in _BOILERPLATE_NEEDLES):
        return True
    return t.count("http://") + t.count("https://") >= 3


def snippet_dedupe_key(text: str) -> str:
    return " ".join(text.split()).casefold()[:600]


# --------------------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------------------


def rrf_scores(
    lists: Sequence[Sequence[ChunkKey]], *, rrf_k: int
) -> dict[ChunkKey, float]:
    """Reciprocal Rank Fusion: ``sum over lists of 1 / (rrf_k + rank)`` (rank is 1-based)."""
    out: dict[ChunkKey, float] = {}
    for ranked in lists:
        for rank, key in enumerate(ranked, start=1):
            out[key] = out.get(key, 0.0) + 1.0 / (rrf_k + rank)
    return out


def is_exact_match(parsed: ParsedQuery, haystack: str) -> bool:
    """Chunk contains every quoted phrase, or every term when the query has <= 3 terms."""
    h = haystack.casefold()
    if parsed.phrases:
        return all(p.casefold() in h for p in parsed.phrases)
    if parsed.terms and len(parsed.terms) <= 3:
        return all(t in h for t in parsed.terms)
    return False


def is_stale(doc_date: str | None, *, today: date, stale_after_days: int) -> bool:
    if not doc_date:
        return False
    try:
        d = date.fromisoformat(doc_date[:10])
    except ValueError:
        return False
    return (today - d) > timedelta(days=stale_after_days)


# --------------------------------------------------------------------------------------
# Lexical: ripgrep fallback (only when SQLite FTS5 is unavailable)
# --------------------------------------------------------------------------------------


def _run_rg(
    patterns: Sequence[str],
    data: Path,
    *,
    rg_bin: str,
    timeout: float,
    max_lines: int,
) -> list[tuple[str, int, str]]:
    cmd = [rg_bin, "-i", "-n", "-F", "--color", "never", "--no-heading"]
    for p in patterns:
        cmd += ["-e", p]
    cmd += ["--", "."]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(data),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as e:
        raise RetrievalError("failed to execute ripgrep (rg)") from e
    except subprocess.TimeoutExpired as e:
        raise RetrievalError(
            f"ripgrep (rg) timed out after {timeout}s; raise process.subprocess_timeout "
            "in config.yaml or narrow your corpus"
        ) from e
    if proc.returncode not in (0, 1):
        raise RetrievalError((proc.stderr or proc.stdout or "ripgrep failed").strip())
    out: list[tuple[str, int, str]] = []
    for raw in (proc.stdout or "").splitlines():
        m = _GREP_RECORD_RE.match(_ANSI_RE.sub("", raw))
        if not m:
            continue
        out.append((m.group("file"), int(m.group("lineno")), m.group("content")))
        if len(out) >= max_lines:
            break
    return out


def lexical_rg_fallback(
    parsed: ParsedQuery,
    data: Path,
    spans: dict[str, list[tuple[int, int, int]]],
    *,
    rg_bin: str,
    timeout: float,
    max_lines: int,
    top_k: int,
) -> list[ChunkKey]:
    """Map ``rg`` line hits to the chunks containing them.

    Chunks rank by number of distinct terms/phrases matched, then total matching lines.
    """
    patterns = [*parsed.phrases, *parsed.terms]
    if not patterns:
        return []
    folded = [p.casefold() for p in patterns]
    data_resolved = data.resolve()
    matched: dict[ChunkKey, set[int]] = {}
    hit_lines: dict[ChunkKey, int] = {}
    for rel, line, content in _run_rg(
        patterns, data_resolved, rg_bin=rg_bin, timeout=timeout, max_lines=max_lines
    ):
        abs_path = str((data_resolved / rel.removeprefix("./")).resolve())
        chunk_spans = spans.get(abs_path)
        if not chunk_spans:
            continue
        low = content.casefold()
        which = {i for i, p in enumerate(folded) if p in low}
        for idx, start, end in chunk_spans:
            if start <= line <= end:
                key = (abs_path, idx)
                matched.setdefault(key, set()).update(which)
                hit_lines[key] = hit_lines.get(key, 0) + 1
    ranked = sorted(matched, key=lambda k: (-len(matched[k]), -hit_lines[k], k[0], k[1]))
    return ranked[:top_k]


# --------------------------------------------------------------------------------------
# retrieve()
# --------------------------------------------------------------------------------------


def _default_rg_bin() -> str:
    found = shutil.which("rg")
    if not found:
        raise RetrievalError("ripgrep (rg) not found on PATH")
    return found


def retrieve(
    query: str,
    data: Path,
    settings: TelescopeSettings,
    *,
    embed: Callable[[str], np.ndarray] | None = None,
    rg_resolver: Callable[[], str] | None = None,
    today: date | None = None,
    limit: int | None = None,
) -> list[Hit]:
    """Ranked hits for ``query`` (lexical + semantic fused per chunk with RRF).

    ``embed`` maps a query string to a normalized vector (default: the configured embedding
    server); ``rg_resolver`` returns the ripgrep path for the FTS5-less fallback.
    ``limit`` overrides ``hybrid_report_max_rows``.
    """
    data = data.resolve()
    conn = vector_cache.open_readonly(data)
    if conn is None:
        return []
    try:
        return _retrieve(
            query,
            data,
            settings,
            conn,
            embed=embed,
            rg_resolver=rg_resolver or _default_rg_bin,
            today=today or datetime.now(timezone.utc).date(),
            limit=limit or settings.hybrid_report_max_rows,
        )
    finally:
        conn.close()


def _lexical_ranked(
    parsed: ParsedQuery,
    data: Path,
    settings: TelescopeSettings,
    conn: sqlite3.Connection,
    rg_resolver: Callable[[], str],
) -> tuple[list[ChunkKey], dict[ChunkKey, float]]:
    if parsed.is_empty:
        return [], {}
    if vector_cache.has_fts(conn):
        try:
            rows = vector_cache.lexical_search_fts(conn, to_fts5(parsed), settings.lexical_top_k)
            return [k for k, _ in rows], dict(rows)
        except sqlite3.OperationalError as e:
            msg = str(e).lower()
            if "fts5" not in msg and "no such module" not in msg:
                sys.stderr.write(f"warning: FTS5 query failed ({e}); using ripgrep fallback\n")
    ranked = lexical_rg_fallback(
        parsed,
        data,
        vector_cache.chunk_spans(conn),
        rg_bin=rg_resolver(),
        timeout=settings.subprocess_timeout,
        max_lines=settings.lexical_max_lines,
        top_k=settings.lexical_top_k,
    )
    return ranked, {}


def _semantic_ranked(
    query: str,
    settings: TelescopeSettings,
    conn: sqlite3.Connection,
    embed: Callable[[str], np.ndarray] | None,
) -> list[tuple[ChunkKey, float]]:
    vm = vector_cache.load_vector_matrix(conn, settings.max_chunk_rows)
    if vm is None:
        return []
    if embed is None:

        def embed(q: str) -> np.ndarray:
            return vector_cache.embed_query(
                q,
                server_url=settings.embedding_server_url,
                model=settings.embedding_model,
                dim=settings.embedding_truncate_dim,
                timeout=settings.embed_http_timeout,
            )

    qv = embed(query)
    return vector_cache.semantic_rank(
        vm,
        qv,
        top_k=settings.semantic_candidate_top_k,
        threshold=settings.semantic_similarity_threshold,
    )


def _retrieve(
    query: str,
    data: Path,
    settings: TelescopeSettings,
    conn: sqlite3.Connection,
    *,
    embed: Callable[[str], np.ndarray] | None,
    rg_resolver: Callable[[], str],
    today: date,
    limit: int,
) -> list[Hit]:
    parsed = parse_query(query)
    lexical_keys, bm25_by_key = _lexical_ranked(parsed, data, settings, conn, rg_resolver)
    semantic = _semantic_ranked(query, settings, conn, embed)
    semantic_keys = [k for k, _ in semantic]
    cos_by_key = dict(semantic)

    fused = rrf_scores([lexical_keys, semantic_keys], rrf_k=settings.rrf_k)
    if not fused:
        return []
    rows: dict[ChunkKey, ChunkRow] = vector_cache.fetch_chunk_rows(conn, fused)

    lexical_set = set(lexical_keys)
    semantic_set = set(semantic_keys)
    hits: list[Hit] = []
    for key, base in fused.items():
        row = rows.get(key)
        if row is None:
            continue
        score = base
        if is_exact_match(parsed, f"{row.heading or ''}\n{row.text}"):
            score += settings.exact_boost
        if row.doc_type == "card":
            score += settings.card_boost
        in_lex, in_sem = key in lexical_set, key in semantic_set
        source = SOURCE_DUAL if in_lex and in_sem else SOURCE_LEXICAL if in_lex else SOURCE_SEMANTIC
        hits.append(
            Hit(
                path=_rel(row.path, data),
                line_no=row.line_no,
                end_line=row.end_line,
                heading=row.heading,
                text=row.text,
                rrf_score=score,
                cos=cos_by_key.get(key),
                bm25=bm25_by_key.get(key),
                source=source,
                doc_date=row.doc_date,
                doc_type=row.doc_type,
                stale=is_stale(
                    row.doc_date, today=today, stale_after_days=settings.stale_after_days
                ),
                idx=row.idx,
            )
        )
    hits.sort(key=lambda h: (-h.rrf_score, h.path, h.line_no))
    return refine_hits(hits, limit=limit)


def refine_hits(hits: list[Hit], *, limit: int) -> list[Hit]:
    """Drop boilerplate and near-duplicate bodies (after fusion); cap length."""
    out: list[Hit] = []
    seen: set[str] = set()
    for h in hits:
        if is_boilerplate_snippet(h.text):
            continue
        dk = snippet_dedupe_key(h.text)
        if len(dk) < 15 or dk in seen:
            continue
        seen.add(dk)
        out.append(h)
        if len(out) >= limit:
            break
    return out


def _rel(abs_path: str, data: Path) -> str:
    try:
        return Path(abs_path).resolve().relative_to(data).as_posix()
    except ValueError:
        return Path(abs_path).name


__all__ = [
    "SOURCE_DUAL",
    "SOURCE_LEXICAL",
    "SOURCE_SEMANTIC",
    "Hit",
    "RetrievalError",
    "is_boilerplate_snippet",
    "is_exact_match",
    "is_stale",
    "lexical_rg_fallback",
    "refine_hits",
    "retrieve",
    "rrf_scores",
    "snippet_dedupe_key",
]

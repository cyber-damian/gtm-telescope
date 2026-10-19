"""Convert PDF and supported Office documents to Markdown for the corpus (Docling)."""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Callable
from datetime import date, datetime, timezone
from pathlib import Path
from typing import IO, Literal

from gtm_telescope.corpus import CONVERTIBLE_SUFFIXES, SKIP_CORPUS_DIRS

Outcome = Literal["converted", "skipped_empty", "skipped_existing", "failed"]
OriginalsMode = Literal["move", "keep", "delete"]

ORIGINALS_DIRNAME = ".originals"

# OOXML + PDF; Docling supports these via DocumentConverter (confirm versions in upstream docs).
_CALIBRATE_SOURCE_SUFFIXES = CONVERTIBLE_SUFFIXES


def iter_calibrate_source_paths(data_root: Path) -> list[Path]:
    """All regular files under data_root with a supported calibrate extension (case-insensitive).

    Tool-managed directories (``.originals/``, ``.lens-cache/``) are never searched.
    """
    if not data_root.is_dir():
        return []
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(data_root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_CORPUS_DIRS]
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in _CALIBRATE_SOURCE_SUFFIXES and p.is_file():
                found.append(p)
    return sorted(found)


def iter_pdf_paths(data_root: Path) -> list[Path]:
    """Deprecated alias for tests; prefer ``iter_calibrate_source_paths``."""
    return iter_calibrate_source_paths(data_root)


def _display_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _write_markdown_atomic(md_path: Path, text: str) -> None:
    """Write UTF-8 text to md_path via a same-directory temp file and os.replace."""
    parent = md_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp_path = parent / f".{md_path.name}.{os.getpid()}.md.tmp"
    try:
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, md_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


_PLAIN_YAML_VALUE_RE = re.compile(r"^[\w./ ()\-]+$")


def _yaml_scalar(value: str) -> str:
    """Plain scalar when safe, otherwise double-quoted (miniyaml-compatible)."""
    if _PLAIN_YAML_VALUE_RE.match(value) and value == value.strip():
        return value
    return '"' + value.replace('"', "'") + '"'


def provenance_frontmatter(source_rel: str, source_mtime: date, converted: date) -> str:
    """YAML frontmatter recording where a converted Markdown file came from."""
    return (
        "---\n"
        f"source: {_yaml_scalar(source_rel)}\n"
        f"source_mtime: {source_mtime.isoformat()}\n"
        f"converted: {converted.isoformat()}\n"
        "---\n"
    )


def _unique_destination(dest: Path) -> Path:
    """Never overwrite an earlier original: ``name.pdf`` -> ``name.1.pdf`` -> ``name.2.pdf``."""
    if not dest.exists():
        return dest
    n = 1
    while True:
        cand = dest.with_name(f"{dest.stem}.{n}{dest.suffix}")
        if not cand.exists():
            return cand
        n += 1


def _process_convertible_file(
    source: Path,
    converter: object,
    *,
    repo_root: Path,
    data_root: Path,
    index: int,
    total: int,
    stderr: IO[str],
    originals: OriginalsMode = "move",
    today: date | None = None,
) -> Outcome:
    rel = _display_rel(source, repo_root)
    md_path = source.with_suffix(".md")
    try:
        src_stat = source.stat()
    except OSError as exc:
        stderr.write(f"[{index}/{total}] {rel}\n    error: {exc}\n")
        stderr.flush()
        return "failed"

    if originals == "keep" and md_path.is_file() and md_path.stat().st_mtime >= src_stat.st_mtime:
        stderr.write(f"[{index}/{total}] {rel}: up to date ({md_path.name} exists), skipping\n")
        stderr.flush()
        return "skipped_existing"

    stderr.write(f"[{index}/{total}] converting {rel} …\n")
    stderr.flush()
    try:
        result = converter.convert(str(source))
        markdown = result.document.export_to_markdown()
    except Exception as exc:
        stderr.write(f"    error: {exc}\n")
        stderr.flush()
        return "failed"

    if not str(markdown).strip():
        stderr.write(f"    warning: empty markdown from Docling; leaving {source.name} in place\n")
        stderr.flush()
        return "skipped_empty"

    try:
        rel_in_data = source.resolve().relative_to(data_root.resolve())
    except ValueError:
        rel_in_data = Path(source.name)
    header = provenance_frontmatter(
        rel_in_data.as_posix(),
        datetime.fromtimestamp(src_stat.st_mtime, tz=timezone.utc).date(),
        today or datetime.now(timezone.utc).date(),
    )
    try:
        _write_markdown_atomic(md_path, header + str(markdown))
    except OSError as exc:
        stderr.write(f"    error writing {md_path}: {exc}\n")
        stderr.flush()
        return "failed"

    try:
        if originals == "delete":
            source.unlink()
        elif originals == "move":
            dest = _unique_destination(data_root / ORIGINALS_DIRNAME / rel_in_data)
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, dest)
    except OSError as exc:
        stderr.write(f"    error: wrote {md_path.name} but could not handle source file: {exc}\n")
        stderr.flush()
        return "failed"

    out_rel = _display_rel(md_path, repo_root)
    stderr.write(f"    → {out_rel}\n")
    stderr.flush()
    return "converted"


def calibrate_corpus(
    data_root: Path,
    *,
    repo_root: Path,
    stderr: IO[str] | None = None,
    converter_factory: Callable[[], object] | None = None,
    originals: OriginalsMode = "move",
    today: date | None = None,
) -> int:
    """Convert supported documents under data_root to Markdown.

    ``originals`` controls the source file after a successful conversion: ``move`` (default)
    relocates it to ``<data_root>/.originals/<same relative path>``, ``keep`` leaves it in place
    (re-runs skip sources that already have an up-to-date sibling ``.md``), ``delete`` removes it.
    Each generated Markdown file starts with ``source`` / ``source_mtime`` / ``converted``
    frontmatter.

    ``converter_factory`` is for tests; when omitted, Docling's DocumentConverter is used.

    Returns a process exit code: 0 if nothing failed, 1 if any conversion/write/move failed.
    """
    err = stderr if stderr is not None else sys.stderr

    if not data_root.is_dir():
        err.write(f"error: data storage path is not a directory: {data_root}\n")
        return 1

    err.write("lens calibrate: checking for convertible documents (PDF, Office)…\n")
    err.flush()
    sources = iter_calibrate_source_paths(data_root)
    if not sources:
        err.write(f"lens calibrate: no convertible documents found under {data_root}\n")
        return 0

    err.write(f"lens calibrate: processing {len(sources)} file(s)…\n")
    err.flush()

    if converter_factory is not None:
        converter = converter_factory()
    else:
        try:
            from docling.document_converter import DocumentConverter
        except ImportError:
            err.write(
                "error: docling is not installed. Install the calibrate extra, e.g. "
                "uv sync --extra calibrate or re-run ./scripts/bootstrap.sh\n"
            )
            return 1
        converter = DocumentConverter()
    converted = skipped_empty = skipped_existing = failed = 0
    total = len(sources)
    for i, src in enumerate(sources, start=1):
        out = _process_convertible_file(
            src,
            converter,
            repo_root=repo_root,
            data_root=data_root,
            index=i,
            total=total,
            stderr=err,
            originals=originals,
            today=today,
        )
        if out == "converted":
            converted += 1
        elif out == "skipped_empty":
            skipped_empty += 1
        elif out == "skipped_existing":
            skipped_existing += 1
        else:
            failed += 1

    err.write(
        f"lens calibrate: done — converted={converted} "
        f"skipped_empty={skipped_empty} skipped_existing={skipped_existing} failed={failed}\n"
    )
    err.flush()
    return 1 if failed else 0


__all__ = [
    "ORIGINALS_DIRNAME",
    "Outcome",
    "OriginalsMode",
    "calibrate_corpus",
    "iter_calibrate_source_paths",
    "iter_pdf_paths",
    "provenance_frontmatter",
]

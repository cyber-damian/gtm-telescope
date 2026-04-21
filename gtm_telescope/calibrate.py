"""Convert PDF and supported Office documents to Markdown for the corpus (Docling)."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import IO, Literal

Outcome = Literal["converted", "skipped_empty", "failed"]

# OOXML + PDF; Docling supports these via DocumentConverter (confirm versions in upstream docs).
_CALIBRATE_SOURCE_SUFFIXES = frozenset({".pdf", ".docx", ".pptx", ".xlsx"})


def iter_calibrate_source_paths(data_root: Path) -> list[Path]:
    """All regular files under data_root with a supported calibrate extension (case-insensitive)."""
    if not data_root.is_dir():
        return []
    found = [
        p
        for p in data_root.rglob("*")
        if p.is_file() and p.suffix.lower() in _CALIBRATE_SOURCE_SUFFIXES
    ]
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


def _process_convertible_file(
    source: Path,
    converter: object,
    *,
    repo_root: Path,
    index: int,
    total: int,
    stderr: IO[str],
) -> Outcome:
    rel = _display_rel(source, repo_root)
    stderr.write(f"[{index}/{total}] converting {rel} …\n")
    stderr.flush()
    md_path = source.with_suffix(".md")
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
        _write_markdown_atomic(md_path, str(markdown))
    except OSError as exc:
        stderr.write(f"    error writing {md_path}: {exc}\n")
        stderr.flush()
        return "failed"

    try:
        source.unlink()
    except OSError as exc:
        stderr.write(f"    error: wrote {md_path.name} but could not remove source file: {exc}\n")
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
) -> int:
    """Convert supported documents under data_root to Markdown and remove sources on success.

    ``converter_factory`` is for tests; when omitted, Docling's DocumentConverter is used.

    Returns a process exit code: 0 if nothing failed, 1 if any conversion/write/delete failed.
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
                "pip install '.[calibrate]' or re-run ./scripts/bootstrap.sh\n"
            )
            return 1
        converter = DocumentConverter()
    converted = skipped_empty = failed = 0
    total = len(sources)
    for i, src in enumerate(sources, start=1):
        out = _process_convertible_file(
            src,
            converter,
            repo_root=repo_root,
            index=i,
            total=total,
            stderr=err,
        )
        if out == "converted":
            converted += 1
        elif out == "skipped_empty":
            skipped_empty += 1
        else:
            failed += 1

    err.write(
        f"lens calibrate: done — converted={converted} "
        f"skipped_empty={skipped_empty} failed={failed}\n"
    )
    err.flush()
    return 1 if failed else 0


__all__ = [
    "Outcome",
    "calibrate_corpus",
    "iter_calibrate_source_paths",
    "iter_pdf_paths",
]

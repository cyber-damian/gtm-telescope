"""Warm up Docling / Hugging Face model cache (used from bootstrap)."""

from __future__ import annotations

import sys
from io import BytesIO
from typing import IO


def _minimal_pdf_bytes() -> bytes:
    """Single blank page; valid PDF with xref (no third-party deps)."""
    header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n",
    ]
    parts: list[bytes] = [header]
    offsets: list[int] = []
    for obj in objects:
        offsets.append(sum(len(p) for p in parts))
        parts.append(obj)
    body = b"".join(parts)
    xref_pos = len(body)
    xref_lines = [b"xref\n", b"0 4\n", b"0000000000 65535 f \n"]
    for off in offsets:
        xref_lines.append(f"{off:010d} 00000 n \n".encode())
    xref_bytes = b"".join(xref_lines)
    trailer = (
        b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n" + str(xref_pos).encode() + b"\n%%EOF\n"
    )
    return body + xref_bytes + trailer


def prefetch_docling_models(stderr: IO[str] | None = None) -> int:
    """Download Docling default PDF pipeline weights by running a trivial conversion.

    Returns 0 on success, 1 on failure. Prints progress to stderr.
    """
    err = stderr if stderr is not None else sys.stderr
    try:
        from docling.datamodel.base_models import DocumentStream
        from docling.document_converter import DocumentConverter
    except ImportError:
        err.write("Note: skipping Docling model prefetch (docling not installed).\n")
        err.flush()
        return 0

    err.write("Prefetching Docling PDF models (Hugging Face Hub)...\n")
    err.flush()
    buf = BytesIO(_minimal_pdf_bytes())
    stream = DocumentStream(name="warmup.pdf", stream=buf)
    try:
        conv = DocumentConverter()
        result = conv.convert(stream)
        _ = result.document.export_to_markdown()
    except Exception as exc:
        err.write(f"error: Docling model prefetch failed: {exc}\n")
        err.flush()
        return 1

    err.write("Docling model cache OK.\n")
    err.flush()
    return 0


__all__ = ["prefetch_docling_models"]

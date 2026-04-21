"""Tests for Docling warmup PDF (no Docling import required for PDF bytes)."""

from __future__ import annotations

import gtm_telescope.docling_prefetch as dp


def test_minimal_pdf_starts_with_pdf_header() -> None:
    b = dp._minimal_pdf_bytes()
    assert b.startswith(b"%PDF-")
    assert b.endswith(b"%%EOF\n") or b.endswith(b"%%EOF\r\n")


def test_prefetch_skips_gracefully_without_docling(monkeypatch) -> None:
    """Simulate missing docling by blocking any ``docling*`` import (optional dependency)."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # noqa: ANN001
        if name.startswith("docling"):
            raise ImportError("no docling")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from io import StringIO

    err = StringIO()
    assert dp.prefetch_docling_models(stderr=err) == 0
    assert "skipping" in err.getvalue().lower()

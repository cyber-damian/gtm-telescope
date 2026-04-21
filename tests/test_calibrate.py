"""Tests for PDF discovery and calibrate pipeline (no Docling dependency)."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

from gtm_telescope.calibrate import (
    _write_markdown_atomic,
    calibrate_corpus,
    iter_calibrate_source_paths,
    iter_pdf_paths,
)


def test_iter_calibrate_source_paths_pdf_docx_nested(tmp_path: Path) -> None:
    root = tmp_path / "data"
    (root / "deep").mkdir(parents=True)
    (root / "a.pdf").write_text("x", encoding="utf-8")
    (root / "note.txt").write_text("x", encoding="utf-8")
    (root / "deep" / "b.PDF").write_text("x", encoding="utf-8")
    (root / "c.docx").write_bytes(b"PK\x03\x04fake")
    got = iter_calibrate_source_paths(root)
    assert got == sorted([root / "a.pdf", root / "c.docx", root / "deep" / "b.PDF"])
    assert iter_pdf_paths(root) == got


def test_iter_pdf_paths_non_dir_returns_empty(tmp_path: Path) -> None:
    f = tmp_path / "notadir"
    f.write_text("", encoding="utf-8")
    assert iter_pdf_paths(f) == []


def test_write_markdown_atomic(tmp_path: Path) -> None:
    md = tmp_path / "out.md"
    _write_markdown_atomic(md, "# title\n")
    assert md.read_text(encoding="utf-8") == "# title\n"
    assert not list(tmp_path.glob(".out.md.*.md.tmp"))


def test_calibrate_corpus_no_pdfs(tmp_path: Path) -> None:
    data = tmp_path / "empty"
    data.mkdir()
    err = StringIO()
    assert calibrate_corpus(data, repo_root=tmp_path, stderr=err) == 0
    assert "no convertible" in err.getvalue()


def test_calibrate_corpus_not_dir(tmp_path: Path) -> None:
    f = tmp_path / "file"
    f.write_text("", encoding="utf-8")
    err = StringIO()
    assert calibrate_corpus(f, repo_root=tmp_path, stderr=err) == 1
    assert "not a directory" in err.getvalue()


class _FakeDoc:
    def __init__(self, md: str) -> None:
        self._md = md

    def export_to_markdown(self) -> str:
        return self._md


class _FakeResult:
    def __init__(self, md: str) -> None:
        self.document = _FakeDoc(md)


class _FakeConverter:
    def __init__(self, md: str = "# ok\n") -> None:
        self._md = md

    def convert(self, path: str) -> _FakeResult:  # noqa: ARG002
        return _FakeResult(self._md)


def test_calibrate_corpus_converts_and_removes_pdf(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    pdf = data / "x.pdf"
    pdf.write_bytes(b"%PDF-1.1 fake\n")
    err = StringIO()
    code = calibrate_corpus(
        data,
        repo_root=tmp_path,
        stderr=err,
        converter_factory=lambda: _FakeConverter("# ok\n"),
    )
    assert code == 0
    assert not pdf.is_file()
    assert (data / "x.md").read_text(encoding="utf-8") == "# ok\n"
    assert "converted=1" in err.getvalue()


def test_calibrate_corpus_skips_empty_markdown(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    pdf = data / "y.pdf"
    pdf.write_bytes(b"%PDF-1.1 fake\n")
    err = StringIO()
    code = calibrate_corpus(
        data,
        repo_root=tmp_path,
        stderr=err,
        converter_factory=lambda: _FakeConverter("   \n"),
    )
    assert code == 0
    assert pdf.is_file()
    assert not (data / "y.md").exists()
    assert "skipped_empty=1" in err.getvalue()


def test_calibrate_corpus_converts_docx(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    docx = data / "w.docx"
    docx.write_bytes(b"PK\x03\x04")
    err = StringIO()
    code = calibrate_corpus(
        data,
        repo_root=tmp_path,
        stderr=err,
        converter_factory=lambda: _FakeConverter("# docx\n"),
    )
    assert code == 0
    assert not docx.is_file()
    assert (data / "w.md").read_text(encoding="utf-8") == "# docx\n"


def test_calibrate_corpus_failure_keeps_pdf(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    pdf = data / "z.pdf"
    pdf.write_bytes(b"x")

    class Boom:
        def convert(self, path: str) -> None:  # noqa: ARG002
            raise RuntimeError("boom")

    err = StringIO()
    code = calibrate_corpus(
        data,
        repo_root=tmp_path,
        stderr=err,
        converter_factory=lambda: Boom(),
    )
    assert code == 1
    assert pdf.is_file()
    assert "failed=1" in err.getvalue()

"""Tests for PDF discovery and calibrate pipeline (no Docling dependency)."""

from __future__ import annotations

from datetime import date
from io import StringIO
from pathlib import Path

from gtm_telescope import miniyaml
from gtm_telescope.calibrate import (
    _write_markdown_atomic,
    calibrate_corpus,
    iter_calibrate_source_paths,
    iter_pdf_paths,
    provenance_frontmatter,
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


TODAY = date(2026, 10, 14)


def _run(data: Path, tmp_path: Path, md: str = "# ok\n", **kw: object) -> tuple[int, str]:
    err = StringIO()
    code = calibrate_corpus(
        data,
        repo_root=tmp_path,
        stderr=err,
        converter_factory=lambda: _FakeConverter(md),
        today=TODAY,
        **kw,  # type: ignore[arg-type]
    )
    return code, err.getvalue()


def _split_fm(text: str) -> tuple[dict, str]:
    assert text.startswith("---\n")
    head, body = text[4:].split("\n---\n", 1)
    return miniyaml.parse_mapping(head), body


def test_calibrate_default_moves_original_and_writes_frontmatter(tmp_path: Path) -> None:
    data = tmp_path / "data"
    (data / "wp").mkdir(parents=True)
    pdf = data / "wp" / "security.pdf"
    pdf.write_bytes(b"%PDF-1.1 fake\n")
    import os

    os.utime(pdf, (1772409600, 1772409600))  # 2026-03-02 UTC
    code, err = _run(data, tmp_path)
    assert code == 0
    assert "converted=1" in err
    assert not pdf.exists()
    assert (data / ".originals" / "wp" / "security.pdf").read_bytes() == b"%PDF-1.1 fake\n"
    meta, body = _split_fm((data / "wp" / "security.md").read_text(encoding="utf-8"))
    assert meta == {
        "source": "wp/security.pdf",
        "source_mtime": "2026-03-02",
        "converted": "2026-10-14",
    }
    assert body == "# ok\n"


def test_calibrate_move_never_overwrites_earlier_original(tmp_path: Path) -> None:
    data = tmp_path / "data"
    (data / ".originals").mkdir(parents=True)
    (data / ".originals" / "x.pdf").write_bytes(b"old")
    (data / "x.pdf").write_bytes(b"new")
    code, _ = _run(data, tmp_path)
    assert code == 0
    assert (data / ".originals" / "x.pdf").read_bytes() == b"old"
    assert (data / ".originals" / "x.1.pdf").read_bytes() == b"new"


def test_calibrate_delete_mode_removes_source(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    pdf = data / "x.pdf"
    pdf.write_bytes(b"%PDF")
    code, _ = _run(data, tmp_path, originals="delete")
    assert code == 0
    assert not pdf.exists()
    assert not (data / ".originals").exists()
    assert (data / "x.md").is_file()


def test_calibrate_keep_mode_leaves_source_and_is_idempotent(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    pdf = data / "x.pdf"
    pdf.write_bytes(b"%PDF")
    code, err = _run(data, tmp_path, originals="keep")
    assert code == 0 and "converted=1" in err
    assert pdf.is_file()
    md = data / "x.md"
    md.write_text(md.read_text(encoding="utf-8") + "\nhuman edit\n", encoding="utf-8")
    code, err = _run(data, tmp_path, "# changed\n", originals="keep")
    assert code == 0
    assert "skipped_existing=1" in err and "converted=0" in err
    assert "human edit" in md.read_text(encoding="utf-8")


def test_iter_calibrate_source_paths_skips_tool_dirs(tmp_path: Path) -> None:
    data = tmp_path / "data"
    for d in (".originals", ".lens-cache", "docs"):
        (data / d).mkdir(parents=True)
        (data / d / "a.pdf").write_bytes(b"x")
    assert iter_calibrate_source_paths(data) == [data / "docs" / "a.pdf"]


def test_provenance_frontmatter_quotes_unusual_paths() -> None:
    fm = provenance_frontmatter("a: b #1/x.pdf", date(2026, 1, 2), date(2026, 1, 3))
    meta, _ = _split_fm(fm + "body")
    assert meta["source"] == "a: b #1/x.pdf"


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
    assert (data / "w.md").read_text(encoding="utf-8").endswith("---\n# docx\n")
    assert (data / ".originals" / "w.docx").is_file()


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

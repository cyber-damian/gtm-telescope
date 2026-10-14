"""Smoke tests for the pure helpers in gtm_telescope.cli (formatting, corpus validation)."""

from __future__ import annotations

from pathlib import Path

from gtm_telescope.cli import (
    _first_non_markdown_or_txt_file_under,
    _snippet_block_lines,
)
from gtm_telescope.retrieval import (
    Hit,
    is_boilerplate_snippet,
    refine_hits,
    snippet_dedupe_key,
)


def test_snippet_block_lines_header_and_wrap() -> None:
    lines = _snippet_block_lines(1, "notes/a.md", 9, "alpha beta gamma delta", wrap=10)
    assert lines[0] == "[1] notes/a.md:9"
    assert "alpha" in lines[1] or lines[1].startswith("alpha")
    assert lines[-1] == ""


def test_snippet_block_lines_empty_body() -> None:
    lines = _snippet_block_lines(2, "b.txt", 1, "   \n", wrap=40)
    assert lines[0] == "[2] b.txt:1"
    assert lines[1] == "(empty)"
    assert lines[-1] == ""


def test_snippet_block_lines_range_heading_and_note() -> None:
    lines = _snippet_block_lines(
        1, "a.md", 3, "body", wrap=40, end_line=9, heading="Top > Sub", note="⚠ stale"
    )
    assert lines[0] == "[1] a.md:3-9 · Top › Sub  ⚠ stale"


def test_is_boilerplate_snippet_footer() -> None:
    assert is_boilerplate_snippet("Copyright 2025 Acme. All rights reserved.") is True
    assert is_boilerplate_snippet("Follow us on LinkedIn for updates.") is True


def test_is_boilerplate_snippet_technical() -> None:
    assert (
        is_boilerplate_snippet(
            "RevealX ingests Zeek logs and correlates them with EDR alerts on Windows hosts."
        )
        is False
    )


def test_snippet_dedupe_key_normalizes() -> None:
    assert snippet_dedupe_key("  A  B\nC  ") == snippet_dedupe_key("A B C")


def _hit(path: str, line: int, text: str) -> Hit:
    return Hit(
        path=path,
        line_no=line,
        end_line=line,
        heading=None,
        text=text,
        rrf_score=0.03,
        cos=None,
        bm25=None,
        source="lexical",
        doc_date=None,
        doc_type=None,
    )


def test_refine_hits_dedupes_drops_boilerplate_and_limits() -> None:
    hits = [
        _hit("a.md", 1, "unique alpha content here for length"),
        _hit("b.md", 2, "unique alpha content here for length"),
        _hit("d.md", 4, "All rights reserved. Contact sales."),
        _hit("e.md", 5, "another distinct passage with enough text"),
    ]
    out = refine_hits(hits, limit=10)
    assert [h.path for h in out] == ["a.md", "e.md"]
    assert [h.path for h in refine_hits(hits, limit=1)] == ["a.md"]


def test_first_non_markdown_or_txt_none_when_only_allowed(tmp_path: Path) -> None:
    d = tmp_path / "corpus"
    d.mkdir()
    (d / "a.md").write_text("# x", encoding="utf-8")
    (d / "sub").mkdir()
    (d / "sub" / "b.markdown").write_text("x", encoding="utf-8")
    (d / "sub" / "c.txt").write_text("x", encoding="utf-8")
    assert _first_non_markdown_or_txt_file_under(d) is None


def test_first_non_markdown_or_txt_finds_pdf(tmp_path: Path) -> None:
    d = tmp_path / "corpus"
    d.mkdir()
    pdf = d / "x.pdf"
    pdf.write_bytes(b"%PDF")
    assert _first_non_markdown_or_txt_file_under(d) == pdf


def test_first_non_markdown_or_txt_skips_ds_store(tmp_path: Path) -> None:
    d = tmp_path / "corpus"
    d.mkdir()
    (d / ".DS_Store").write_bytes(b"\x00")
    (d / "note.md").write_text("ok", encoding="utf-8")
    assert _first_non_markdown_or_txt_file_under(d) is None


def test_first_non_markdown_or_txt_non_dir_returns_none(tmp_path: Path) -> None:
    f = tmp_path / "notdir"
    f.write_text("", encoding="utf-8")
    assert _first_non_markdown_or_txt_file_under(f) is None

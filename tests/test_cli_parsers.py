"""Smoke tests for the pure parsing helpers in gtm_telescope.cli."""

from __future__ import annotations

from pathlib import Path

from gtm_telescope.cli import (
    _first_non_markdown_or_txt_file_under,
    _is_boilerplate_snippet,
    _MergedRow,
    _parse_path_line_content,
    _refine_merged_hits,
    _snippet_block_lines,
    _snippet_dedupe_key,
    _strip_ansi,
)


def test_strip_ansi_removes_sgr() -> None:
    assert _strip_ansi("\x1b[31mred\x1b[0m text") == "red text"


def test_parse_path_line_content_basic() -> None:
    parsed = _parse_path_line_content("notes/foo.md:42:some content")
    assert parsed == ("notes/foo.md", 42, "some content")


def test_parse_path_line_content_rejects_garbage() -> None:
    assert _parse_path_line_content("not a grep record") is None


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


def test_is_boilerplate_snippet_footer() -> None:
    assert _is_boilerplate_snippet("Copyright 2025 Acme. All rights reserved.") is True
    assert _is_boilerplate_snippet("Follow us on LinkedIn for updates.") is True


def test_is_boilerplate_snippet_technical() -> None:
    assert (
        _is_boilerplate_snippet(
            "RevealX ingests Zeek logs and correlates them with EDR alerts on Windows hosts."
        )
        is False
    )


def test_snippet_dedupe_key_normalizes() -> None:
    assert _snippet_dedupe_key("  A  B\nC  ") == _snippet_dedupe_key("A B C")


def test_refine_merged_hits_dedupes_and_score_floor() -> None:
    merged = [
        _MergedRow("a.md", 1, "semantic", 0.9, "unique alpha content here for length"),
        _MergedRow("b.md", 2, "semantic", 0.85, "unique alpha content here for length"),
        _MergedRow("c.md", 3, "semantic", 0.2, "unique beta low score passage text here"),
        _MergedRow("d.md", 4, "semantic", 0.88, "All rights reserved. Contact sales."),
    ]
    out = _refine_merged_hits(merged, score_floor=0.38)
    assert len(out) == 1
    assert out[0].file_display == "a.md"


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

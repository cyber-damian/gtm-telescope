"""Tests for gtm_telescope.chunking."""

from __future__ import annotations

from gtm_telescope import chunking
from gtm_telescope.chunking import ChunkParams, chunk_markdown, chunk_plain_text, split_frontmatter

SMALL = ChunkParams(chunk_tokens=350, min_chunk_tokens=0)  # no merging, no size splits


def _body(n_chars: int, word: str = "word") -> str:
    return " ".join([word] * (n_chars // (len(word) + 1) + 1))[:n_chars]


def test_heading_paths_nested_and_levels() -> None:
    text = (
        "# Security\n\n" + _body(200) + "\n\n"
        "## Encryption at rest\n\n" + _body(200) + "\n\n"
        "### AES\n\n" + _body(200) + "\n\n"
        "## Transit\n\n" + _body(200) + "\n"
    )
    doc = chunk_markdown(text, SMALL)
    assert [c.heading_path for c in doc.chunks] == [
        "Security",
        "Security > Encryption at rest",
        "Security > Encryption at rest > AES",
        "Security > Transit",
    ]
    assert doc.title == "Security"
    aes = doc.chunks[2]
    assert aes.text.startswith("### AES")


def test_h4_and_deeper_do_not_split() -> None:
    text = "# A\n\n" + _body(100) + "\n\n#### Minor\n\n" + _body(100) + "\n"
    doc = chunk_markdown(text, SMALL)
    assert len(doc.chunks) == 1
    assert "#### Minor" in doc.chunks[0].text


def test_headings_inside_code_fences_ignored() -> None:
    text = "# Real\n\n```bash\n# not a heading\n## nor this\n```\n\n" + _body(100) + "\n"
    doc = chunk_markdown(text, SMALL)
    assert len(doc.chunks) == 1
    assert doc.chunks[0].heading_path == "Real"


def test_line_numbers_are_exact_and_inclusive() -> None:
    text = "# T\n\nalpha line\n\n## S\n\nbeta line\ngamma line\n"
    doc = chunk_markdown(text, SMALL)
    lines = text.splitlines()
    for c in doc.chunks:
        assert c.text == "\n".join(lines[c.line_no - 1 : c.end_line])
    assert [(c.line_no, c.end_line) for c in doc.chunks] == [(1, 3), (5, 8)]


def test_size_split_with_overlap_lines() -> None:
    lines = [f"line {i:02d} " + "x" * 30 for i in range(30)]
    text = "# Big\n" + "\n".join(lines) + "\n"
    params = ChunkParams(chunk_tokens=60, min_chunk_tokens=0, chunk_overlap_lines=2)
    doc = chunk_markdown(text, params)
    assert len(doc.chunks) > 2
    all_lines = text.splitlines()
    for c in doc.chunks:
        assert c.text == "\n".join(all_lines[c.line_no - 1 : c.end_line])
        assert chunking.estimate_tokens(c.text) <= 60 + 20
        assert c.heading_path == "Big"
    for prev, nxt in zip(doc.chunks, doc.chunks[1:], strict=False):
        assert nxt.line_no == prev.end_line - 1  # two lines of overlap
    assert doc.chunks[0].line_no == 1
    assert doc.chunks[-1].end_line == len(all_lines)


def test_zero_overlap_split_is_contiguous() -> None:
    lines = [f"line {i:02d} " + "x" * 30 for i in range(20)]
    doc = chunk_markdown(
        "\n".join(lines), ChunkParams(chunk_tokens=60, min_chunk_tokens=0, chunk_overlap_lines=0)
    )
    for prev, nxt in zip(doc.chunks, doc.chunks[1:], strict=False):
        assert nxt.line_no == prev.end_line + 1


def test_small_sections_merge_under_same_parent() -> None:
    text = "# Doc\n\n## A\n\nshort a\n\n## B\n\nshort b\n\n## C\n\n" + _body(280) + "\n"
    doc = chunk_markdown(text, ChunkParams(chunk_tokens=100, min_chunk_tokens=40))
    # "# Doc" + A + B are all tiny and share the parent -> one chunk labelled with the parent;
    # merging C as well would exceed chunk_tokens, so C stands alone.
    assert len(doc.chunks) == 2
    assert "short a" in doc.chunks[0].text and "short b" in doc.chunks[0].text
    assert doc.chunks[0].heading_path == "Doc"
    assert doc.chunks[1].heading_path == "Doc > C"


def test_small_ancestor_merges_into_child_keeping_child_path() -> None:
    text = "# Runbook\n\n## Outage\n\n" + _body(600) + "\n"
    doc = chunk_markdown(text, ChunkParams(chunk_tokens=350, min_chunk_tokens=40))
    assert len(doc.chunks) == 1
    assert doc.chunks[0].heading_path == "Runbook > Outage"
    assert doc.chunks[0].line_no == 1


def test_small_section_does_not_merge_across_parents() -> None:
    text = "# One\n\n## Tiny\n\nx\n\n# Two\n\n## Big\n\n" + _body(600) + "\n"
    doc = chunk_markdown(text, ChunkParams(chunk_tokens=350, min_chunk_tokens=40))
    paths = [c.heading_path for c in doc.chunks]
    assert "One > Tiny" in paths or "One" in paths
    for c in doc.chunks:
        assert not (c.heading_path.startswith("Two") and "Tiny" in c.text)


def test_frontmatter_stripped_returned_and_line_numbers_original() -> None:
    text = (
        "---\ntype: battlecard\nupdated: 2023-01-10\ntitle: Pricing\n---\n"
        "# H\n\nbody text here\n"
    )
    meta, body_lines, offset = split_frontmatter(text)
    assert meta == {"type": "battlecard", "updated": "2023-01-10", "title": "Pricing"}
    assert offset == 5 and body_lines[0] == "# H"
    doc = chunk_markdown(text, SMALL)
    assert doc.meta["type"] == "battlecard"
    assert doc.title == "Pricing"
    assert "battlecard" not in doc.chunks[0].text
    c = doc.chunks[0]
    assert (c.line_no, c.end_line) == (6, 8)
    assert text.splitlines()[c.line_no - 1] == "# H"


def test_unclosed_frontmatter_is_not_frontmatter() -> None:
    meta, lines, offset = split_frontmatter("---\nkey: v\nno end\n")
    assert meta == {} and offset == 0 and len(lines) == 3


def test_plain_text_paragraphs_and_merge() -> None:
    text = "Title line\n\n" + _body(400) + "\n\n" + _body(400, "other") + "\n"
    doc = chunk_plain_text(text, ChunkParams(chunk_tokens=350, min_chunk_tokens=40))
    # Tiny title merges into the first paragraph; second paragraph stands alone.
    assert len(doc.chunks) == 2
    assert doc.chunks[0].text.startswith("Title line")
    assert doc.chunks[0].line_no == 1
    assert all(c.heading_path == "" for c in doc.chunks)


def test_chunk_document_dispatch_by_suffix() -> None:
    text = "# H\n\nbody\n"
    assert chunking.chunk_document(text, ".md").chunks[0].heading_path == "H"
    assert chunking.chunk_document(text, ".TXT").chunks[0].heading_path == ""


def test_embedding_input_prepends_heading_only_when_present() -> None:
    with_h = chunking.Chunk(1, 2, "A > B", "text")
    without = chunking.Chunk(1, 2, "", "text")
    assert chunking.embedding_input(with_h) == "A > B\ntext"
    assert chunking.embedding_input(without) == "text"


def test_empty_and_blank_documents() -> None:
    assert chunk_markdown("").chunks == []
    assert chunk_markdown("\n\n  \n").chunks == []

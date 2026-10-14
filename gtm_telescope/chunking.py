"""Markdown-aware chunking with optional YAML frontmatter.

* Frontmatter (``---`` block at the top of the file) is parsed with ``miniyaml``, stripped
  from chunk text and returned as metadata. Chunk line numbers always refer to the *original*
  file, so ``path:line`` citations stay correct.
* Markdown is split at ATX headings ``#``-``###`` (``####`` and deeper stay in their parent
  section; ``#`` inside fenced code blocks is ignored). Each chunk carries its heading path,
  e.g. ``"Security > Encryption at rest"``.
* Sections larger than ``chunk_tokens`` are split at line boundaries with
  ``chunk_overlap_lines`` of overlap. Sections smaller than ``min_chunk_tokens`` merge with the
  following section under the same parent heading.
* Plain text is split into paragraphs (blank lines) and then follows the same size rules.
* Token estimate is the project-wide heuristic ``len(text) // 3``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from gtm_telescope import miniyaml

DEFAULT_CHUNK_TOKENS = 350
DEFAULT_MIN_CHUNK_TOKENS = 40
DEFAULT_CHUNK_OVERLAP_LINES = 2
HEADING_SEP = " > "

_HEADING_RE = re.compile(r"^ {0,3}(#{1,3})[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$")
_FENCE_RE = re.compile(r"^ {0,3}(```+|~~~+)")


@dataclass(frozen=True)
class Chunk:
    """One retrievable passage. Lines are 1-based and inclusive, in the original file."""

    line_no: int
    end_line: int
    heading_path: str
    text: str


@dataclass(frozen=True)
class ChunkParams:
    granularity: str = "markdown"
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS
    min_chunk_tokens: int = DEFAULT_MIN_CHUNK_TOKENS
    chunk_overlap_lines: int = DEFAULT_CHUNK_OVERLAP_LINES


@dataclass
class ChunkedDoc:
    meta: dict[str, Any] = field(default_factory=dict)
    title: str | None = None
    chunks: list[Chunk] = field(default_factory=list)


def estimate_tokens(text: str) -> int:
    return len(text) // 3


def split_frontmatter(content: str) -> tuple[dict[str, Any], list[str], int]:
    """Return ``(meta, body_lines, line_offset)``.

    ``line_offset`` is the number of leading lines consumed by frontmatter, so body line ``i``
    (0-based) is original line ``line_offset + i + 1``. No frontmatter -> ``({}, all, 0)``.
    """
    lines = content.splitlines()
    if lines and lines[0].lstrip("\ufeff").rstrip() == "---":
        for j in range(1, len(lines)):
            if lines[j].rstrip() in ("---", "..."):
                meta = miniyaml.parse_mapping("\n".join(lines[1:j]))
                return meta, lines[j + 1 :], j + 1
    return {}, lines, 0


@dataclass
class _Section:
    start: int  # index into body lines (inclusive)
    end: int  # exclusive
    parts: list[str]
    mergeable: bool = True
    composite: bool = False  # True once sibling sections were merged (path is their parent)


def _markdown_sections(lines: list[str]) -> tuple[list[_Section], str | None]:
    """Split body lines at headings. Returns sections plus the first H1 text."""
    sections: list[_Section] = []
    stack: list[tuple[int, str]] = []
    first_h1: str | None = None
    in_fence: str | None = None
    cur_start = 0
    cur_parts: list[str] = []
    cur_mergeable = False  # preamble before the first heading stays standalone

    def close(end: int) -> None:
        if end > cur_start:
            sections.append(_Section(cur_start, end, list(cur_parts), cur_mergeable))

    for i, line in enumerate(lines):
        fm = _FENCE_RE.match(line)
        if fm:
            marker = fm.group(1)[0]
            if in_fence is None:
                in_fence = marker
            elif in_fence == marker:
                in_fence = None
            continue
        if in_fence is not None:
            continue
        hm = _HEADING_RE.match(line)
        if not hm or not hm.group(2).strip():
            continue
        level = len(hm.group(1))
        title = hm.group(2).strip()
        if level == 1 and first_h1 is None:
            first_h1 = title
        close(i)
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        cur_start = i
        cur_parts = [t for _, t in stack]
        cur_mergeable = True
    close(len(lines))
    return sections, first_h1


def _paragraph_sections(lines: list[str]) -> list[_Section]:
    sections: list[_Section] = []
    start: int | None = None
    for i, line in enumerate(lines):
        if line.strip():
            if start is None:
                start = i
        elif start is not None:
            sections.append(_Section(start, i, []))
            start = None
    if start is not None:
        sections.append(_Section(start, len(lines), []))
    return sections


def _common_prefix(a: list[str], b: list[str]) -> list[str]:
    out: list[str] = []
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        out.append(x)
    return out


def _mergeable_pair(acc: _Section, nxt: _Section) -> bool:
    if not nxt.mergeable:
        return False
    child = nxt.parts[: len(acc.parts)] == acc.parts
    sibling = acc.parts[:-1] == nxt.parts[:-1]
    return child or sibling


def _merge_small(
    lines: list[str], sections: list[_Section], min_tokens: int, max_tokens: int
) -> list[_Section]:
    """Merge a small section into the following one under the same parent heading.

    A merge never produces a chunk larger than ``max_tokens``.
    """
    out: list[_Section] = []
    acc: _Section | None = None
    for sec in sections:
        if acc is None:
            acc = sec
            continue
        acc_small = estimate_tokens("\n".join(lines[acc.start : acc.end])) < min_tokens
        fits = estimate_tokens("\n".join(lines[acc.start : sec.end])) <= max_tokens
        if acc_small and fits and acc.mergeable and _mergeable_pair(acc, sec):
            # A lone ancestor heading merges into its child and keeps the child's (more
            # specific) path; merged siblings are labelled with their shared parent.
            is_child = sec.parts[: len(acc.parts)] == acc.parts
            if is_child and not acc.composite:
                acc = _Section(acc.start, sec.end, list(sec.parts))
            else:
                acc = _Section(
                    acc.start, sec.end, _common_prefix(acc.parts, sec.parts), composite=True
                )
        else:
            out.append(acc)
            acc = sec
    if acc is not None:
        out.append(acc)
    return out


def _trim_blank(lines: list[str], start: int, end: int) -> tuple[int, int]:
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return start, end


def _split_by_size(
    lines: list[str], start: int, end: int, max_chars: int, overlap: int
) -> list[tuple[int, int]]:
    """Greedy line-boundary windows over ``[start, end)`` with ``overlap`` repeated lines."""
    spans: list[tuple[int, int]] = []
    s = start
    while s < end:
        size = 0
        e = s
        while e < end:
            add = len(lines[e]) + 1
            if e > s and size + add > max_chars:
                break
            size += add
            e += 1
        spans.append((s, e))
        if e >= end:
            break
        s = max(e - overlap, s + 1)
    return spans


def _build_chunks(
    lines: list[str], sections: list[_Section], offset: int, params: ChunkParams
) -> list[Chunk]:
    max_chars = max(params.chunk_tokens, 1) * 3
    chunks: list[Chunk] = []
    for sec in sections:
        s, e = _trim_blank(lines, sec.start, sec.end)
        if s >= e:
            continue
        path = HEADING_SEP.join(sec.parts)
        spans = [(s, e)]
        if estimate_tokens("\n".join(lines[s:e])) > params.chunk_tokens:
            spans = _split_by_size(lines, s, e, max_chars, max(params.chunk_overlap_lines, 0))
        for a, b in spans:
            a, b = _trim_blank(lines, a, b)
            if a >= b:
                continue
            chunks.append(
                Chunk(
                    line_no=offset + a + 1,
                    end_line=offset + b,
                    heading_path=path,
                    text="\n".join(lines[a:b]),
                )
            )
    return chunks


def chunk_markdown(content: str, params: ChunkParams | None = None) -> ChunkedDoc:
    p = params or ChunkParams()
    meta, lines, offset = split_frontmatter(content)
    sections, first_h1 = _markdown_sections(lines)
    sections = _merge_small(lines, sections, p.min_chunk_tokens, p.chunk_tokens)
    title = str(meta["title"]).strip() if meta.get("title") else first_h1
    chunks = _build_chunks(lines, sections, offset, p)
    return ChunkedDoc(meta=meta, title=title or None, chunks=chunks)


def chunk_plain_text(content: str, params: ChunkParams | None = None) -> ChunkedDoc:
    p = params or ChunkParams()
    meta, lines, offset = split_frontmatter(content)
    sections = _merge_small(
        lines, _paragraph_sections(lines), p.min_chunk_tokens, p.chunk_tokens
    )
    title = str(meta["title"]).strip() if meta.get("title") else None
    return ChunkedDoc(meta=meta, title=title, chunks=_build_chunks(lines, sections, offset, p))


def chunk_document(content: str, suffix: str, params: ChunkParams | None = None) -> ChunkedDoc:
    """Dispatch on file suffix: ``.txt`` -> paragraphs, everything else -> Markdown."""
    if suffix.lower() == ".txt":
        return chunk_plain_text(content, params)
    return chunk_markdown(content, params)


def embedding_input(chunk: Chunk) -> str:
    """Text sent to the embedding model: heading path adds context; display text stays raw."""
    if chunk.heading_path:
        return f"{chunk.heading_path}\n{chunk.text}"
    return chunk.text


__all__ = [
    "DEFAULT_CHUNK_OVERLAP_LINES",
    "DEFAULT_CHUNK_TOKENS",
    "DEFAULT_MIN_CHUNK_TOKENS",
    "HEADING_SEP",
    "Chunk",
    "ChunkParams",
    "ChunkedDoc",
    "chunk_document",
    "chunk_markdown",
    "chunk_plain_text",
    "embedding_input",
    "estimate_tokens",
    "split_frontmatter",
]

"""Cited answers for ``lens render``: context building, citation post-processing, confidence.

The LLM sees the top ``render_max_chunks`` hits with their *full* chunk text, numbered
``[1]..[N]``. Its answer is then audited deterministically: citations outside ``1..N`` are
dropped, a ``Sources:`` block maps each used ``[n]`` to ``path:line-end_line``, uncited answers
are flagged, and stale sources are annotated.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from gtm_telescope.retrieval import SOURCE_DUAL, Hit

UNCITED_WARNING = "⚠ Uncited answer — verify before sending."
NOT_COVERED_PREFIX = "not covered"
NO_HITS_ANSWER = "Not covered in the docs: no relevant excerpts were retrieved for this question."

# [1], [12], and the tolerated list form [1, 2]
_CITE_RE = re.compile(r"(\s*)\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass(frozen=True)
class Citation:
    n: int
    path: str
    line: int
    end_line: int
    doc_date: str | None
    stale: bool

    @property
    def location(self) -> str:
        """``path:line-end_line`` (``path:line`` for single-line chunks)."""
        if self.end_line > self.line:
            return f"{self.path}:{self.line}-{self.end_line}"
        return f"{self.path}:{self.line}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "path": self.path,
            "line": self.line,
            "end_line": self.end_line,
            "doc_date": self.doc_date,
            "stale": self.stale,
        }


@dataclass
class RenderResult:
    answer: str  # post-processed body (uncited warning / stale notes included, no Sources block)
    citations: list[Citation] = field(default_factory=list)  # used, valid, in order of use
    not_covered: bool = False
    uncited: bool = False
    confidence: str = "low"
    elapsed_ms: float = 0.0

    @property
    def sources_block(self) -> str:
        if not self.citations:
            return ""
        lines = ["Sources:"] + [f"[{c.n}] {c.location}" for c in self.citations]
        return "\n".join(lines)

    def full_text(self) -> str:
        """Answer plus ``Sources:`` block, as printed by ``lens render``."""
        block = self.sources_block
        return f"{self.answer}\n\n{block}\n" if block else f"{self.answer}\n"

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "confidence": self.confidence,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_json_dict(), indent=2, ensure_ascii=False) + "\n"


def _heading_label(heading: str | None) -> str:
    return heading.replace(" > ", " › ") if heading else ""


def context_header(n: int, hit: Hit) -> str:
    parts = [f"[{n}] {hit.path}:{hit.line_no}-{hit.end_line}"]
    label = _heading_label(hit.heading)
    if label:
        parts.append(label)
    if hit.doc_date:
        parts.append(f"updated {hit.doc_date}" + (" (stale)" if hit.stale else ""))
    elif hit.stale:
        parts.append("(stale)")
    return " · ".join(parts)


def build_context(hits: Sequence[Hit], max_chunks: int) -> tuple[str, list[Hit]]:
    """Numbered sources with full chunk text. Returns ``(context, hits_used)``."""
    used = list(hits[:max_chunks])
    blocks = [f"{context_header(i, h)}\n{h.text.strip()}" for i, h in enumerate(used, start=1)]
    return "\n\n".join(blocks), used


def build_user_prompt(question: str, context: str) -> str:
    return f"Question:\n{question}\n\nSources:\n{context}\n"


def sanitize_citations(text: str, n_sources: int) -> tuple[str, list[int]]:
    """Drop ``[n]`` markers outside ``1..n_sources``. Returns ``(clean_text, used_in_order)``."""
    used: list[int] = []

    def repl(m: re.Match[str]) -> str:
        nums = [int(x) for x in re.split(r"\s*,\s*", m.group(2))]
        valid = [n for n in nums if 1 <= n <= n_sources]
        if not valid:
            return ""  # also drops the whitespace before the marker
        for n in valid:
            if n not in used:
                used.append(n)
        if len(valid) == len(nums):
            return m.group(0)
        return m.group(1) + "".join(f"[{n}]" for n in valid)

    return _CITE_RE.sub(repl, text), used


def is_not_covered(text: str) -> bool:
    return text.strip().lower().startswith(NOT_COVERED_PREFIX)


def compute_confidence(
    *,
    not_covered: bool,
    n_valid_citations: int,
    top_hit: Hit | None,
    semantic_threshold: float,
    batch_low_rrf: float,
) -> str:
    """``not_covered`` | ``high`` | ``low`` | ``medium`` (see README for the rules).

    ``n_valid_citations`` counts distinct valid sources cited.
    """
    if not_covered:
        return "not_covered"
    if (
        n_valid_citations >= 2
        and top_hit is not None
        and (
            top_hit.source == SOURCE_DUAL
            or (top_hit.cos is not None and top_hit.cos >= semantic_threshold + 0.15)
        )
    ):
        return "high"
    if n_valid_citations == 0 or top_hit is None or top_hit.rrf_score < batch_low_rrf:
        return "low"
    return "medium"


def _stale_notes(citations: Sequence[Citation], answer: str) -> str:
    notes = [
        f"Stale source [{c.n}] (verify: source dated {c.doc_date})"
        for c in citations
        if c.stale and c.doc_date and f"source dated {c.doc_date}" not in answer
    ]
    return "\n".join(notes)


def postprocess_answer(
    raw: str,
    hits_used: Sequence[Hit],
    *,
    top_hit: Hit | None,
    semantic_threshold: float,
    batch_low_rrf: float,
) -> RenderResult:
    """Audit the model output against the sources it was given."""
    clean, used = sanitize_citations(raw.strip(), len(hits_used))
    citations = [
        Citation(
            n=n,
            path=hits_used[n - 1].path,
            line=hits_used[n - 1].line_no,
            end_line=hits_used[n - 1].end_line,
            doc_date=hits_used[n - 1].doc_date,
            stale=hits_used[n - 1].stale,
        )
        for n in used
    ]
    not_covered = is_not_covered(clean)
    answer = clean
    notes = _stale_notes(citations, answer)
    if notes:
        answer = f"{answer}\n\n{notes}"
    uncited = not citations and not not_covered
    if uncited:
        answer = f"{UNCITED_WARNING}\n\n{answer}"
    return RenderResult(
        answer=answer,
        citations=citations,
        not_covered=not_covered,
        uncited=uncited,
        confidence=compute_confidence(
            not_covered=not_covered,
            n_valid_citations=len(citations),
            top_hit=top_hit,
            semantic_threshold=semantic_threshold,
            batch_low_rrf=batch_low_rrf,
        ),
    )


def render_from_hits(
    question: str,
    hits: Sequence[Hit],
    *,
    chat: Callable[[str], str],
    max_chunks: int,
    semantic_threshold: float,
    batch_low_rrf: float,
) -> RenderResult:
    """Answer ``question`` from retrieved ``hits``. ``chat`` maps the user prompt to model text."""
    if not hits:
        return RenderResult(
            answer=NO_HITS_ANSWER, not_covered=True, confidence="not_covered", citations=[]
        )
    context, used = build_context(hits, max_chunks)
    raw = chat(build_user_prompt(question, context))
    return postprocess_answer(
        raw,
        used,
        top_hit=hits[0],
        semantic_threshold=semantic_threshold,
        batch_low_rrf=batch_low_rrf,
    )


__all__ = [
    "NOT_COVERED_PREFIX",
    "NO_HITS_ANSWER",
    "UNCITED_WARNING",
    "Citation",
    "RenderResult",
    "build_context",
    "build_user_prompt",
    "compute_confidence",
    "context_header",
    "is_not_covered",
    "postprocess_answer",
    "render_from_hits",
    "sanitize_citations",
]

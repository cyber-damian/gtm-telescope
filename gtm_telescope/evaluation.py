"""Golden-question evaluation for retrieval (``lens eval``).

Pure logic only: loading ``eval/questions.yaml``, ranking, and metrics. The CLI supplies a
``search`` callable that returns the ranked corpus-relative file path of every hit, so the
exact same retrieval path as ``lens focus`` is measured (no LLM involved).

Metric definitions
------------------
* ``rank``: 1-based position of the first hit whose file is in ``expect`` (``None`` = miss).
* ``Recall@k``: fraction of answerable questions with ``rank <= k``.
* ``MRR``: mean of ``1/rank`` over answerable questions (miss counts as 0).
* Questions with ``expect: []`` are *negative* ("the corpus should not answer this"). They are
  excluded from recall/MRR and reported as an abstain rate: the fraction that returned no hits.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gtm_telescope.miniyaml import parse_list_of_mappings

DEFAULT_KS = (3, 10)


@dataclass(frozen=True)
class Question:
    q: str
    expect: tuple[str, ...]
    tags: tuple[str, ...] = ()

    @property
    def negative(self) -> bool:
        return not self.expect


@dataclass
class QuestionResult:
    question: Question
    rank: int | None
    ranked_paths: list[str] = field(default_factory=list)

    @property
    def abstained(self) -> bool:
        return not self.ranked_paths


def _as_list(v: Any) -> list[str]:
    if v is None or v == "":
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    return [str(v)]


def _norm_rel(p: str) -> str:
    return Path(p.strip()).as_posix().removeprefix("./")


def load_questions(path: Path) -> list[Question]:
    """Load and validate an eval file; raises ``ValueError`` with a clear message."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ValueError(f"cannot read eval file {path}: {e}") from e
    items = parse_list_of_mappings(text)
    if not items:
        raise ValueError(f"no questions found in {path}")
    out: list[Question] = []
    for i, item in enumerate(items, start=1):
        q = item.get("q")
        if not isinstance(q, str) or not q.strip():
            raise ValueError(f"{path}: question #{i} is missing a non-empty 'q'")
        if "expect" not in item:
            raise ValueError(f"{path}: question #{i} ({q!r}) is missing 'expect'")
        out.append(
            Question(
                q=q.strip(),
                expect=tuple(_norm_rel(p) for p in _as_list(item["expect"])),
                tags=tuple(_as_list(item.get("tags"))),
            )
        )
    return out


def first_expected_rank(ranked_paths: Sequence[str], expect: Sequence[str]) -> int | None:
    wanted = set(expect)
    for i, p in enumerate(ranked_paths, start=1):
        if p in wanted:
            return i
    return None


def evaluate(
    questions: Sequence[Question], search: Callable[[str], list[str]]
) -> list[QuestionResult]:
    results: list[QuestionResult] = []
    for question in questions:
        ranked = [_norm_rel(p) for p in search(question.q)]
        rank = None if question.negative else first_expected_rank(ranked, question.expect)
        results.append(QuestionResult(question=question, rank=rank, ranked_paths=ranked))
    return results


def compute_metrics(results: Sequence[QuestionResult], ks: Sequence[int]) -> dict[str, Any]:
    positives = [r for r in results if not r.question.negative]
    negatives = [r for r in results if r.question.negative]
    n_pos = len(positives)
    recall: dict[str, float | None] = {}
    for k in ks:
        if n_pos:
            recall[str(k)] = sum(1 for r in positives if r.rank is not None and r.rank <= k) / n_pos
        else:
            recall[str(k)] = None
    mrr = sum(1.0 / r.rank for r in positives if r.rank) / n_pos if n_pos else None
    abstain = sum(1 for r in negatives if r.abstained) / len(negatives) if negatives else None
    return {
        "n": len(results),
        "n_answerable": n_pos,
        "n_negative": len(negatives),
        "recall": recall,
        "mrr": mrr,
        "abstain_rate": abstain,
    }


def summarize(results: Sequence[QuestionResult], ks: Sequence[int]) -> dict[str, Any]:
    tags = sorted({t for r in results for t in r.question.tags})
    return {
        "ks": list(ks),
        "overall": compute_metrics(results, ks),
        "by_tag": {
            t: compute_metrics([r for r in results if t in r.question.tags], ks) for t in tags
        },
    }


def parse_ks(raw: str) -> list[int]:
    try:
        ks = sorted({int(x) for x in raw.split(",") if x.strip()})
    except ValueError as e:
        raise ValueError(f"--k must be comma-separated integers (got {raw!r})") from e
    if not ks or any(k < 1 for k in ks):
        raise ValueError(f"--k must be positive integers (got {raw!r})")
    return ks


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.3f}"


def format_report(results: Sequence[QuestionResult], summary: dict[str, Any]) -> str:
    ks: list[int] = summary["ks"]
    lines: list[str] = []
    lines.append("Per-question rank of first expected file")
    lines.append("-" * 72)
    for r in results:
        if r.question.negative:
            verdict = "abstained" if r.abstained else f"returned {len(r.ranked_paths)} hit(s)"
            lines.append(f"[neg] {verdict:<10} {r.question.q}")
        else:
            rank = "miss" if r.rank is None else str(r.rank)
            lines.append(f"{rank:>5}  {r.question.q}")
    lines.append("")

    headers = ["Group", "N", *[f"Recall@{k}" for k in ks], "MRR", "Abstain"]
    rows: list[list[str]] = []

    def add(name: str, m: dict[str, Any]) -> None:
        rows.append(
            [
                name,
                f"{m['n_answerable']}+{m['n_negative']}neg",
                *[_fmt(m["recall"][str(k)]) for k in ks],
                _fmt(m["mrr"]),
                _fmt(m["abstain_rate"]),
            ]
        )

    add("overall", summary["overall"])
    for tag, m in summary["by_tag"].items():
        add(f"tag:{tag}", m)
    widths = [max(len(headers[c]), *(len(row[c]) for row in rows)) for c in range(len(headers))]
    lines.append(" | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    lines.append("-+-".join("-" * w for w in widths))
    for row in rows:
        lines.append(" | ".join(c.ljust(widths[i]) for i, c in enumerate(row)))
    lines.append("")
    lines.append("N = answerable+negative questions. Abstain = negatives that returned no hits.")
    return "\n".join(lines) + "\n"


def to_json(results: Sequence[QuestionResult], summary: dict[str, Any]) -> str:
    payload = {
        "questions": [
            {
                "q": r.question.q,
                "expect": list(r.question.expect),
                "tags": list(r.question.tags),
                "negative": r.question.negative,
                "rank": r.rank,
                "abstained": r.abstained if r.question.negative else None,
                "ranked_paths": r.ranked_paths,
            }
            for r in results
        ],
        "metrics": summary,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


__all__ = [
    "DEFAULT_KS",
    "Question",
    "QuestionResult",
    "compute_metrics",
    "evaluate",
    "first_expected_rank",
    "format_report",
    "load_questions",
    "parse_ks",
    "summarize",
    "to_json",
]

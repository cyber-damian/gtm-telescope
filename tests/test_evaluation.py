"""Tests for gtm_telescope.evaluation and ``lens eval`` (no embedding server)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import gtm_telescope.cli as cli
from gtm_telescope import evaluation
from gtm_telescope.evaluation import Question


def _q(q: str, expect: list[str], tags: list[str] | None = None) -> Question:
    return Question(q=q, expect=tuple(expect), tags=tuple(tags or ()))


def test_first_expected_rank() -> None:
    assert evaluation.first_expected_rank(["a", "b", "c"], ["c", "b"]) == 2
    assert evaluation.first_expected_rank(["a"], ["z"]) is None
    assert evaluation.first_expected_rank([], ["z"]) is None


def test_metrics_recall_mrr_and_negatives() -> None:
    questions = [
        _q("one", ["a.md"], ["exact"]),
        _q("two", ["b.md"], ["exact"]),
        _q("three", ["c.md"], ["semantic"]),
        _q("neg-quiet", [], ["negative"]),
        _q("neg-noisy", [], ["negative"]),
    ]
    hits = {
        "one": ["a.md", "x.md"],  # rank 1
        "two": ["x.md", "y.md", "b.md"],  # rank 3
        "three": ["x.md"],  # miss
        "neg-quiet": [],
        "neg-noisy": ["x.md"],
    }
    results = evaluation.evaluate(questions, lambda q: hits[q])
    summary = evaluation.summarize(results, [1, 3])
    overall = summary["overall"]
    assert overall["n_answerable"] == 3
    assert overall["n_negative"] == 2
    assert overall["recall"]["1"] == pytest.approx(1 / 3)
    assert overall["recall"]["3"] == pytest.approx(2 / 3)
    assert overall["mrr"] == pytest.approx((1 + 1 / 3 + 0) / 3)
    assert overall["abstain_rate"] == pytest.approx(0.5)
    assert summary["by_tag"]["exact"]["recall"]["1"] == pytest.approx(0.5)
    assert summary["by_tag"]["semantic"]["mrr"] == 0
    assert summary["by_tag"]["negative"]["recall"]["3"] is None


def test_any_expected_file_counts_and_paths_normalized() -> None:
    results = evaluation.evaluate(
        [_q("q", ["a.md", "b.md"])], lambda _q: ["./z.md", "b.md"]
    )
    assert results[0].rank == 2


def test_load_questions_and_validation(tmp_path: Path) -> None:
    f = tmp_path / "q.yaml"
    f.write_text(
        '- q: "hi"\n  expect: ["a.md"]\n  tags: [x]\n- q: "none"\n  expect: []\n',
        encoding="utf-8",
    )
    qs = evaluation.load_questions(f)
    assert [q.negative for q in qs] == [False, True]
    assert qs[0].tags == ("x",)

    f.write_text('- q: "no expect"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="expect"):
        evaluation.load_questions(f)
    f.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no questions"):
        evaluation.load_questions(f)


def test_shipped_eval_file_is_valid() -> None:
    path = Path(__file__).resolve().parent.parent / "eval" / "questions.yaml"
    qs = evaluation.load_questions(path)
    assert 10 <= len(qs) <= 20
    assert any(q.negative for q in qs)
    marker = [q for q in qs if "LEXICAL_ONLY_MARKER_7f4c2b91" in q.q]
    assert marker and marker[0].expect == ("lexical-only/marker.txt",)


def test_parse_ks() -> None:
    assert evaluation.parse_ks("10, 3,3") == [3, 10]
    with pytest.raises(ValueError):
        evaluation.parse_ks("a,b")
    with pytest.raises(ValueError):
        evaluation.parse_ks("0")


def test_report_and_json_shapes() -> None:
    results = evaluation.evaluate(
        [_q("hit", ["a.md"], ["t"]), _q("nothing", [], ["negative"])],
        lambda q: ["a.md"] if q == "hit" else [],
    )
    summary = evaluation.summarize(results, [3, 10])
    text = evaluation.format_report(results, summary)
    assert "Recall@3" in text and "Recall@10" in text and "MRR" in text
    assert "tag:t" in text and "abstained" in text
    payload = json.loads(evaluation.to_json(results, summary))
    assert payload["metrics"]["overall"]["recall"]["3"] == 1.0
    assert payload["questions"][0]["rank"] == 1
    assert payload["questions"][1]["negative"] is True


def test_cmd_eval_uses_hybrid_rows_and_exits_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = tmp_path / "data"
    (data / "sub").mkdir(parents=True)
    qfile = tmp_path / "q.yaml"
    qfile.write_text(
        '- q: "alpha"\n  expect: ["sub/a.md"]\n- q: "beta"\n  expect: ["sub/zzz.md"]\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "load_config", lambda _p: ({}, {"data_dir": str(data)}))
    monkeypatch.setattr(cli, "config_path", lambda: tmp_path / "config.yaml")
    monkeypatch.setattr(cli, "data_storage_dir", lambda _s=None: data)
    monkeypatch.setattr(cli, "_require_calibrated_vector_cache", lambda _d: None)
    monkeypatch.setattr(cli, "_require_markdown_or_text_corpus", lambda _d: None)

    def fake_rows(q: str, _storage: object = None) -> tuple[list[cli._MergedRow], float]:
        row = cli._MergedRow(
            file_display="x",
            line=1,
            source="semantic",
            score=0.5,
            detected="t",
            path_key=str((data / "sub" / "a.md").resolve()),
        )
        return ([row] if q == "alpha" else [row]), 0.0

    monkeypatch.setattr(cli, "hybrid_search_rows", fake_rows)
    cli.cmd_eval(["--file", str(qfile), "--k", "1,3", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert out["metrics"]["ks"] == [1, 3]
    assert out["metrics"]["overall"]["recall"]["1"] == 0.5
    assert [q["rank"] for q in out["questions"]] == [1, None]

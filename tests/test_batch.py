"""Tests for CSV/XLSX batch processing in gtm_telescope.batch."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from gtm_telescope import batch
from gtm_telescope.render import Citation, RenderResult


def result_for(q: str) -> RenderResult:
    return RenderResult(
        answer=f"ans:{q}",
        citations=[
            Citation(1, "a.md", 1, 4, "2020-01-01", True),
            Citation(2, "b.md", 7, 7, "2026-01-01", False),
        ],
        confidence="medium",
    )


def write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)


def read_csv(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


def test_pick_question_column() -> None:
    assert batch.pick_question_column(["id", "Requirement", "owner"], None) == 1
    assert batch.pick_question_column(["id", "text"], None) == 0
    assert batch.pick_question_column(["id", "text"], "TEXT") == 1
    with pytest.raises(batch.BatchError):
        batch.pick_question_column(["id"], "nope")


def test_csv_roundtrip_passthrough_and_columns(tmp_path: Path) -> None:
    src = tmp_path / "q.csv"
    write_csv(src, [["id", "Question", "owner"], ["1", "Do you, support SSO?", "ann"]])
    out = batch.run_batch(src, answer=result_for, progress=open_null())
    assert out == tmp_path / "q.answers.csv"
    rows = read_csv(out)
    assert rows[0] == ["id", "Question", "owner", *batch.OUTPUT_COLUMNS]
    assert rows[1][:3] == ["1", "Do you, support SSO?", "ann"]
    assert rows[1][3] == "ans:Do you, support SSO?"
    assert rows[1][4] == "a.md:1-4; b.md:7"
    assert rows[1][5] == "medium"
    assert rows[1][6] == "a.md"


def open_null():
    import io

    return io.StringIO()


def test_resume_skips_answered_rows(tmp_path: Path) -> None:
    src = tmp_path / "q.csv"
    write_csv(src, [["question"], ["one"], ["two"], ["three"]])
    asked: list[str] = []

    def flaky(q: str) -> RenderResult:
        asked.append(q)
        if q == "two":
            raise KeyboardInterrupt  # simulated interruption
        return result_for(q)

    with pytest.raises(KeyboardInterrupt):
        batch.run_batch(src, answer=flaky, progress=open_null())
    partial = read_csv(tmp_path / "q.answers.csv")
    assert partial[1][1] == "ans:one" and partial[2][1] == ""

    asked.clear()
    batch.run_batch(src, answer=lambda q: (asked.append(q), result_for(q))[1], progress=open_null())
    assert asked == ["two", "three"]
    final = read_csv(tmp_path / "q.answers.csv")
    assert [r[1] for r in final[1:]] == ["ans:one", "ans:two", "ans:three"]


def test_resume_refuses_mismatched_output(tmp_path: Path) -> None:
    src = tmp_path / "q.csv"
    write_csv(src, [["question"], ["one"]])
    write_csv(tmp_path / "q.answers.csv", [["other"], ["x"]])
    with pytest.raises(batch.BatchError, match="refusing to resume"):
        batch.run_batch(src, answer=result_for, progress=open_null())


def test_blank_question_rows_pass_through(tmp_path: Path) -> None:
    src = tmp_path / "q.csv"
    write_csv(src, [["question", "n"], ["", "1"], ["real", "2"]])
    asked: list[str] = []
    batch.run_batch(
        src,
        answer=lambda q: (asked.append(q), result_for(q))[1],
        progress=open_null(),
    )
    assert asked == ["real"]
    rows = read_csv(tmp_path / "q.answers.csv")
    assert rows[1][:2] == ["", "1"] and rows[1][2] == ""


def test_custom_out_and_progress_to_stream(tmp_path: Path) -> None:
    import io

    src = tmp_path / "q.csv"
    write_csv(src, [["question"], ["one"], ["two"]])
    buf = io.StringIO()
    out = batch.run_batch(src, answer=result_for, out_path=tmp_path / "o" / "x.csv", progress=buf)
    assert out.is_file()
    assert "[1/2] one" in buf.getvalue() and "[2/2] two" in buf.getvalue()


def test_missing_file_and_empty_file(tmp_path: Path) -> None:
    with pytest.raises(batch.BatchError, match="not found"):
        batch.read_table(tmp_path / "nope.csv")
    empty = tmp_path / "e.csv"
    empty.write_text("")
    with pytest.raises(batch.BatchError, match="no rows"):
        batch.read_table(empty)


def test_short_rows_are_padded(tmp_path: Path) -> None:
    src = tmp_path / "q.csv"
    src.write_text("question,owner\nonly question\n", encoding="utf-8")
    headers, rows = batch.read_table(src)
    assert rows == [["only question", ""]]
    assert headers == ["question", "owner"]


def test_xlsx_roundtrip_and_sheet(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    src = tmp_path / "q.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Other"
    ws.append(["nope"])
    ws2 = wb.create_sheet("RFP")
    ws2.append(["Ref", "Requirement"])
    ws2.append([7, "Support SAML?"])
    wb.save(src)

    out = batch.run_batch(src, answer=result_for, sheet="RFP", progress=open_null())
    assert out.name == "q.answers.xlsx"
    headers, rows = batch.read_table(out, "RFP")
    assert headers == ["Ref", "Requirement", *batch.OUTPUT_COLUMNS]
    assert rows[0][0] == 7 and rows[0][2] == "ans:Support SAML?"
    with pytest.raises(batch.BatchError, match="sheet"):
        batch.read_table(src, "Missing")


def test_xlsx_without_openpyxl_gives_install_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *a: object, **k: object):
        if name == "openpyxl":
            raise ImportError("no openpyxl")
        return real_import(name, *a, **k)

    (tmp_path / "q.xlsx").write_bytes(b"x")
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(batch.BatchError, match=r"\.\[batch\]"):
        batch.read_table(tmp_path / "q.xlsx")

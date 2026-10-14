"""Tests for the LLM wrapper and `lens render` argument/flow handling in cli."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest

import gtm_telescope.cli as cli


def test_chat_completion_rejects_plain_http_non_local() -> None:
    with pytest.raises(SystemExit):
        cli.chat_completion(
            "http://evil.example.com/v1",
            "m",
            "k",
            "hi",
        )


def test_chat_completion_parses_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"choices": [{"message": {"content": "hello from model"}}]}

    class _Resp:
        def read(self) -> bytes:
            return json.dumps(payload).encode("utf-8")

        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

    monkeypatch.setattr(cli, "_ts", lambda: cli.load_telescope_settings(""))
    with patch.object(cli.urllib.request, "urlopen", return_value=_Resp()):
        out = cli.chat_completion("https://api.example.com/v1", "m", "k", "user text")
    assert out == "hello from model"


def test_chat_completion_http_error_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.error

    err = urllib.error.HTTPError(
        "url",
        500,
        "Internal",
        {},
        BytesIO(b'{"error":"bad"}'),
    )
    monkeypatch.setattr(cli, "_ts", lambda: cli.load_telescope_settings(""))
    with (
        patch.object(cli.urllib.request, "urlopen", side_effect=err),
        pytest.raises(SystemExit),
    ):
        cli.chat_completion("https://api.example.com/v1", "m", "k", "x")


# --- render arg parsing ----------------------------------------------------------------


def test_parse_render_args_question_and_json() -> None:
    ra = cli._parse_render_args(["--json", "does", "it", "do", "SSO?"])
    assert ra.question == "does it do SSO?" and ra.json_out and ra.batch is None


def test_parse_render_args_batch_flags() -> None:
    ra = cli._parse_render_args(
        ["--batch", "q.csv", "--question-col", "Req", "--out", "o.csv", "--sheet", "S"]
    )
    assert (ra.batch, ra.question_col, ra.out, ra.sheet) == ("q.csv", "Req", "o.csv", "S")
    assert ra.question == ""


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--bogus", "q"],
        ["--batch"],
        ["--batch", "q.csv", "extra question"],
        ["--batch", "q.csv", "--json"],
        ["question", "--out", "x.csv"],
    ],
)
def test_parse_render_args_errors(argv: list[str]) -> None:
    with pytest.raises(ValueError):
        cli._parse_render_args(argv)


# --- cmd_render flow -------------------------------------------------------------------


def _prep_render(
    monkeypatch: pytest.MonkeyPatch, *, base_url: str, hits: list | None = None
) -> list[str]:
    """Stub config, corpus checks, retrieval and the LLM; returns the list of prompts sent."""
    from gtm_telescope.retrieval import Hit

    prompts: list[str] = []
    hit = Hit("a.md", 3, 9, "H", "body text", 0.05, 0.6, None, "dual", "2026-01-01", None)
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda _p: (
            {"base_url": base_url, "model": "m", "api_key_env": "TEST_LLM_KEY"},
            {"data_dir": "."},
        ),
    )
    monkeypatch.setattr(cli, "config_path", lambda: cli.Path("config.yaml"))
    monkeypatch.setattr(cli, "data_storage_dir", lambda _s: cli.Path("."))
    monkeypatch.setattr(cli, "_require_calibrated_vector_cache", lambda _d: None)
    monkeypatch.setattr(cli, "_require_markdown_or_text_corpus", lambda _d: None)
    monkeypatch.setattr(cli, "_ts", lambda: cli.load_telescope_settings(""))
    monkeypatch.setattr(
        cli, "retrieve_hits", lambda _q, _s=None: ([hit] if hits is None else hits, 0.01)
    )

    def fake_llm(_b: str, _m: str, _k: str, user: str) -> str:
        prompts.append(user)
        return "Yes, supported [1]."

    monkeypatch.setattr(cli, "chat_completion", fake_llm)
    return prompts


def test_render_local_url_needs_no_api_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("TEST_LLM_KEY", raising=False)
    prompts = _prep_render(monkeypatch, base_url="http://localhost:11434/v1")
    cli.cmd_render(["does", "it", "work?"])
    out = capsys.readouterr().out
    assert "Yes, supported [1]." in out
    assert "Sources:\n[1] a.md:3-9" in out
    assert "does it work?" in prompts[0] and "[1] a.md:3-9" in prompts[0]


def test_render_remote_url_requires_api_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("TEST_LLM_KEY", raising=False)
    _prep_render(monkeypatch, base_url="https://api.openai.com/v1")
    with pytest.raises(SystemExit) as e:
        cli.cmd_render(["q"])
    assert e.value.code == 1
    assert "set TEST_LLM_KEY" in capsys.readouterr().err


def test_render_rejects_bad_base_url_before_retrieval(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _prep_render(monkeypatch, base_url="http://example.com/v1")

    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError("retrieval must not run")

    monkeypatch.setattr(cli, "retrieve_hits", boom)
    with pytest.raises(SystemExit):
        cli.cmd_render(["q"])
    assert "https://" in capsys.readouterr().err


def test_render_json_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _prep_render(monkeypatch, base_url="http://localhost:11434/v1")
    cli.cmd_render(["--json", "q"])
    d = json.loads(capsys.readouterr().out)
    assert d["citations"][0]["path"] == "a.md"
    assert d["confidence"] in {"high", "medium", "low", "not_covered"}


def test_render_no_hits_does_not_call_llm(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    prompts = _prep_render(monkeypatch, base_url="http://localhost:11434/v1", hits=[])
    cli.cmd_render(["q"])
    assert "Not covered in the docs" in capsys.readouterr().out
    assert prompts == []


def test_render_batch_end_to_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import csv

    _prep_render(monkeypatch, base_url="http://localhost:11434/v1")
    src = tmp_path / "q.csv"
    src.write_text("id,question\n1,does it work?\n", encoding="utf-8")
    cli.cmd_render(["--batch", str(src)])
    rows = list(csv.reader((tmp_path / "q.answers.csv").open(encoding="utf-8")))
    assert rows[0][-4:] == ["answer", "citations", "confidence", "stale_sources"]
    assert "Yes, supported" in rows[1][2] and rows[1][3] == "a.md:3-9"

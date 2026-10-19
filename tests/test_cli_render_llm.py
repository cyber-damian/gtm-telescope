"""Tests for the LLM wrapper and `lens render` argument/flow handling in cli."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest

import gtm_telescope.cli as cli
from gtm_telescope import llm as llm_mod
from gtm_telescope.prompts import system_prompt

KEY_ENVS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "MISTRAL_API_KEY", "LLM_PROVIDER")


def _openai_provider() -> llm_mod.Provider:
    return llm_mod.select_provider({"OPENAI_API_KEY": "k"}, {"openai": "m"})


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
    with patch.object(llm_mod.urllib.request, "urlopen", return_value=_Resp()):
        out = cli.chat_completion(_openai_provider(), "SYS", "user text", max_tokens=10)
    assert out == "hello from model"


def test_chat_completion_http_error_exits(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import urllib.error

    err = urllib.error.HTTPError("url", 500, "Internal", {}, BytesIO(b'{"error":"bad"}'))  # type: ignore[arg-type]
    monkeypatch.setattr(cli, "_ts", lambda: cli.load_telescope_settings(""))
    with (
        patch.object(llm_mod.urllib.request, "urlopen", side_effect=err),
        pytest.raises(SystemExit),
    ):
        cli.chat_completion(_openai_provider(), "SYS", "x", max_tokens=10)
    assert "error: openai: LLM HTTP 500" in capsys.readouterr().err


# --- render arg parsing ----------------------------------------------------------------


def test_parse_render_args_question_and_json() -> None:
    ra = cli._parse_render_args(["--json", "does", "it", "do", "SSO?"])
    assert ra.question == "does it do SSO?" and ra.json_out and ra.batch is None
    assert ra.mode == "short"


@pytest.mark.parametrize("mode", ["short", "story", "explain"])
def test_parse_render_args_mode(mode: str) -> None:
    ra = cli._parse_render_args([f"--{mode}", "q"])
    assert ra.mode == mode and ra.question == "q"


def test_parse_render_args_repeated_same_mode_ok() -> None:
    assert cli._parse_render_args(["--story", "q", "--story"]).mode == "story"


def test_parse_render_args_batch_flags() -> None:
    ra = cli._parse_render_args(
        ["--explain", "--batch", "q.csv", "--question-col", "Req", "--out", "o.csv", "--sheet", "S"]
    )
    assert (ra.batch, ra.question_col, ra.out, ra.sheet) == ("q.csv", "Req", "o.csv", "S")
    assert ra.question == "" and ra.mode == "explain"


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--bogus", "q"],
        ["--batch"],
        ["--batch", "q.csv", "extra question"],
        ["--batch", "q.csv", "--json"],
        ["question", "--out", "x.csv"],
        ["--short", "--story", "q"],
        ["--explain", "--short", "q"],
    ],
)
def test_parse_render_args_errors(argv: list[str]) -> None:
    with pytest.raises(ValueError):
        cli._parse_render_args(argv)


# --- cmd_render flow -------------------------------------------------------------------


def _prep_render(
    monkeypatch: pytest.MonkeyPatch,
    *,
    env: dict[str, str] | None = None,
    hits: list | None = None,
) -> list[tuple[llm_mod.Provider, str, str]]:
    """Stub config, corpus checks, retrieval and the LLM; returns ``(provider, system, user)``."""
    from gtm_telescope.retrieval import Hit

    for name in KEY_ENVS:
        monkeypatch.delenv(name, raising=False)
    for name, value in ({"OPENAI_API_KEY": "k"} if env is None else env).items():
        monkeypatch.setenv(name, value)

    calls: list[tuple[llm_mod.Provider, str, str]] = []
    hit = Hit("a.md", 3, 9, "H", "body text", 0.05, 0.6, None, "dual", "2026-01-01", None)
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda _p: ({"openai_model": "gpt-test", "max_tokens": "77"}, {"data_dir": "."}),
    )
    monkeypatch.setattr(cli, "config_path", lambda: cli.Path("config.yaml"))
    monkeypatch.setattr(cli, "data_storage_dir", lambda _s: cli.Path("."))
    monkeypatch.setattr(cli, "_require_calibrated_vector_cache", lambda _d: None)
    monkeypatch.setattr(cli, "_require_markdown_or_text_corpus", lambda _d: None)
    monkeypatch.setattr(cli, "_ts", lambda: cli.load_telescope_settings(""))
    monkeypatch.setattr(
        cli, "retrieve_hits", lambda _q, _s=None: ([hit] if hits is None else hits, 0.01)
    )

    def fake_llm(provider: llm_mod.Provider, system: str, user: str, *, max_tokens: int) -> str:
        assert max_tokens == 77
        calls.append((provider, system, user))
        return "Yes, supported [1]."

    monkeypatch.setattr(cli, "chat_completion", fake_llm)
    return calls


def test_render_default_short_mode(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _prep_render(monkeypatch)
    cli.cmd_render(["does", "it", "work?"])
    out = capsys.readouterr().out
    assert "SAY THIS" in out
    assert "Yes, supported." in out
    assert "IF THEY ASK WHERE THIS IS FROM" in out
    assert "a.md:3-9" in out
    provider, system, user = calls[0]
    assert (provider.name, provider.model) == ("openai", "gpt-test")
    assert system == system_prompt("short")
    assert "does it work?" in user and "[1] a.md:3-9" in user


@pytest.mark.parametrize("mode", ["short", "story", "explain"])
def test_render_mode_selects_template(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    calls = _prep_render(monkeypatch)
    cli.cmd_render([f"--{mode}", "q"])
    assert calls[0][1] == system_prompt(mode)


def test_render_picks_provider_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _prep_render(monkeypatch, env={"MISTRAL_API_KEY": "m", "ANTHROPIC_API_KEY": "a"})
    cli.cmd_render(["q"])
    assert calls[0][0].name == "anthropic"


def test_render_without_key_fails_before_retrieval(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _prep_render(monkeypatch, env={})

    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError("retrieval must not run")

    monkeypatch.setattr(cli, "retrieve_hits", boom)
    with pytest.raises(SystemExit) as e:
        cli.cmd_render(["q"])
    assert e.value.code == 1
    err = capsys.readouterr().err
    assert "ANTHROPIC_API_KEY" in err and "OPENAI_API_KEY" in err and "MISTRAL_API_KEY" in err


def test_render_json_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _prep_render(monkeypatch)
    cli.cmd_render(["--explain", "--json", "q"])
    d = json.loads(capsys.readouterr().out)
    assert d["citations"][0]["path"] == "a.md"
    assert d["confidence"] in {"high", "medium", "low", "not_covered"}


def test_render_no_hits_does_not_call_llm(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _prep_render(monkeypatch, hits=[])
    cli.cmd_render(["q"])
    out = capsys.readouterr().out
    assert "Not covered in the docs" in out
    assert "SAY THIS" in out
    assert calls == []


def test_render_batch_end_to_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import csv

    calls = _prep_render(monkeypatch)
    src = tmp_path / "q.csv"
    src.write_text("id,question\n1,does it work?\n", encoding="utf-8")
    cli.cmd_render(["--story", "--batch", str(src)])
    rows = list(csv.reader((tmp_path / "q.answers.csv").open(encoding="utf-8")))
    assert rows[0][-4:] == ["answer", "citations", "confidence", "stale_sources"]
    assert "Yes, supported" in rows[1][2] and rows[1][3] == "a.md:3-9"
    assert calls[0][1] == system_prompt("story")

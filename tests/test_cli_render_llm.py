"""Tests for LLM HTTP path in cli.chat_completion."""

from __future__ import annotations

import json
from io import BytesIO
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

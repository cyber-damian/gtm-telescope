"""Tests for the OpenAI-compatible client in gtm_telescope.llm."""

from __future__ import annotations

import json
import urllib.error
from io import BytesIO
from typing import Any
from unittest.mock import patch

import pytest

from gtm_telescope import llm


class _Resp:
    def __init__(self, payload: Any) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


OK = {"choices": [{"message": {"content": "hi [1]"}}]}


@pytest.mark.parametrize(
    "url,expected",
    [
        ("http://localhost:11434/v1", True),
        ("http://127.0.0.1:1234/v1", True),
        ("https://localhost/v1", True),
        ("https://api.openai.com/v1", False),
        ("http://localhost.evil.example.com/v1", False),
        ("http://127.0.0.1.evil.example.com/v1", False),
    ],
)
def test_is_local_url_uses_hostname(url: str, expected: bool) -> None:
    assert llm.is_local_url(url) is expected
    assert llm.api_key_required(url) is (not expected)


def test_validate_base_url_rules() -> None:
    llm.validate_base_url("https://api.openai.com/v1")
    llm.validate_base_url("http://localhost:11434/v1")
    llm.validate_base_url("http://127.0.0.1:8080/v1/")
    for bad in ("http://example.com/v1", "http://localhost.evil.com/v1", "ftp://x"):
        with pytest.raises(llm.LLMError):
            llm.validate_base_url(bad)


def test_no_authorization_header_without_key() -> None:
    with patch.object(llm.urllib.request, "urlopen", return_value=_Resp(OK)) as m:
        out = llm.chat_completion("http://localhost:11434/v1", "llama3.1:8b", "", "q")
    assert out == "hi [1]"
    req = m.call_args.args[0]
    assert "Authorization" not in req.headers
    assert req.full_url == "http://localhost:11434/v1/chat/completions"
    body = json.loads(req.data)
    assert body["model"] == "llama3.1:8b"
    assert body["messages"][0]["content"] == llm.SYSTEM_PROMPT
    assert body["messages"][1] == {"role": "user", "content": "q"}


def test_authorization_header_with_key() -> None:
    with patch.object(llm.urllib.request, "urlopen", return_value=_Resp(OK)) as m:
        llm.chat_completion("https://api.example.com/v1", "m", "secret", "q")
    assert m.call_args.args[0].get_header("Authorization") == "Bearer secret"


def test_http_error_becomes_llm_error() -> None:
    err = urllib.error.HTTPError("u", 500, "x", {}, BytesIO(b'{"error":"bad"}'))  # type: ignore[arg-type]
    with (
        patch.object(llm.urllib.request, "urlopen", side_effect=err),
        pytest.raises(llm.LLMError, match="HTTP 500"),
    ):
        llm.chat_completion("https://api.example.com/v1", "m", "k", "x")


def test_connection_refused_becomes_llm_error() -> None:
    with (
        patch.object(
            llm.urllib.request, "urlopen", side_effect=urllib.error.URLError("refused")
        ),
        pytest.raises(llm.LLMError, match="request failed"),
    ):
        llm.chat_completion("http://localhost:11434/v1", "m", "", "x")


def test_bad_shape_becomes_llm_error() -> None:
    with (
        patch.object(llm.urllib.request, "urlopen", return_value=_Resp({"choices": []})),
        pytest.raises(llm.LLMError, match="unexpected"),
    ):
        llm.chat_completion("http://localhost:11434/v1", "m", "", "x")


def test_remote_http_rejected_before_request() -> None:
    with (
        patch.object(llm.urllib.request, "urlopen") as m,
        pytest.raises(llm.LLMError, match="https://"),
    ):
        llm.chat_completion("http://example.com/v1", "m", "k", "x")
    m.assert_not_called()

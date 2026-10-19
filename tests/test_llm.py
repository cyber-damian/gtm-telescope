"""Tests for provider selection and the hosted LLM clients in gtm_telescope.llm."""

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


OPENAI_OK = {"choices": [{"message": {"content": "hi [1]"}}]}
ANTHROPIC_OK = {"content": [{"type": "text", "text": "hello "}, {"type": "text", "text": "[1]"}]}

ALL_KEYS = {"ANTHROPIC_API_KEY": "a", "OPENAI_API_KEY": "o", "MISTRAL_API_KEY": "m"}


# --- provider selection ----------------------------------------------------------------


@pytest.mark.parametrize(
    "env,expected",
    [
        (ALL_KEYS, "anthropic"),
        ({"OPENAI_API_KEY": "o", "MISTRAL_API_KEY": "m"}, "openai"),
        ({"MISTRAL_API_KEY": "m"}, "mistral"),
        ({"ANTHROPIC_API_KEY": "  ", "MISTRAL_API_KEY": "m"}, "mistral"),
    ],
)
def test_select_provider_order(env: dict[str, str], expected: str) -> None:
    assert llm.select_provider(env).name == expected


def test_select_provider_override() -> None:
    p = llm.select_provider({**ALL_KEYS, "LLM_PROVIDER": "Mistral"})
    assert (p.name, p.api_key, p.base_url) == ("mistral", "m", "https://api.mistral.ai/v1")


def test_select_provider_override_with_empty_key_fails() -> None:
    with pytest.raises(llm.LLMError, match="OPENAI_API_KEY is empty"):
        llm.select_provider({"ANTHROPIC_API_KEY": "a", "LLM_PROVIDER": "openai"})


def test_select_provider_unknown_override_fails() -> None:
    with pytest.raises(llm.LLMError, match="not a known provider"):
        llm.select_provider({**ALL_KEYS, "LLM_PROVIDER": "ollama"})


def test_select_provider_no_key_fails() -> None:
    with pytest.raises(llm.LLMError, match="no LLM API key set"):
        llm.select_provider({"LLM_PROVIDER": ""})


def test_select_provider_models_from_config() -> None:
    assert llm.select_provider(ALL_KEYS, {"anthropic": "claude-x"}).model == "claude-x"
    default = llm.select_provider(ALL_KEYS, {"anthropic": ""}).model
    assert default == llm.PROVIDERS[0].default_model


# --- clients ---------------------------------------------------------------------------


def _provider(name: str) -> llm.Provider:
    env = {"LLM_PROVIDER": name, f"{name.upper()}_API_KEY": "secret"}
    return llm.select_provider(env, {name: "model-x"})


def test_anthropic_request_shape_and_parse() -> None:
    with patch.object(llm.urllib.request, "urlopen", return_value=_Resp(ANTHROPIC_OK)) as m:
        out = llm.complete(_provider("anthropic"), "SYS", "q", max_tokens=321)
    assert out == "hello [1]"
    req = m.call_args.args[0]
    assert req.full_url == "https://api.anthropic.com/v1/messages"
    assert req.get_header("X-api-key") == "secret"
    assert req.get_header("Anthropic-version") == llm.ANTHROPIC_VERSION
    assert req.get_header("Authorization") is None
    body = json.loads(req.data)
    assert body == {
        "model": "model-x",
        "max_tokens": 321,
        "system": "SYS",
        "messages": [{"role": "user", "content": "q"}],
    }


def test_anthropic_bad_shape_becomes_llm_error() -> None:
    with (
        patch.object(llm.urllib.request, "urlopen", return_value=_Resp({"content": []})),
        pytest.raises(llm.LLMError, match="unexpected"),
    ):
        llm.complete(_provider("anthropic"), "SYS", "q")


@pytest.mark.parametrize(
    "name,url,token_field",
    [
        ("openai", "https://api.openai.com/v1/chat/completions", "max_completion_tokens"),
        ("mistral", "https://api.mistral.ai/v1/chat/completions", "max_tokens"),
    ],
)
def test_openai_format_providers(name: str, url: str, token_field: str) -> None:
    with patch.object(llm.urllib.request, "urlopen", return_value=_Resp(OPENAI_OK)) as m:
        out = llm.complete(_provider(name), "SYS", "q", max_tokens=99)
    assert out == "hi [1]"
    req = m.call_args.args[0]
    assert req.full_url == url
    assert req.get_header("Authorization") == "Bearer secret"
    body = json.loads(req.data)
    assert body["model"] == "model-x" and body[token_field] == 99
    assert body["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "q"},
    ]


def test_http_error_becomes_llm_error() -> None:
    err = urllib.error.HTTPError("u", 500, "x", {}, BytesIO(b'{"error":"bad"}'))  # type: ignore[arg-type]
    with (
        patch.object(llm.urllib.request, "urlopen", side_effect=err),
        pytest.raises(llm.LLMError, match="HTTP 500"),
    ):
        llm.complete(_provider("openai"), "SYS", "x")


def test_connection_refused_becomes_llm_error() -> None:
    with (
        patch.object(
            llm.urllib.request, "urlopen", side_effect=urllib.error.URLError("refused")
        ),
        pytest.raises(llm.LLMError, match="request failed"),
    ):
        llm.complete(_provider("mistral"), "SYS", "x")


def test_openai_bad_shape_becomes_llm_error() -> None:
    with (
        patch.object(llm.urllib.request, "urlopen", return_value=_Resp({"choices": []})),
        pytest.raises(llm.LLMError, match="unexpected"),
    ):
        llm.complete(_provider("openai"), "SYS", "x")


def test_plain_http_rejected_before_request() -> None:
    with (
        patch.object(llm.urllib.request, "urlopen") as m,
        pytest.raises(llm.LLMError, match="https://"),
    ):
        llm.chat_completion("http://example.com/v1", "m", "k", "x", system_prompt="S")
    m.assert_not_called()

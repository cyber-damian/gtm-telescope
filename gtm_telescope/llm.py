"""Hosted LLM clients for ``lens render`` (stdlib only): Anthropic, OpenAI and Mistral.

The provider is chosen from whichever API key is set (see :func:`select_provider`). OpenAI and
Mistral speak the OpenAI chat-completions format; Anthropic uses its Messages API. All
endpoints must use HTTPS.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 1024


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    key_env: str
    base_url: str
    default_model: str


# Selection order when several keys are set (LLM_PROVIDER overrides it).
PROVIDERS: tuple[ProviderSpec, ...] = (
    ProviderSpec(
        "anthropic", "ANTHROPIC_API_KEY", "https://api.anthropic.com/v1", "claude-sonnet-4-5"
    ),
    ProviderSpec("openai", "OPENAI_API_KEY", "https://api.openai.com/v1", "gpt-4.1-mini"),
    ProviderSpec(
        "mistral", "MISTRAL_API_KEY", "https://api.mistral.ai/v1", "mistral-small-latest"
    ),
)
PROVIDER_ENV = "LLM_PROVIDER"


@dataclass(frozen=True)
class Provider:
    name: str
    api_key: str
    model: str
    base_url: str


class LLMError(RuntimeError):
    """The LLM request could not be made or its response could not be understood."""


def validate_base_url(base_url: str) -> None:
    base = base_url.rstrip("/")
    if not base.lower().startswith("https://"):
        raise LLMError(f"LLM endpoint must use https:// (got {base!r}).")


def _key_names() -> str:
    return ", ".join(p.key_env for p in PROVIDERS)


def select_provider(env: Mapping[str, str], models: Mapping[str, str] | None = None) -> Provider:
    """Pick the provider from ``env`` (first populated key, or ``LLM_PROVIDER``).

    ``models`` maps provider name to model id (from config.yaml); missing names use defaults.
    """
    models = models or {}
    by_name = {p.name: p for p in PROVIDERS}
    forced = env.get(PROVIDER_ENV, "").strip().lower()
    if forced:
        spec = by_name.get(forced)
        if spec is None:
            raise LLMError(
                f"{PROVIDER_ENV}={forced!r} is not a known provider "
                f"(expected one of {', '.join(by_name)})."
            )
        if not env.get(spec.key_env, "").strip():
            raise LLMError(f"{PROVIDER_ENV}={forced} but {spec.key_env} is empty.")
        candidates: tuple[ProviderSpec, ...] = (spec,)
    else:
        candidates = tuple(p for p in PROVIDERS if env.get(p.key_env, "").strip())
    if not candidates:
        raise LLMError(
            f"no LLM API key set. Set one of {_key_names()} in .env (see .env.example)."
        )
    spec = candidates[0]
    return Provider(
        name=spec.name,
        api_key=env[spec.key_env].strip(),
        model=(models.get(spec.name) or "").strip() or spec.default_model,
        base_url=spec.base_url,
    )


def _post_json(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float) -> Any:
    validate_base_url(url)
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", **headers},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        snippet = e.read()[:800].decode("utf-8", errors="replace")
        raise LLMError(f"LLM HTTP {e.code}\n{snippet}") from e
    except urllib.error.URLError as e:
        raise LLMError(f"LLM request failed: {e}") from e
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise LLMError(f"invalid JSON from LLM: {e}") from e


def chat_completion(
    base_url: str,
    model: str,
    api_key: str,
    user_content: str,
    *,
    system_prompt: str,
    timeout: float = 120.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_tokens_field: str = "max_tokens",
) -> str:
    """OpenAI-format ``POST /chat/completions`` (OpenAI, Mistral); returns the message text.

    OpenAI's newer models only accept ``max_completion_tokens``; Mistral uses ``max_tokens``.
    """
    payload = _post_json(
        f"{base_url.rstrip('/')}/chat/completions",
        {
            "model": model,
            max_tokens_field: max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        },
        {"Authorization": f"Bearer {api_key}"},
        timeout,
    )
    try:
        return str(payload["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError(f"unexpected LLM response shape: {payload!r}") from e


def anthropic_messages(
    base_url: str,
    model: str,
    api_key: str,
    user_content: str,
    *,
    system_prompt: str,
    timeout: float = 120.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Anthropic ``POST /messages``; returns the joined text blocks."""
    payload = _post_json(
        f"{base_url.rstrip('/')}/messages",
        {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_content}],
        },
        {"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION},
        timeout,
    )
    try:
        texts = [b["text"] for b in payload["content"] if b.get("type") == "text"]
    except (KeyError, TypeError, AttributeError) as e:
        raise LLMError(f"unexpected LLM response shape: {payload!r}") from e
    if not texts:
        raise LLMError(f"unexpected LLM response shape: {payload!r}")
    return "".join(texts)


def complete(
    provider: Provider,
    system_prompt: str,
    user_content: str,
    *,
    timeout: float = 120.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Send one system + user turn to ``provider`` and return the reply text."""
    args = (provider.base_url, provider.model, provider.api_key, user_content)
    if provider.name == "anthropic":
        return anthropic_messages(
            *args, system_prompt=system_prompt, timeout=timeout, max_tokens=max_tokens
        )
    return chat_completion(
        *args,
        system_prompt=system_prompt,
        timeout=timeout,
        max_tokens=max_tokens,
        max_tokens_field="max_completion_tokens" if provider.name == "openai" else "max_tokens",
    )


__all__ = [
    "PROVIDERS",
    "PROVIDER_ENV",
    "LLMError",
    "Provider",
    "anthropic_messages",
    "chat_completion",
    "complete",
    "select_provider",
    "validate_base_url",
]

"""OpenAI-compatible chat completion client (stdlib only).

Defaults target a local server (Ollama / LM Studio). API keys are only needed for remote
endpoints; remote endpoints must use HTTPS.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

SYSTEM_PROMPT = (
    "You write answers for sales engineers using ONLY the numbered sources provided in the "
    "user message. Follow these rules exactly:\n"
    "1. Every factual sentence must end with one or more citations such as [1] or [2][3] "
    "that refer only to the provided source numbers. Never cite a number that was not "
    "provided and never invent a source.\n"
    "2. If the sources do not answer the question fully or partly, say so explicitly with "
    "`Not covered in the docs: <what is missing>`. Do not guess and do not use outside "
    "knowledge. If nothing is covered, the whole answer is that one line.\n"
    "3. Do not merge facts from different sources into one claim unless both sources are "
    "cited for it.\n"
    "4. If a cited source is marked (stale), append `(verify: source dated <date>)` to the "
    "claim that relies on it, using the date shown for that source.\n"
    "5. Ignore legal or marketing boilerplate, copyright lines and generic 'contact us' text; "
    "they are not evidence.\n"
    "6. No marketing language. Be concise."
)


class LLMError(RuntimeError):
    """The LLM request could not be made or its response could not be understood."""


def is_local_url(base_url: str) -> bool:
    host = (urllib.parse.urlparse(base_url.strip()).hostname or "").lower()
    return host in LOCAL_HOSTS


def validate_base_url(base_url: str) -> None:
    """Remote endpoints must use https://; plaintext http:// is allowed only for localhost."""
    base = base_url.rstrip("/")
    if base.lower().startswith("https://") or (
        base.lower().startswith("http://") and is_local_url(base)
    ):
        return
    raise LLMError(
        f"llm.base_url must use https:// for remote endpoints (got {base!r}). Set an https:// "
        "URL in config.yaml, or use http://localhost / http://127.0.0.1 for a local server."
    )


def api_key_required(base_url: str) -> bool:
    return not is_local_url(base_url)


def chat_completion(
    base_url: str,
    model: str,
    api_key: str,
    user_content: str,
    *,
    system_prompt: str | None = None,
    timeout: float = 120.0,
) -> str:
    """POST ``/chat/completions``; returns the assistant message text.

    No ``Authorization`` header is sent when ``api_key`` is empty (local servers).
    """
    validate_base_url(base_url)
    url = f"{base_url.rstrip('/')}/chat/completions"
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), method="POST", headers=headers
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        snippet = e.read()[:800].decode("utf-8", errors="replace")
        raise LLMError(f"LLM HTTP {e.code}\n{snippet}") from e
    except urllib.error.URLError as e:
        raise LLMError(f"LLM request failed: {e}") from e
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise LLMError(f"invalid JSON from LLM: {e}") from e
    try:
        return str(payload["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError(f"unexpected LLM response shape: {payload!r}") from e


__all__ = [
    "LOCAL_HOSTS",
    "SYSTEM_PROMPT",
    "LLMError",
    "api_key_required",
    "chat_completion",
    "is_local_url",
    "validate_base_url",
]

"""System prompt templates for ``lens render`` modes (``gtm_telescope/prompt_templates/*.md``)."""

from __future__ import annotations

from functools import cache
from importlib import resources

MODES = ("short", "story", "explain")
DEFAULT_MODE = "short"


@cache
def _template(name: str) -> str:
    return resources.files("gtm_telescope").joinpath("prompt_templates", f"{name}.md").read_text(
        encoding="utf-8"
    ).strip()


def system_prompt(mode: str) -> str:
    """Shared grounding/citation rules followed by the mode's format instructions."""
    if mode not in MODES:
        raise ValueError(f"unknown render mode {mode!r} (expected one of {', '.join(MODES)})")
    return f"{_template('rules')}\n\n{_template(mode)}"


__all__ = ["DEFAULT_MODE", "MODES", "system_prompt"]

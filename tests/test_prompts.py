"""Tests for the lens render prompt templates."""

from __future__ import annotations

import pytest

from gtm_telescope.prompts import DEFAULT_MODE, MODES, system_prompt


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_includes_shared_rules(mode: str) -> None:
    text = system_prompt(mode)
    assert "ONLY the numbered sources" in text
    assert "Not covered in the docs" in text
    assert "(verify: source dated <date>)" in text


def test_modes_differ() -> None:
    prompts = {m: system_prompt(m) for m in MODES}
    assert len(set(prompts.values())) == len(MODES)
    assert "Exactly 2 sentences" in prompts["short"]
    assert "6 to 8 sentences" in prompts["story"]
    assert "no customer story found" in prompts["story"]
    assert "Up to 2 paragraphs" in prompts["explain"]


def test_default_mode_is_short() -> None:
    assert DEFAULT_MODE == "short"


def test_unknown_mode_rejected() -> None:
    with pytest.raises(ValueError, match="unknown render mode"):
        system_prompt("poem")

"""Tests for loading secrets from repo .env via ensure_dotenv_loaded."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import gtm_telescope.cli as cli


def _minimal_repo(root: Path, *, env_key: str = "FOO_API_KEY") -> None:
    (root / "pyproject.toml").write_text(
        '[project]\nname = "gtm-telescope"\nversion = "0"\n',
        encoding="utf-8",
    )
    (root / "config.yaml").write_text(
        f"llm:\n"
        f'  base_url: "https://api.example.com/v1"\n'
        f"  model: m\n"
        f"  api_key_env: {env_key}\n"
        f'  env_file: ".env"\n'
        "storage:\n"
        '  data_dir: "data-storage"\n'
        'embedding:\n  server_url: "http://localhost:8089"\n',
        encoding="utf-8",
    )
    (root / "data-storage").mkdir()
    (root / "data-storage" / "a.md").write_text("# x\n", encoding="utf-8")


def test_ensure_dotenv_loads_vars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _minimal_repo(tmp_path, env_key="DOTENV_TEST_KEY")
    (tmp_path / ".env").write_text("DOTENV_TEST_KEY=from-dotenv-file\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DOTENV_TEST_KEY", raising=False)
    cli.project_root.cache_clear()
    cli._dotenv_loaded = False
    cli.ensure_dotenv_loaded()
    assert os.environ.get("DOTENV_TEST_KEY") == "from-dotenv-file"
    cli._dotenv_loaded = False
    cli.project_root.cache_clear()

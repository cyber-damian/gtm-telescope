"""Tests for `lens status` (embedding server status line + configured URL)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import gtm_telescope.cli as cli


def test_cmd_status_runs_jina_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "\n".join(
            [
                "llm:",
                '  base_url: "https://api.example.com/v1"',
                '  model: "gpt-test"',
                "  api_key_env: OPENAI_API_KEY",
                "storage:",
                '  data_dir: "data-storage"',
                "embedding:",
                '  server_url: "http://127.0.0.1:8089"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert "serve" in cmd and cmd[-1] == "status"
        return subprocess.CompletedProcess(cmd, 0, stdout="Server is not running\n", stderr="")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(cli, "config_path", lambda: cfg)
    cli.clear_telescope_settings_cache()
    cli.cmd_status()
    out = capsys.readouterr().out
    assert "Server is not running" in out
    assert "Configured embedding server:" in out
    assert "Port (from URL):" in out


def test_embedding_url_port_hint_explicit_port() -> None:
    assert cli._embedding_url_port_hint("http://127.0.0.1:8089") == "8089"


def test_embedding_url_port_hint_https_default() -> None:
    assert "443" in cli._embedding_url_port_hint("https://api.example.com/v1")

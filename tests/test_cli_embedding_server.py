"""Tests for embedding server readiness (no real jina-grep process)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

import gtm_telescope.cli as cli
from gtm_telescope import vector_cache


def test_ensure_embedding_server_ready_noop_when_healthy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    called: dict[str, int] = {"popen": 0}

    def hc(_url: str, timeout: float = 2.0) -> bool:
        return True

    def popen(*_a: object, **_k: object) -> None:
        called["popen"] += 1
        raise AssertionError("should not spawn when healthy")

    monkeypatch.setattr(vector_cache, "health_check", hc)
    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    cli._ensure_embedding_server_ready("http://127.0.0.1:8089", data)
    assert called["popen"] == 0


def test_ensure_embedding_server_ready_spawns_and_waits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    n = [0]

    def hc(_url: str, timeout: float = 2.0) -> bool:
        n[0] += 1
        return n[0] >= 4

    times = iter([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    def mono() -> float:
        return next(times)

    monkeypatch.setattr(vector_cache, "health_check", hc)
    monkeypatch.setattr(cli.time, "monotonic", mono)
    monkeypatch.setattr(cli.time, "sleep", lambda _s: None)
    monkeypatch.setattr(cli.subprocess, "Popen", MagicMock())
    monkeypatch.setattr(cli, "jina_grep_executable", lambda: "/bin/true")
    monkeypatch.setattr(
        cli.vector_cache,
        "embed_batch",
        lambda *_a, **_k: np.array([[1.0, 0.0]], dtype=np.float32),
    )
    s = cli.load_telescope_settings(
        "\nembedding:\n  embed_server_ready_sec: \"30\"\n  embed_http_timeout: \"5\"\n"
    )
    monkeypatch.setattr(cli, "_ts", lambda: s)
    cli._ensure_embedding_server_ready("http://127.0.0.1:8089", data)
    assert n[0] >= 4

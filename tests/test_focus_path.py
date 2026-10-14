"""`lens focus` read path: privacy (network), no corpus embedding, stale/schema guards."""

from __future__ import annotations

import http.client
import json
import socket
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import pytest

import gtm_telescope.cli as cli
from gtm_telescope import vector_cache
from gtm_telescope.config import load_telescope_settings
from tests.conftest import FAKE_DIM, fake_vector

_REAL_EMBED_BATCH = vector_cache.embed_batch  # captured before fixtures patch it

CORPUS = {
    "ops/runbook.md": "# Runbook\n\nPromote the warm standby and update the connection string.\n",
    "old/pricing.md": (
        "---\nupdated: 2023-01-10\n---\n# Pricing\n\nStarter rate-limit is 600 per minute.\n"
    ),
}


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.status = 200
        self.reason = "OK"
        self.headers: dict = {}
        self._body = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_a: object) -> None:
        return None


def _patch_cli(monkeypatch: pytest.MonkeyPatch, data: Path, settings_yaml: str = "") -> None:
    s = load_telescope_settings('retrieval:\n  semantic_threshold: "0.2"\n' + settings_yaml)
    monkeypatch.setattr(cli, "_ts", lambda: s)
    monkeypatch.setattr(cli, "data_storage_dir", lambda _s=None: data)
    monkeypatch.setattr(cli, "rg_executable", lambda: "rg")


def test_focus_only_talks_to_embedding_server_and_never_embeds_corpus(
    monkeypatch: pytest.MonkeyPatch, build_cache
) -> None:
    data = build_cache(CORPUS)
    _patch_cli(monkeypatch, data)

    # From here on use the real embedding client, but with every network primitive recorded.
    monkeypatch.setattr(vector_cache, "embed_batch", _REAL_EMBED_BATCH)
    hosts: list[str] = []
    embedded: list[list[str]] = []

    class FakeConn:
        def __init__(self, host: str, port: int | None = None, **_k: object) -> None:
            hosts.append(host)

        def request(self, method: str, path: str, body: bytes = b"", **_k: object) -> None:
            texts = json.loads(body)["input"]
            embedded.append(texts)
            data = [{"embedding": fake_vector(t).tolist()} for t in texts]
            self._resp = _FakeResponse({"data": data})

        def getresponse(self) -> _FakeResponse:
            return self._resp

        def close(self) -> None:
            return None

    def fake_urlopen(req: urllib.request.Request | str, timeout: float | None = None):
        url = req if isinstance(req, str) else req.full_url
        hosts.append(urllib.parse.urlparse(url).hostname or "")
        return _FakeResponse({})

    def no_sockets(*_a: object, **_k: object) -> None:
        raise AssertionError("unexpected raw socket connection")

    monkeypatch.setattr(http.client, "HTTPConnection", FakeConn)
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(socket.socket, "connect", no_sockets)
    monkeypatch.setattr(
        vector_cache, "_sync_corpus_into_cache", lambda *_a, **_k: pytest.fail("focus synced")
    )

    hits, _elapsed = cli.retrieve_hits("warm standby connection string")
    assert hits and hits[0].path == "ops/runbook.md"
    assert set(hosts) == {"localhost"}  # the configured embedding server, nothing else
    assert embedded == [["warm standby connection string"]]  # only the query was embedded


def test_format_hits_report_shows_lines_heading_date_and_stale(
    monkeypatch: pytest.MonkeyPatch, build_cache, fake_embeddings
) -> None:
    data = build_cache(CORPUS)
    _patch_cli(monkeypatch, data)
    orig = cli.retrieval.retrieve

    def with_fakes(q, d, st, **_k):
        return orig(q, d, st, embed=fake_embeddings, today=date(2026, 10, 14))

    monkeypatch.setattr(cli.retrieval, "retrieve", with_fakes)
    hits, _ = cli.retrieve_hits("rate-limit starter")
    report = cli.format_hits_report(hits)
    assert "Rank | File" in report and "Source" in report
    assert "old/pricing.md" in report
    assert "2023-01-10 ⚠ stale" in report
    assert "· Pricing" in report and "⚠ stale" in report.split("Snippets", 1)[1]


def test_format_hits_report_empty() -> None:
    assert cli.format_hits_report([]) == "(no hits)\n"


def test_require_calibrated_rejects_old_schema(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
) -> None:
    (tmp_path / "a.md").write_text("# x\n\nbody text here\n", encoding="utf-8")
    monkeypatch.setattr(
        vector_cache,
        "cache_info",
        lambda _d: {
            "exists": True,
            "chunk_rows": 5,
            "stale_or_new_files": 0,
            "schema_ok": False,
            "meta": {"schema_version": "1"},
        },
    )
    with pytest.raises(SystemExit):
        cli._require_calibrated_vector_cache(tmp_path)
    assert "older version" in capsys.readouterr().err


def test_validator_ignores_convertible_files_with_sibling_markdown(tmp_path: Path) -> None:
    (tmp_path / "a.pdf").write_bytes(b"%PDF")
    assert cli._first_non_markdown_or_txt_file_under(tmp_path) == tmp_path / "a.pdf"
    (tmp_path / "a.md").write_text("# a\n", encoding="utf-8")
    assert cli._first_non_markdown_or_txt_file_under(tmp_path) is None
    (tmp_path / "b.docx").write_bytes(b"PK")
    assert cli._first_non_markdown_or_txt_file_under(tmp_path) == tmp_path / "b.docx"


def test_validator_skips_tool_dirs_and_bootstrap_marker(tmp_path: Path) -> None:
    for d in (".originals", ".lens-cache"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "x.pdf").write_bytes(b"%PDF")
    (tmp_path / ".gtm_telescope_bootstrap_prefetch_ok").write_text("", encoding="utf-8")
    (tmp_path / "ok.md").write_text("# ok\n", encoding="utf-8")
    assert cli._first_non_markdown_or_txt_file_under(tmp_path) is None


def test_skip_dirs_are_one_shared_constant() -> None:
    from gtm_telescope import corpus

    assert cli._SKIP_CORPUS_DIRS is corpus.SKIP_CORPUS_DIRS
    assert vector_cache._CORPUS_SKIP_DIRS is corpus.SKIP_CORPUS_DIRS
    assert ".originals" in corpus.SKIP_CORPUS_DIRS and ".lens-cache" in corpus.SKIP_CORPUS_DIRS


def test_walk_corpus_skips_originals(tmp_path: Path) -> None:
    (tmp_path / "keep.md").write_text("x", encoding="utf-8")
    (tmp_path / ".originals").mkdir()
    (tmp_path / ".originals" / "old.md").write_text("x", encoding="utf-8")
    assert {p.name for p in vector_cache._walk_corpus(tmp_path)} == {"keep.md"}


def test_fake_dim_constant_matches_vectors() -> None:
    assert len(fake_vector("x")) == FAKE_DIM

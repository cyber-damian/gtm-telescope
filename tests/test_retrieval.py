"""Tests for gtm_telescope.retrieval and the v2 cache (fake embeddings, no server)."""

from __future__ import annotations

import sqlite3
import subprocess
from collections.abc import Callable
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from gtm_telescope import retrieval, vector_cache
from gtm_telescope.chunking import ChunkParams
from gtm_telescope.config import load_telescope_settings
from gtm_telescope.query import parse_query

TODAY = date(2026, 10, 14)

CORPUS = {
    "ops/runbook.md": (
        "# Runbook\n\n## Database outage\n\n"
        "Promote the warm standby and update the connection string. Expect read skew.\n\n"
        "## Contacts\n\nPage the on-call engineer about unrelated rota details here.\n"
    ),
    "product/auth.md": (
        "# Auth\n\n## SSO\n\nSAML and OIDC are supported. SCIM provisioning is on the roadmap.\n"
    ),
    "old/pricing.md": (
        "---\ntype: battlecard\nupdated: 2023-01-10\n---\n# Pricing\n\n"
        "Starter tier has a rate-limit of 600 requests per minute.\n"
    ),
    "notes/plain.txt": "Docker layers cache build steps when ordering is stable.\n",
}


def _settings(extra: str = "") -> object:
    return load_telescope_settings(
        'retrieval:\n  semantic_threshold: "0.2"\n' + extra
    )


def _retrieve(data: Path, q: str, embed: Callable[[str], np.ndarray], **kw: object):
    return retrieval.retrieve(q, data, _settings(), embed=embed, today=TODAY, **kw)


# ---- RRF and boosts -------------------------------------------------------------------


def test_rrf_chunk_in_both_lists_outranks_single_list_at_equal_ranks() -> None:
    a, b, c = ("a", 0), ("b", 0), ("c", 0)
    scores = retrieval.rrf_scores([[a, b], [a, c]], rrf_k=60)
    assert scores[a] > scores[b]
    assert scores[b] == pytest.approx(scores[c])
    assert scores[a] == pytest.approx(2 / 61)
    # Equal ranks, different lists: b is rank 2 lexical, c is rank 2 semantic -> equal.
    only = retrieval.rrf_scores([[b], [c]], rrf_k=60)
    assert only[b] == only[c] == pytest.approx(1 / 61)


def test_exact_match_rules() -> None:
    assert retrieval.is_exact_match(parse_query('"TLS 1.3" okta'), "we require tls 1.3 only")
    assert not retrieval.is_exact_match(parse_query('"TLS 1.3" okta'), "tls 1.2")
    assert retrieval.is_exact_match(parse_query("scim provisioning"), "SCIM and Provisioning")
    assert not retrieval.is_exact_match(parse_query("scim provisioning"), "only scim")
    # More than three terms and no phrase: no exact boost.
    four = parse_query("alpha beta gamma delta")
    assert not retrieval.is_exact_match(four, "alpha beta gamma delta")


def test_exact_boost_is_applied(build_cache, fake_embeddings) -> None:
    data = build_cache(CORPUS)
    plain = retrieval.retrieve(
        "scim provisioning",
        data,
        load_telescope_settings('retrieval:\n  semantic_threshold: "0.2"\n  exact_boost: "0"'),
        embed=fake_embeddings,
        today=TODAY,
    )
    boosted = _retrieve(data, "scim provisioning", fake_embeddings)
    assert plain[0].path == boosted[0].path == "product/auth.md"
    assert boosted[0].rrf_score == pytest.approx(plain[0].rrf_score + 0.02)


# ---- retrieve() end to end ------------------------------------------------------------


def test_dual_source_ranks_first_with_metadata(build_cache, fake_embeddings) -> None:
    data = build_cache(CORPUS, ChunkParams(min_chunk_tokens=0))
    hits = _retrieve(data, "warm standby connection string", fake_embeddings)
    top = hits[0]
    assert top.path == "ops/runbook.md"
    assert top.source == "dual"
    assert top.cos is not None and top.bm25 is not None
    assert top.heading == "Runbook > Database outage"
    assert top.line_no <= 5 <= top.end_line
    assert top.rrf_score > 1 / 61


def test_exact_term_found_without_semantic_support(build_cache, fake_embeddings) -> None:
    data = build_cache(CORPUS)
    # A tiny query whose vocabulary has no semantic overlap above threshold still matches
    # lexically; the lexical-only hit is returned and labelled accordingly.
    hits = retrieval.retrieve(
        "OIDC",
        data,
        load_telescope_settings('retrieval:\n  semantic_threshold: "0.99"'),
        embed=fake_embeddings,
        today=TODAY,
    )
    assert hits and hits[0].path == "product/auth.md"
    assert hits[0].source == "lexical" and hits[0].cos is None


def test_lexical_hit_ranks_above_weak_semantic_noise(build_cache, fake_embeddings) -> None:
    data = build_cache(CORPUS)
    hits = _retrieve(data, "Do you support SCIM provisioning?", fake_embeddings)
    assert hits[0].path == "product/auth.md"


def test_stale_flag_and_doc_metadata(build_cache, fake_embeddings) -> None:
    data = build_cache(CORPUS)
    hits = _retrieve(data, "starter rate-limit", fake_embeddings)
    old = next(h for h in hits if h.path == "old/pricing.md")
    assert old.doc_date == "2023-01-10"
    assert old.doc_type == "battlecard"
    assert old.stale is True
    fresh = next(h for h in _retrieve(data, "scim roadmap", fake_embeddings))
    assert fresh.stale is False  # file mtime is "today" for freshly written test files


def test_results_report_original_file_lines_after_frontmatter(build_cache, fake_embeddings) -> None:
    data = build_cache(CORPUS)
    hit = next(h for h in _retrieve(data, "rate-limit", fake_embeddings) if "pricing" in h.path)
    lines = (data / hit.path).read_text(encoding="utf-8").splitlines()
    assert "\n".join(lines[hit.line_no - 1 : hit.end_line]) == hit.text
    assert "battlecard" not in hit.text


def test_no_cache_returns_empty(tmp_path: Path, fake_embeddings) -> None:
    assert _retrieve(tmp_path, "anything", fake_embeddings) == []


def test_retrieve_respects_limit_and_dedupes(build_cache, fake_embeddings) -> None:
    files = {f"d/{i}.md": "# T\n\nidentical body text for dedupe check here\n" for i in range(4)}
    data = build_cache(files)
    hits = _retrieve(data, "identical body text dedupe", fake_embeddings)
    assert len(hits) == 1
    assert len(retrieval.retrieve("scim", data, _settings(), embed=fake_embeddings, limit=1)) <= 1


# ---- FTS index sync -------------------------------------------------------------------


def _match(conn: sqlite3.Connection, word: str) -> list[int]:
    sql = "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?"
    return [r[0] for r in conn.execute(sql, (f'"{word}"',))]


def test_fts_triggers_track_insert_update_delete(tmp_path: Path) -> None:
    conn = vector_cache._connect(tmp_path / "c.sqlite")
    blob = vector_cache._vector_to_blob(np.zeros(4, dtype=np.float32))
    conn.execute(
        "INSERT INTO chunks(path, idx, line_no, end_line, heading, text, vector) "
        "VALUES ('p', 0, 1, 2, 'Head Alpha', 'banana split', ?)",
        (blob,),
    )
    assert len(_match(conn, "banana")) == 1
    assert len(_match(conn, "alpha")) == 1  # heading column is indexed too
    conn.execute("UPDATE chunks SET text = 'cherry pie' WHERE path = 'p'")
    assert _match(conn, "banana") == []
    assert len(_match(conn, "cherry")) == 1
    conn.execute("DELETE FROM chunks WHERE path = 'p'")
    assert _match(conn, "cherry") == [] and _match(conn, "alpha") == []
    conn.close()


def test_reindex_replaces_fts_rows(build_cache, fake_embeddings) -> None:
    data = build_cache(CORPUS)
    conn = vector_cache.open_readonly(data)
    assert conn is not None and len(_match(conn, "oidc")) == 1
    conn.close()
    (data / "product/auth.md").write_text("# Auth\n\nOnly passwords now.\n", encoding="utf-8")
    vector_cache.refresh_index(
        data, server_url="http://x", model="fake", dim=None, granularity="markdown",
        timeout=1.0, log_progress=False,
    )  # fmt: skip
    conn = vector_cache.open_readonly(data)
    assert conn is not None
    assert _match(conn, "oidc") == [] and len(_match(conn, "passwords")) == 1
    conn.close()


def test_heading_column_weighted_in_bm25(build_cache) -> None:
    files = {
        "a.md": "# Doc\n\n## Kerberos\n\nsome unrelated words about nothing in particular here.\n",
        "b.md": "# Doc\n\n## Other\n\nkerberos appears once in this body text of the section.\n",
    }
    data = build_cache(files)
    conn = vector_cache.open_readonly(data)
    assert conn is not None
    rows = vector_cache.lexical_search_fts(conn, '"kerberos"', 10)
    conn.close()
    assert rows[0][0][0].endswith("a.md")
    assert rows[0][1] < rows[1][1]  # bm25 is lower (better) for the heading match


# ---- ripgrep fallback -----------------------------------------------------------------


def test_rg_fallback_maps_line_hits_into_containing_chunk(tmp_path: Path) -> None:
    data = tmp_path
    p = str((data / "a.md").resolve())
    q = str((data / "b.md").resolve())
    spans = {p: [(0, 1, 5), (1, 7, 12)], q: [(0, 1, 3)]}
    out = "\n".join(
        [
            "./a.md:9:alpha and beta here",  # chunk (a, 1)
            "./a.md:10:alpha again",  # chunk (a, 1)
            "./b.md:2:beta only",  # chunk (b, 0)
            "./a.md:3:alpha",  # chunk (a, 0)
            "./other.pdf:1:alpha",  # not an indexed chunk
        ]
    )
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_k: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=out + "\n", stderr="")

    import gtm_telescope.retrieval as r

    orig = r.subprocess.run
    r.subprocess.run = fake_run  # type: ignore[assignment]
    try:
        keys = retrieval.lexical_rg_fallback(
            parse_query("alpha beta"), data, spans,
            rg_bin="rg", timeout=5, max_lines=100, top_k=10,
        )  # fmt: skip
    finally:
        r.subprocess.run = orig  # type: ignore[assignment]
    # (a,1) matched two distinct terms over two lines; (a,0) and (b,0) one term each.
    assert keys[0] == (p, 1)
    assert set(keys[1:]) == {(p, 0), (q, 0)}
    cmd = calls[0]
    assert cmd.count("-e") == 2 and "alpha" in cmd and "beta" in cmd
    assert "-F" in cmd and "-i" in cmd


def test_rg_fallback_used_when_fts_missing(build_cache, fake_embeddings, monkeypatch) -> None:
    data = build_cache(CORPUS)
    conn = sqlite3.connect(str(vector_cache.cache_db_path(data)))
    for trig in ("chunks_fts_ai", "chunks_fts_ad", "chunks_fts_au"):
        conn.execute(f"DROP TRIGGER {trig}")
    conn.execute("DROP TABLE chunks_fts")
    conn.commit()
    conn.close()
    abs_auth = str((data / "product/auth.md").resolve())
    assert abs_auth

    def fake_run(cmd: list[str], **_k: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            cmd, 0, stdout="./product/auth.md:5:SAML and OIDC are supported.\n", stderr=""
        )

    monkeypatch.setattr(retrieval.subprocess, "run", fake_run)
    hits = retrieval.retrieve(
        "OIDC",
        data,
        load_telescope_settings('retrieval:\n  semantic_threshold: "0.99"'),
        embed=fake_embeddings,
        rg_resolver=lambda: "rg",
        today=TODAY,
    )
    assert [h.path for h in hits] == ["product/auth.md"]
    assert hits[0].source == "lexical"


def test_rg_errors_become_retrieval_errors(monkeypatch, tmp_path: Path) -> None:
    def boom(cmd: list[str], **_k: object):
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(retrieval.subprocess, "run", boom)
    with pytest.raises(retrieval.RetrievalError, match="timed out"):
        retrieval.lexical_rg_fallback(
            parse_query("alpha"), tmp_path, {}, rg_bin="rg", timeout=1, max_lines=5, top_k=5
        )


# ---- cache schema ---------------------------------------------------------------------


def test_schema_v1_cache_is_rebuilt_with_notice(tmp_path: Path, capsys) -> None:
    db = vector_cache.cache_db_path(tmp_path)
    db.parent.mkdir(parents=True)
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE files (path TEXT PRIMARY KEY, mtime_ns INTEGER NOT NULL,
            size INTEGER NOT NULL, sha256 TEXT NOT NULL);
        CREATE TABLE chunks (path TEXT NOT NULL, idx INTEGER NOT NULL, line_no INTEGER NOT NULL,
            text TEXT NOT NULL, vector BLOB NOT NULL, PRIMARY KEY (path, idx));
        INSERT INTO meta VALUES ('schema_version', '1');
        INSERT INTO chunks VALUES ('p', 0, 1, 't', x'00');
        """
    )
    conn.commit()
    conn.close()
    assert vector_cache.cache_info(tmp_path)["schema_ok"] is False
    conn = vector_cache._connect(db)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(chunks)")}
    assert {"end_line", "heading"} <= cols
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
    assert vector_cache.has_fts(conn)
    conn.close()
    err = capsys.readouterr().err
    assert "schema v1 -> v2" in err and err.count("schema v1") == 1


def test_changing_chunk_params_triggers_reindex(build_cache) -> None:
    data = build_cache(CORPUS)
    files, _ = vector_cache.refresh_index(
        data, server_url="x", model="fake", dim=None, granularity="markdown",
        timeout=1.0, log_progress=False, chunk_params=ChunkParams(),
    )  # fmt: skip
    assert files == 0  # nothing changed
    files, chunks = vector_cache.refresh_index(
        data, server_url="x", model="fake", dim=None, granularity="markdown",
        timeout=1.0, log_progress=False, chunk_params=ChunkParams(chunk_tokens=120),
    )  # fmt: skip
    assert files == len(CORPUS) and chunks > 0


def test_resolve_doc_date_precedence() -> None:
    mtime = 1772409600 * 1_000_000_000  # 2026-03-02
    both = {"updated": "2024-08-11", "source_mtime": "2020-01-01"}
    assert vector_cache.resolve_doc_date(both, mtime) == "2024-08-11"
    assert vector_cache.resolve_doc_date({"date": "2024-05-01T10:00:00"}, mtime) == "2024-05-01"
    assert vector_cache.resolve_doc_date({"source_mtime": "2025-02-03"}, mtime) == "2025-02-03"
    assert vector_cache.resolve_doc_date({"updated": "not a date"}, mtime) == "2026-03-02"
    assert vector_cache.resolve_doc_date({}, mtime) == "2026-03-02"


def test_empty_file_does_not_cause_permanent_drift(build_cache) -> None:
    data = build_cache({"empty.md": "", "real.md": "# T\n\nsome real content for the index\n"})
    assert vector_cache.cache_info(data)["stale_or_new_files"] == 0


def test_is_stale_boundaries() -> None:
    assert retrieval.is_stale("2025-10-14", today=TODAY, stale_after_days=365) is False
    assert retrieval.is_stale("2025-10-13", today=TODAY, stale_after_days=365) is True
    assert retrieval.is_stale(None, today=TODAY, stale_after_days=365) is False
    assert retrieval.is_stale("garbage", today=TODAY, stale_after_days=365) is False

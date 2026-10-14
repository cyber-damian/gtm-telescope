"""Unit tests for gtm_telescope.vector_cache (no embedding server required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gtm_telescope import vector_cache


def test_vector_blob_round_trip() -> None:
    v = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float32)
    out = vector_cache._blob_to_vector(vector_cache._vector_to_blob(v))
    assert out.dtype == np.float32
    assert np.array_equal(out, v)


def test_sha256_deterministic() -> None:
    assert vector_cache._sha256(b"hello") == vector_cache._sha256(b"hello")
    assert vector_cache._sha256(b"hello") != vector_cache._sha256(b"world")


def test_walk_corpus_includes_md_md_markdown_txt(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    (tmp_path / "b.markdown").write_text("y", encoding="utf-8")
    (tmp_path / "c.txt").write_text("z", encoding="utf-8")
    (tmp_path / "d.pdf").write_bytes(b"%PDF")
    found = {p.name for p in vector_cache._walk_corpus(tmp_path)}
    assert found == {"a.md", "b.markdown", "c.txt"}


def test_walk_corpus_skips_lens_cache_only(tmp_path: Path) -> None:
    (tmp_path / "keep.md").write_text("x", encoding="utf-8")
    cache = tmp_path / ".lens-cache"
    cache.mkdir()
    (cache / "skip.md").write_text("x", encoding="utf-8")
    lex = tmp_path / "lexical-only"
    lex.mkdir()
    (lex / "under.txt").write_text("x", encoding="utf-8")
    found = {p.name for p in vector_cache._walk_corpus(tmp_path)}
    assert found == {"keep.md", "under.txt"}


def test_corpus_has_indexable_files(tmp_path: Path) -> None:
    d = tmp_path / "data"
    d.mkdir()
    assert vector_cache.corpus_has_indexable_files(d) is False
    (d / "a.md").write_text("x", encoding="utf-8")
    assert vector_cache.corpus_has_indexable_files(d) is True


def test_walk_corpus_recurses(tmp_path: Path) -> None:
    sub = tmp_path / "deeper" / "nested"
    sub.mkdir(parents=True)
    (sub / "doc.md").write_text("x", encoding="utf-8")
    found = {p.name for p in vector_cache._walk_corpus(tmp_path)}
    assert found == {"doc.md"}


def test_cache_info_missing_db(tmp_path: Path) -> None:
    d = tmp_path / "data"
    d.mkdir()
    info = vector_cache.cache_info(d)
    assert info["exists"] is False


def test_count_corpus_drift_no_false_positive_when_sha_matches(tmp_path: Path) -> None:
    """Stale mtime/size in DB but matching sha256 should not count as drift."""
    d = tmp_path / "data"
    d.mkdir()
    f = d / "note.md"
    f.write_text("stable body\n", encoding="utf-8")
    dbp = vector_cache.cache_db_path(d)
    conn = vector_cache._connect(dbp)
    p = str(f.resolve())
    content_bytes = f.read_bytes()
    sha = vector_cache._sha256(content_bytes)
    conn.execute(
        "INSERT INTO files(path, mtime_ns, size, sha256) VALUES (?,?,?,?)",
        (p, 1, 1, sha),
    )
    conn.commit()
    assert vector_cache.count_corpus_drift(conn, d) == 0
    conn.close()


def test_cache_info_and_clear(tmp_path: Path) -> None:
    d = tmp_path / "data"
    d.mkdir()
    (d / "note.md").write_text("# Title\nbody\n", encoding="utf-8")
    dbp = vector_cache.cache_db_path(d)
    conn = vector_cache._connect(dbp)
    vector_cache._meta_set(
        conn,
        {
            "schema_version": str(vector_cache.SCHEMA_VERSION),
            "model": vector_cache.DEFAULT_MODEL,
            "task": vector_cache.DEFAULT_TASK,
            "granularity": "token",
            "dim": "full",
        },
    )
    p = str((d / "note.md").resolve())
    conn.execute(
        "INSERT INTO files(path, mtime_ns, size, sha256) VALUES (?,?,?,?)",
        (p, 1, 1, "x" * 64),
    )
    blob = vector_cache._vector_to_blob(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    conn.execute(
        "INSERT INTO chunks(path, idx, line_no, end_line, heading, text, vector) "
        "VALUES (?,?,?,?,?,?,?)",
        (p, 0, 1, 1, None, "snippet", blob),
    )
    conn.commit()
    conn.close()

    info = vector_cache.cache_info(d)
    assert info["exists"] is True
    assert info["file_rows"] == 1
    assert info["chunk_rows"] == 1
    assert info["vector_bytes"] == len(blob)

    removed, cdir, errs = vector_cache.clear_cache(d)
    assert removed is True
    assert errs == []
    assert not dbp.is_file()
    info2 = vector_cache.cache_info(d)
    assert info2["exists"] is False

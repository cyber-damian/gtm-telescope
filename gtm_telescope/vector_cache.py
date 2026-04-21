"""Local vector cache for `lens focus` / `lens render`.

Stores per-chunk embeddings in SQLite under <data_dir>/.lens-cache/embeddings.sqlite.
Embeds via HTTP POST /v1/embeddings on the jina-grep embedding server. User-facing
tunables are loaded in `gtm_telescope.config`; this module only accepts explicit parameters.
"""

from __future__ import annotations

import hashlib
import http.client
import io
import json
import os
import shutil
import sqlite3
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
DEFAULT_SERVER_URL = "http://localhost:8089"
DEFAULT_MODEL = "jina-embeddings-v5-nano"
DEFAULT_TASK = "retrieval"
EMBED_BATCH = 256  # jina server allows up to 512
HEALTH_TIMEOUT_SEC = 2.0
VALID_TRUNCATE_DIMS = frozenset({32, 64, 128, 256, 512, 768, 1024})


def _http_post_json(url: str, payload: dict, timeout: float) -> dict:
    """One-shot POST (e.g. health checks). Prefer EmbeddingHttpClient for repeated embeds."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class EmbeddingHttpClient:
    """Reuses one TCP connection for repeated POST /v1/embeddings to the same host."""

    __slots__ = ("_base", "_timeout", "_conn", "_parsed")

    def __init__(self, server: str, timeout: float) -> None:
        self._base = server.rstrip("/")
        self._timeout = timeout
        self._parsed = urllib.parse.urlparse(self._base)
        self._conn: http.client.HTTPConnection | http.client.HTTPSConnection | None = None

    def close(self) -> None:
        if self._conn is not None:
            with suppress(OSError):
                self._conn.close()
            self._conn = None

    def __enter__(self) -> EmbeddingHttpClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _ensure_conn(self) -> http.client.HTTPConnection | http.client.HTTPSConnection:
        if self._conn is not None:
            return self._conn
        host = self._parsed.hostname
        if not host:
            raise ValueError(f"invalid embedding server URL: {self._base!r}")
        port = self._parsed.port
        if port is None:
            port = 443 if self._parsed.scheme == "https" else 80
        if self._parsed.scheme == "https":
            ctx = ssl.create_default_context()
            self._conn = http.client.HTTPSConnection(
                host, port, context=ctx, timeout=self._timeout
            )
        elif self._parsed.scheme == "http":
            self._conn = http.client.HTTPConnection(host, port, timeout=self._timeout)
        else:
            raise ValueError(f"unsupported URL scheme for embedding server: {self._base!r}")
        return self._conn

    def post_json(self, path: str, payload: dict) -> dict:
        conn = self._ensure_conn()
        body = json.dumps(payload).encode("utf-8")
        conn.request(
            "POST",
            path,
            body=body,
            headers={"Content-Type": "application/json", "Connection": "keep-alive"},
        )
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8")
        if resp.status >= 400:
            self.close()
            raise urllib.error.HTTPError(
                self._base + path,
                resp.status,
                resp.reason or "",
                resp.headers,
                io.BytesIO(raw.encode("utf-8")),
            )
        return json.loads(raw)


def health_check(url: str, timeout: float = HEALTH_TIMEOUT_SEC) -> bool:
    try:
        req = urllib.request.Request(f"{url.rstrip('/')}/health")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def embed_batch(
    texts: list[str],
    *,
    prompt_name: str,
    model: str,
    task: str,
    dim: int | None,
    server: str,
    timeout: float,
    http_client: EmbeddingHttpClient | None = None,
) -> np.ndarray:
    """POST /v1/embeddings, return L2-normalized matrix [N, D] as float32."""
    body: dict[str, Any] = {
        "input": texts,
        "model": model,
        "task": task,
        "prompt_name": prompt_name,
    }
    if dim is not None:
        body["truncate_dim"] = dim
    url = f"{server.rstrip('/')}/v1/embeddings"
    if http_client is not None:
        out = http_client.post_json("/v1/embeddings", body)
    else:
        out = _http_post_json(url, body, timeout=timeout)
    arr = np.asarray([row["embedding"] for row in out["data"]], dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def split_into_chunks(content: str, granularity: str) -> list[tuple[int, str]]:
    """Defer to jina-grep's splitter so cached chunks match server-side semantics."""
    from jina_grep.client import split_into_chunks as _impl

    return _impl(content, granularity)


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _expected_meta(model: str, granularity: str, dim: int | None) -> dict[str, str]:
    return {
        "schema_version": str(SCHEMA_VERSION),
        "model": model,
        "task": DEFAULT_TASK,
        "granularity": granularity,
        "dim": str(dim) if dim is not None else "full",
    }


_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    mtime_ns INTEGER NOT NULL,
    size INTEGER NOT NULL,
    sha256 TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    path TEXT NOT NULL,
    idx INTEGER NOT NULL,
    line_no INTEGER NOT NULL,
    text TEXT NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (path, idx)
);
CREATE INDEX IF NOT EXISTS chunks_path_idx ON chunks(path);
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(_SCHEMA_DDL)
    return conn


def _meta_get(conn: sqlite3.Connection) -> dict[str, str]:
    return dict(conn.execute("SELECT key, value FROM meta"))


def _meta_set(conn: sqlite3.Connection, meta: dict[str, str]) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
        list(meta.items()),
    )


def _prepare_meta(conn: sqlite3.Connection, expected_meta: dict[str, str]) -> None:
    current = _meta_get(conn)
    if current and any(current.get(k) != v for k, v in expected_meta.items()):
        conn.executescript("DELETE FROM chunks; DELETE FROM files; DELETE FROM meta;")
    _meta_set(conn, expected_meta)


_CORPUS_SUFFIXES = frozenset({".md", ".markdown", ".txt"})
_CORPUS_SKIP_DIRS = frozenset({".lens-cache"})


def _walk_corpus(data: Path) -> Iterable[Path]:
    stack: list[Path] = [data]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if entry.name not in _CORPUS_SKIP_DIRS:
                                stack.append(Path(entry.path))
                            continue
                        suf = Path(entry.name).suffix.lower()
                        if entry.is_file(follow_symlinks=False) and suf in _CORPUS_SUFFIXES:
                            yield Path(entry.path)
                    except OSError:
                        continue
        except OSError:
            continue


def _vector_to_blob(v: np.ndarray) -> bytes:
    return v.astype(np.float32, copy=False).tobytes()


def _blob_to_vector(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


def _sync_corpus_into_cache(
    conn: sqlite3.Connection,
    data: Path,
    *,
    granularity: str,
    model: str,
    dim: int | None,
    server: str,
    timeout: float,
    log_progress: bool = True,
    max_file_bytes: int | None = None,
) -> tuple[int, int]:
    """Bring the cache in sync with disk. Returns (files_reindexed, chunks_embedded)."""
    data_resolved = data.resolve()
    live_paths: set[str] = set()
    pending_meta_updates: list[tuple[int, int, str]] = []
    to_index: list[tuple[str, int, int, str, bytes]] = []

    for f in _walk_corpus(data_resolved):
        try:
            st = f.stat()
        except OSError:
            continue
        abs_path = str(f.resolve())
        live_paths.add(abs_path)
        if max_file_bytes is not None and st.st_size > max_file_bytes:
            if log_progress:
                rel = f.name
                sys.stderr.write(
                    f"vector cache: skipping {rel} ({st.st_size} bytes > max_file_bytes); "
                    "raise embedding.max_file_bytes in config.yaml to index it\n"
                )
                sys.stderr.flush()
            continue
        row = conn.execute(
            "SELECT mtime_ns, size, sha256 FROM files WHERE path = ?",
            (abs_path,),
        ).fetchone()
        if row and row[0] == st.st_mtime_ns and row[1] == st.st_size:
            continue
        try:
            content_bytes = f.read_bytes()
        except OSError:
            continue
        sha = _sha256(content_bytes)
        if row and row[2] == sha:
            pending_meta_updates.append((st.st_mtime_ns, st.st_size, abs_path))
            continue
        to_index.append((abs_path, st.st_mtime_ns, st.st_size, sha, content_bytes))

    existing = {r[0] for r in conn.execute("SELECT path FROM files")}
    gone = existing - live_paths

    buffered: list[tuple[str, int, int, str]] = []
    for abs_path, _mtime, _size, _sha, content_bytes in to_index:
        text = content_bytes.decode("utf-8", errors="replace")
        chunks = split_into_chunks(text, granularity)
        for idx, (line_no, body) in enumerate(chunks):
            if not body.strip():
                continue
            buffered.append((abs_path, idx, line_no, body))

    paths_dirty = [t[0] for t in to_index]
    n_chunks = len(buffered)

    if log_progress and to_index:
        sys.stderr.write(
            f"vector cache: indexing {len(to_index)} file(s) "
            f"(model={model}, granularity={granularity})…\n"
        )
        sys.stderr.flush()

    # sqlite3 leaves an implicit transaction open after prior statements (meta writes,
    # corpus SELECTs). BEGIN IMMEDIATE requires no active transaction.
    conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    try:
        for mtime_ns, size, path in pending_meta_updates:
            conn.execute(
                "UPDATE files SET mtime_ns=?, size=? WHERE path=?",
                (mtime_ns, size, path),
            )
        if gone:
            qmarks = ",".join("?" * len(gone))
            gone_list = list(gone)
            conn.execute(f"DELETE FROM chunks WHERE path IN ({qmarks})", gone_list)
            conn.execute(f"DELETE FROM files WHERE path IN ({qmarks})", gone_list)
        if paths_dirty:
            qmarks = ",".join("?" * len(paths_dirty))
            conn.execute(f"DELETE FROM chunks WHERE path IN ({qmarks})", paths_dirty)

        if not buffered and not pending_meta_updates and not gone:
            conn.commit()
            return (0, 0)

        if buffered:
            with EmbeddingHttpClient(server, timeout) as http_client:
                for start in range(0, n_chunks, EMBED_BATCH):
                    batch = buffered[start : start + EMBED_BATCH]
                    texts = [b[3] for b in batch]
                    vecs = embed_batch(
                        texts,
                        prompt_name="document",
                        model=model,
                        task=DEFAULT_TASK,
                        dim=dim,
                        server=server,
                        timeout=timeout,
                        http_client=http_client,
                    )
                    ins = (
                        "INSERT INTO chunks(path, idx, line_no, text, vector) "
                        "VALUES (?, ?, ?, ?, ?)"
                    )
                    conn.executemany(
                        ins,
                        [
                            (p, i, ln, t, _vector_to_blob(vecs[k]))
                            for k, (p, i, ln, t) in enumerate(batch)
                        ],
                    )
                    if log_progress and n_chunks:
                        end = min(start + EMBED_BATCH, n_chunks)
                        sys.stderr.write(f"vector cache: embedded {end}/{n_chunks} chunk(s)…\n")
                        sys.stderr.flush()

        if to_index:
            conn.executemany(
                "INSERT OR REPLACE INTO files(path, mtime_ns, size, sha256) VALUES (?,?,?,?)",
                [(p, mt, sz, sh) for p, mt, sz, sh, _ in to_index],
            )
        conn.commit()
    except (
        OSError,
        ValueError,
        urllib.error.URLError,
        urllib.error.HTTPError,
        json.JSONDecodeError,
        KeyError,
    ):
        conn.rollback()
        raise

    return (len(to_index), n_chunks)


def cache_db_path(data: Path) -> Path:
    return data.resolve() / ".lens-cache" / "embeddings.sqlite"


def clear_cache(data: Path) -> tuple[bool, Path, list[str]]:
    """Remove <data>/.lens-cache/.

    Returns ``(fully_removed, cache_dir, errors)`` where ``fully_removed`` is True only if
    the directory no longer exists after the operation and there were no rmtree errors.
    """
    cache_dir = data.resolve() / ".lens-cache"
    if not cache_dir.is_dir():
        return (False, cache_dir, [])
    errors: list[str] = []
    try:
        shutil.rmtree(cache_dir)
    except OSError as e:
        errors.append(str(e))
        still = cache_dir.exists()
        return (not still, cache_dir, errors)
    return (True, cache_dir, [])


def count_corpus_drift(conn: sqlite3.Connection, data: Path) -> int:
    """How many on-disk corpus files are missing or stale vs the files table.

    Uses the same rules as :func:`_sync_corpus_into_cache`: match on ``mtime_ns`` + ``size``,
    else re-read bytes and compare ``sha256`` to the indexed hash (no false drift when only
    metadata changed).
    """
    n = 0
    for f in _walk_corpus(data.resolve()):
        try:
            st = f.stat()
        except OSError:
            continue
        ap = str(f.resolve())
        row = conn.execute(
            "SELECT mtime_ns, size, sha256 FROM files WHERE path=?",
            (ap,),
        ).fetchone()
        if row is None:
            n += 1
            continue
        if row[0] == st.st_mtime_ns and row[1] == st.st_size:
            continue
        try:
            content_bytes = f.read_bytes()
        except OSError:
            n += 1
            continue
        if row[2] == _sha256(content_bytes):
            continue
        n += 1
    return n


def corpus_has_indexable_files(data: Path) -> bool:
    """True if data_dir contains at least one .md / .markdown / .txt (same rules as cache walk)."""
    for _ in _walk_corpus(data.resolve()):
        return True
    return False


def cache_info(data: Path) -> dict[str, Any]:
    """Read-only cache statistics (no server calls)."""
    db_path = cache_db_path(data)
    out: dict[str, Any] = {
        "exists": db_path.is_file(),
        "db_path": db_path,
    }
    if not out["exists"]:
        return out
    try:
        out["db_size_bytes"] = db_path.stat().st_size
    except OSError:
        out["db_size_bytes"] = 0
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        meta = _meta_get(conn)
        out["meta"] = meta
        out["file_rows"] = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        out["chunk_rows"] = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        row = conn.execute("SELECT SUM(LENGTH(vector)) FROM chunks").fetchone()
        out["vector_bytes"] = int(row[0] or 0)
        row_max = conn.execute("SELECT MAX(mtime_ns) FROM files").fetchone()
        out["max_mtime_ns"] = row_max[0]
        out["stale_or_new_files"] = count_corpus_drift(conn, data)
    finally:
        conn.close()
    return out


def refresh_index(
    data: Path,
    *,
    server_url: str,
    model: str,
    dim: int | None,
    granularity: str,
    timeout: float,
    log_progress: bool = True,
    max_file_bytes: int | None = None,
) -> tuple[int, int]:
    """Open DB, ensure meta matches, sync corpus. Returns (files_reindexed, chunks_embedded)."""
    data_resolved = data.resolve()
    db_path = cache_db_path(data_resolved)
    expected = _expected_meta(model, granularity, dim)
    conn = _connect(db_path)
    try:
        _prepare_meta(conn, expected)
        return _sync_corpus_into_cache(
            conn,
            data_resolved,
            granularity=granularity,
            model=model,
            dim=dim,
            server=server_url,
            timeout=timeout,
            log_progress=log_progress,
            max_file_bytes=max_file_bytes,
        )
    finally:
        conn.close()


def collect_semantic_results(
    query: str,
    data: Path,
    *,
    server_url: str,
    model: str,
    dim: int | None,
    granularity: str,
    threshold: float,
    top_k: int,
    timeout: float,
    max_file_bytes: int | None = None,
    max_chunk_rows: int = 500_000,
) -> dict[tuple[str, int], tuple[float, str]]:
    """Cache-backed semantic search.

    Returns { (abs_path, line_no): (cosine_score, snippet_text) }.
    Caller must ensure the embedding server is reachable before calling.
    """
    expected_meta = _expected_meta(model, granularity, dim)
    data_resolved = data.resolve()
    db_path = cache_db_path(data_resolved)
    conn = _connect(db_path)
    try:
        _prepare_meta(conn, expected_meta)
        _sync_corpus_into_cache(
            conn,
            data_resolved,
            granularity=granularity,
            model=model,
            dim=dim,
            server=server_url,
            timeout=timeout,
            max_file_bytes=max_file_bytes,
        )

        n_total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        if n_total == 0:
            return {}

        if n_total > max_chunk_rows:
            sys.stderr.write(
                f"vector cache: warning: {n_total} chunk rows exceed retrieval.max_chunk_rows "
                f"({max_chunk_rows}); loading first {max_chunk_rows} rows for this query only\n"
            )
            sys.stderr.flush()

        cur = conn.execute(
            "SELECT path, line_no, text, vector FROM chunks LIMIT ?",
            (max_chunk_rows,),
        )
        first = cur.fetchone()
        if first is None:
            return {}
        dim_vec = len(first[3]) // 4
        rows = [first]
        rows.extend(cur.fetchall())
        n = len(rows)
        paths = [r[0] for r in rows]
        lines = [r[1] for r in rows]
        texts = [r[2] for r in rows]
        mat = np.empty((n, dim_vec), dtype=np.float32)
        for i, r in enumerate(rows):
            mat[i] = _blob_to_vector(r[3])

        with EmbeddingHttpClient(server_url, timeout) as http_client:
            q = embed_batch(
                [query],
                prompt_name="query",
                model=model,
                task=DEFAULT_TASK,
                dim=dim,
                server=server_url,
                timeout=timeout,
                http_client=http_client,
            )[0]

        scores = mat @ q
        n_scores = scores.shape[0]
        k = min(top_k, n_scores)
        if k <= 0:
            return {}

        idx_part = np.argpartition(-scores, k - 1)[:k]
        idx_sorted = idx_part[np.argsort(-scores[idx_part])]

        results: dict[tuple[str, int], tuple[float, str]] = {}
        for j in idx_sorted:
            s = float(scores[j])
            if s < threshold:
                continue
            key = (paths[j], int(lines[j]))
            prev = results.get(key)
            if prev is None or s > prev[0]:
                results[key] = (s, texts[j])
        return results
    finally:
        conn.close()


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_SERVER_URL",
    "DEFAULT_TASK",
    "EMBED_BATCH",
    "EmbeddingHttpClient",
    "SCHEMA_VERSION",
    "VALID_TRUNCATE_DIMS",
    "cache_db_path",
    "cache_info",
    "clear_cache",
    "collect_semantic_results",
    "corpus_has_indexable_files",
    "count_corpus_drift",
    "embed_batch",
    "health_check",
    "refresh_index",
]

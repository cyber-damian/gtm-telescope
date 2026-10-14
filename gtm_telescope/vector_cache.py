"""Local vector cache for `lens focus` / `lens render`.

Stores per-chunk embeddings (plus an FTS5 lexical index) in SQLite under
<data_dir>/.lens-cache/embeddings.sqlite. Indexing happens only in ``refresh_index`` (lens
calibrate); the read path (focus/render/eval) never embeds corpus files.
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
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gtm_telescope import chunking
from gtm_telescope.chunking import Chunk, ChunkedDoc, ChunkParams
from gtm_telescope.corpus import CORPUS_SUFFIXES, SKIP_CORPUS_DIRS

SCHEMA_VERSION = 2
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
    """Defer to jina-grep's splitter (granularity line/paragraph/sentence/token)."""
    from jina_grep.client import split_into_chunks as _impl

    return _impl(content, granularity)


def chunk_file(content: str, suffix: str, params: ChunkParams) -> ChunkedDoc:
    """Chunk one corpus file. ``markdown`` uses our chunker; other granularities use jina-grep."""
    if params.granularity == "markdown":
        return chunking.chunk_document(content, suffix, params)
    meta, _body, _offset = chunking.split_frontmatter(content)
    chunks: list[Chunk] = []
    for line_no, body in split_into_chunks(content, params.granularity):
        if not body.strip():
            continue
        end_line = line_no + body.rstrip("\n").count("\n")
        chunks.append(Chunk(line_no=line_no, end_line=end_line, heading_path="", text=body))
    title = str(meta["title"]).strip() if meta.get("title") else None
    return ChunkedDoc(meta=meta, title=title, chunks=chunks)


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _expected_meta(
    model: str, granularity: str, dim: int | None, chunk: ChunkParams | None = None
) -> dict[str, str]:
    meta = {
        "schema_version": str(SCHEMA_VERSION),
        "model": model,
        "task": DEFAULT_TASK,
        "granularity": granularity,
        "dim": str(dim) if dim is not None else "full",
    }
    if granularity == "markdown":
        c = chunk or ChunkParams()
        meta["chunk_tokens"] = str(c.chunk_tokens)
        meta["min_chunk_tokens"] = str(c.min_chunk_tokens)
        meta["chunk_overlap_lines"] = str(c.chunk_overlap_lines)
    return meta


def _iso_date(raw: object) -> str | None:
    if not isinstance(raw, str):
        return None
    try:
        return date.fromisoformat(raw.strip()[:10]).isoformat()
    except ValueError:
        return None


def resolve_doc_date(meta: dict[str, Any], mtime_ns: int) -> str:
    """ISO date: frontmatter ``updated``/``date`` > ``source_mtime`` (calibrate) > file mtime."""
    for key in ("updated", "date", "source_mtime"):
        d = _iso_date(meta.get(key))
        if d:
            return d
    return datetime.fromtimestamp(mtime_ns / 1e9, tz=timezone.utc).date().isoformat()


def _doc_type(meta: dict[str, Any]) -> str | None:
    raw = meta.get("type")
    if isinstance(raw, str) and raw.strip():
        return raw.strip().lower()
    return None


_BASE_DDL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    mtime_ns INTEGER NOT NULL,
    size INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    doc_date TEXT,
    doc_type TEXT,
    title TEXT
);
CREATE TABLE IF NOT EXISTS chunks (
    path TEXT NOT NULL,
    idx INTEGER NOT NULL,
    line_no INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    heading TEXT,
    text TEXT NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (path, idx)
);
CREATE INDEX IF NOT EXISTS chunks_path_idx ON chunks(path);
"""

# External-content FTS5 index kept in sync by triggers. Note: chunks are never written with
# INSERT OR REPLACE (REPLACE does not fire delete triggers unless recursive_triggers is on);
# the sync code deletes a file's chunks explicitly before re-inserting.
_FTS_DDL = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    heading, text,
    content='chunks', content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS chunks_fts_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, heading, text) VALUES (new.rowid, new.heading, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_fts_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, heading, text)
    VALUES ('delete', old.rowid, old.heading, old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_fts_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, heading, text)
    VALUES ('delete', old.rowid, old.heading, old.text);
    INSERT INTO chunks_fts(rowid, heading, text) VALUES (new.rowid, new.heading, new.text);
END;
"""

_fts_warned = False


def _warn_fts_unavailable() -> None:
    global _fts_warned
    if _fts_warned:
        return
    _fts_warned = True
    sys.stderr.write(
        "vector cache: SQLite FTS5 is unavailable; lexical search falls back to ripgrep\n"
    )
    sys.stderr.flush()


def _migrate_legacy_schema(conn: sqlite3.Connection) -> None:
    """Drop a pre-v2 cache wholesale (column layout changed; ``DELETE`` is not enough)."""
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not tables:
        return
    old: str | None = None
    if "meta" in tables:
        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        if row and row[0] != str(SCHEMA_VERSION):
            old = row[0]
    if old is None and "chunks" in tables:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(chunks)")}
        if "end_line" not in cols:
            old = "1"
    if old is None:
        return
    for trig in ("chunks_fts_ai", "chunks_fts_ad", "chunks_fts_au"):
        conn.execute(f"DROP TRIGGER IF EXISTS {trig}")
    with suppress(sqlite3.OperationalError):
        conn.execute("DROP TABLE IF EXISTS chunks_fts")
    for t in ("chunks", "files", "meta"):
        conn.execute(f"DROP TABLE IF EXISTS {t}")
    conn.commit()
    sys.stderr.write(
        f"vector cache: schema v{old} -> v{SCHEMA_VERSION}; rebuilding the index "
        "(re-embeds the whole corpus)\n"
    )
    sys.stderr.flush()


def _ensure_fts(conn: sqlite3.Connection) -> bool:
    try:
        conn.executescript(_FTS_DDL)
    except sqlite3.OperationalError as e:
        msg = str(e).lower()
        if "fts5" in msg or "no such module" in msg:
            _warn_fts_unavailable()
            return False
        raise
    return True


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    _migrate_legacy_schema(conn)
    conn.executescript(_BASE_DDL)
    _ensure_fts(conn)
    return conn


def has_fts(conn: sqlite3.Connection) -> bool:
    """True if the FTS5 index exists in this database."""
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE name='chunks_fts'").fetchone()
    return row is not None


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
        # Triggers keep chunks_fts in sync with the chunk deletes.
        conn.executescript("DELETE FROM chunks; DELETE FROM files; DELETE FROM meta;")
    _meta_set(conn, expected_meta)


_CORPUS_SUFFIXES = CORPUS_SUFFIXES
_CORPUS_SKIP_DIRS = SKIP_CORPUS_DIRS


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


@dataclass(frozen=True)
class _PendingChunk:
    path: str
    idx: int
    chunk: Chunk
    embed_text: str


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
    chunk_params: ChunkParams | None = None,
) -> tuple[int, int]:
    """Bring the cache in sync with disk. Returns (files_reindexed, chunks_embedded)."""
    params = chunk_params or ChunkParams(granularity=granularity)
    if params.granularity != granularity:
        params = ChunkParams(
            granularity=granularity,
            chunk_tokens=params.chunk_tokens,
            min_chunk_tokens=params.min_chunk_tokens,
            chunk_overlap_lines=params.chunk_overlap_lines,
        )
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

    buffered: list[_PendingChunk] = []
    file_rows: list[tuple[str, int, int, str, str, str | None, str | None]] = []
    for abs_path, mtime_ns, size, sha, content_bytes in to_index:
        text = content_bytes.decode("utf-8", errors="replace")
        doc = chunk_file(text, Path(abs_path).suffix, params)
        file_rows.append(
            (
                abs_path,
                mtime_ns,
                size,
                sha,
                resolve_doc_date(doc.meta, mtime_ns),
                _doc_type(doc.meta),
                doc.title,
            )
        )
        for idx, ch in enumerate(doc.chunks):
            buffered.append(_PendingChunk(abs_path, idx, ch, chunking.embedding_input(ch)))

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

        if not to_index and not pending_meta_updates and not gone:
            conn.commit()
            return (0, 0)

        if buffered:
            with EmbeddingHttpClient(server, timeout) as http_client:
                for start in range(0, n_chunks, EMBED_BATCH):
                    batch = buffered[start : start + EMBED_BATCH]
                    vecs = embed_batch(
                        [b.embed_text for b in batch],
                        prompt_name="document",
                        model=model,
                        task=DEFAULT_TASK,
                        dim=dim,
                        server=server,
                        timeout=timeout,
                        http_client=http_client,
                    )
                    conn.executemany(
                        "INSERT INTO chunks(path, idx, line_no, end_line, heading, text, vector) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        [
                            (
                                b.path,
                                b.idx,
                                b.chunk.line_no,
                                b.chunk.end_line,
                                b.chunk.heading_path or None,
                                b.chunk.text,
                                _vector_to_blob(vecs[k]),
                            )
                            for k, b in enumerate(batch)
                        ],
                    )
                    if log_progress and n_chunks:
                        end = min(start + EMBED_BATCH, n_chunks)
                        sys.stderr.write(f"vector cache: embedded {end}/{n_chunks} chunk(s)…\n")
                        sys.stderr.flush()

        if file_rows:
            conn.executemany(
                "INSERT OR REPLACE INTO files"
                "(path, mtime_ns, size, sha256, doc_date, doc_type, title) "
                "VALUES (?,?,?,?,?,?,?)",
                file_rows,
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
        out["schema_ok"] = meta.get("schema_version") == str(SCHEMA_VERSION)
        out["file_rows"] = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        out["chunk_rows"] = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        row = conn.execute("SELECT SUM(LENGTH(vector)) FROM chunks").fetchone()
        out["vector_bytes"] = int(row[0] or 0)
        row_max = conn.execute("SELECT MAX(mtime_ns) FROM files").fetchone()
        out["max_mtime_ns"] = row_max[0]
        out["stale_or_new_files"] = count_corpus_drift(conn, data)
        out["fts5"] = has_fts(conn)
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
    chunk_params: ChunkParams | None = None,
) -> tuple[int, int]:
    """Open DB, ensure meta matches, sync corpus. Returns (files_reindexed, chunks_embedded)."""
    data_resolved = data.resolve()
    db_path = cache_db_path(data_resolved)
    expected = _expected_meta(model, granularity, dim, chunk_params)
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
            chunk_params=chunk_params,
        )
    finally:
        conn.close()


# --------------------------------------------------------------------------------------
# Read path (focus / render / eval). Never writes, never embeds corpus files.
# --------------------------------------------------------------------------------------

ChunkKey = tuple[str, int]


@dataclass(frozen=True)
class ChunkRow:
    path: str
    idx: int
    line_no: int
    end_line: int
    heading: str | None
    text: str
    doc_date: str | None
    doc_type: str | None
    title: str | None


@dataclass(frozen=True)
class VectorMatrix:
    keys: list[ChunkKey]
    matrix: np.ndarray  # [N, D] float32, rows L2-normalized


def open_readonly(data: Path) -> sqlite3.Connection | None:
    """Open the cache read-only; ``None`` if it does not exist."""
    db_path = cache_db_path(data)
    if not db_path.is_file():
        return None
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def lexical_search_fts(
    conn: sqlite3.Connection, match_expr: str, limit: int
) -> list[tuple[ChunkKey, float]]:
    """FTS5 BM25 search (heading weighted 2x). Returns ``[((path, idx), bm25)]``, best first.

    ``match_expr`` must come from :func:`gtm_telescope.query.to_fts5` (fully quoted).
    """
    rows = conn.execute(
        "SELECT c.path, c.idx, bm25(chunks_fts, 2.0, 1.0) AS score "
        "FROM chunks_fts JOIN chunks c ON c.rowid = chunks_fts.rowid "
        "WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
        (match_expr, limit),
    ).fetchall()
    return [((r[0], int(r[1])), float(r[2])) for r in rows]


def chunk_spans(conn: sqlite3.Connection) -> dict[str, list[tuple[int, int, int]]]:
    """``{path: [(idx, line_no, end_line), ...]}`` for mapping line hits to chunks."""
    out: dict[str, list[tuple[int, int, int]]] = {}
    for path, idx, a, b in conn.execute(
        "SELECT path, idx, line_no, end_line FROM chunks ORDER BY path, idx"
    ):
        out.setdefault(path, []).append((int(idx), int(a), int(b)))
    return out


def fetch_chunk_rows(
    conn: sqlite3.Connection, keys: Iterable[ChunkKey]
) -> dict[ChunkKey, ChunkRow]:
    wanted = list(dict.fromkeys(keys))
    out: dict[ChunkKey, ChunkRow] = {}
    for start in range(0, len(wanted), 400):
        part = wanted[start : start + 400]
        clause = " OR ".join("(c.path = ? AND c.idx = ?)" for _ in part)
        params = [v for k in part for v in k]
        for r in conn.execute(
            "SELECT c.path, c.idx, c.line_no, c.end_line, c.heading, c.text, "
            "f.doc_date, f.doc_type, f.title "
            "FROM chunks c LEFT JOIN files f ON f.path = c.path "
            f"WHERE {clause}",
            params,
        ):
            row = ChunkRow(r[0], int(r[1]), int(r[2]), int(r[3]), r[4], r[5], r[6], r[7], r[8])
            out[(row.path, row.idx)] = row
    return out


def load_vector_matrix(conn: sqlite3.Connection, max_chunk_rows: int) -> VectorMatrix | None:
    n_total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    if n_total == 0:
        return None
    if n_total > max_chunk_rows:
        sys.stderr.write(
            f"vector cache: warning: {n_total} chunk rows exceed retrieval.max_chunk_rows "
            f"({max_chunk_rows}); loading first {max_chunk_rows} rows for this query only\n"
        )
        sys.stderr.flush()
    keys: list[ChunkKey] = []
    blobs: list[bytes] = []
    for path, idx, blob in conn.execute(
        "SELECT path, idx, vector FROM chunks LIMIT ?", (max_chunk_rows,)
    ):
        keys.append((path, int(idx)))
        blobs.append(blob)
    if not blobs:
        return None
    mat = np.frombuffer(b"".join(blobs), dtype=np.float32).reshape(len(blobs), -1)
    return VectorMatrix(keys=keys, matrix=mat)


def semantic_rank(
    vm: VectorMatrix, query_vec: np.ndarray, *, top_k: int, threshold: float
) -> list[tuple[ChunkKey, float]]:
    """Cosine top-k (vectors are normalized), filtered by ``threshold``; best first."""
    scores = vm.matrix @ query_vec
    k = min(top_k, scores.shape[0])
    if k <= 0:
        return []
    part = np.argpartition(-scores, k - 1)[:k]
    ordered = part[np.argsort(-scores[part])]
    return [(vm.keys[int(j)], float(scores[j])) for j in ordered if float(scores[j]) >= threshold]


def embed_query(
    query: str,
    *,
    server_url: str,
    model: str,
    dim: int | None,
    timeout: float,
) -> np.ndarray:
    with EmbeddingHttpClient(server_url, timeout) as http_client:
        return embed_batch(
            [query],
            prompt_name="query",
            model=model,
            task=DEFAULT_TASK,
            dim=dim,
            server=server_url,
            timeout=timeout,
            http_client=http_client,
        )[0]


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_SERVER_URL",
    "DEFAULT_TASK",
    "EMBED_BATCH",
    "SCHEMA_VERSION",
    "VALID_TRUNCATE_DIMS",
    "ChunkRow",
    "EmbeddingHttpClient",
    "VectorMatrix",
    "cache_db_path",
    "cache_info",
    "chunk_file",
    "chunk_spans",
    "clear_cache",
    "corpus_has_indexable_files",
    "count_corpus_drift",
    "embed_batch",
    "embed_query",
    "fetch_chunk_rows",
    "has_fts",
    "health_check",
    "lexical_search_fts",
    "load_vector_matrix",
    "open_readonly",
    "refresh_index",
    "resolve_doc_date",
    "semantic_rank",
]

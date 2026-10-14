"""Shared fixtures: a deterministic fake embedding model and a cache builder (no server)."""

from __future__ import annotations

import re
import zlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from gtm_telescope import vector_cache
from gtm_telescope.chunking import ChunkParams

FAKE_DIM = 64
_WORD_RE = re.compile(r"[a-z0-9]+")


def fake_vector(text: str) -> np.ndarray:
    """Hashed bag-of-words, L2-normalized: cosine rises with shared vocabulary."""
    v = np.zeros(FAKE_DIM, dtype=np.float32)
    for w in _WORD_RE.findall(text.lower()):
        v[zlib.crc32(w.encode()) % FAKE_DIM] += 1.0
    n = float(np.linalg.norm(v))
    if n == 0:
        v[0] = 1.0
        return v
    return v / n


def fake_embed_batch(texts: list[str], **_kwargs: object) -> np.ndarray:
    return np.stack([fake_vector(t) for t in texts])


@pytest.fixture
def fake_embeddings(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], np.ndarray]:
    """Patch ``vector_cache.embed_batch``; returns the matching query-embedding callable."""
    monkeypatch.setattr(vector_cache, "embed_batch", fake_embed_batch)
    return fake_vector


@pytest.fixture
def build_cache(
    tmp_path: Path, fake_embeddings: Callable[[str], np.ndarray]
) -> Callable[..., Path]:
    """``build_cache({"a.md": "text"})`` -> data dir with a fully indexed v2 cache."""

    def _build(files: dict[str, str], params: ChunkParams | None = None) -> Path:
        data = tmp_path / "data"
        for rel, body in files.items():
            p = data / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body, encoding="utf-8")
        vector_cache.refresh_index(
            data,
            server_url="http://127.0.0.1:1",
            model="fake",
            dim=None,
            granularity="markdown",
            timeout=1.0,
            log_progress=False,
            chunk_params=params,
        )
        return data

    return _build

"""Shared minimal YAML parser and telescope tunables (defaults + config.yaml only).

Used by the Python CLI (`gtm_telescope.cli`) and the bootstrap shell script
(`scripts/bootstrap.sh`) for `parse_yaml_block` semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from gtm_telescope import chunking, vector_cache

# --- Defaults (also used when YAML omits a value) ---

DEFAULT_SEMANTIC_CANDIDATE_TOP_K = 15
DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD = 0.38
VALID_SEMANTIC_GRANULARITY = frozenset(
    {"markdown", "line", "paragraph", "sentence", "token"}
)
DEFAULT_SEMANTIC_GRANULARITY = "markdown"
DEFAULT_CHUNK_TOKENS = chunking.DEFAULT_CHUNK_TOKENS
DEFAULT_MIN_CHUNK_TOKENS = chunking.DEFAULT_MIN_CHUNK_TOKENS
DEFAULT_CHUNK_OVERLAP_LINES = chunking.DEFAULT_CHUNK_OVERLAP_LINES
DEFAULT_LEXICAL_TOP_K = 15
DEFAULT_RRF_K = 60
DEFAULT_EXACT_BOOST = 0.02
DEFAULT_CARD_BOOST = 0.01
DEFAULT_STALE_AFTER_DAYS = 365
VALID_ORIGINALS_MODES = frozenset({"move", "keep", "delete"})
DEFAULT_ORIGINALS_MODE = "move"
DEFAULT_LEXICAL_MAX_LINES = 500
DEFAULT_HYBRID_REPORT_MAX_ROWS = 10
DEFAULT_HYBRID_REPORT_SNIPPET_BLOCKS = 3
DEFAULT_HYBRID_REPORT_SNIPPET_WRAP = 88

DEFAULT_SUBPROCESS_TIMEOUT_SEC = 120.0
DEFAULT_HTTP_TIMEOUT_SEC = 120.0
DEFAULT_EMBED_HTTP_TIMEOUT_SEC = 60.0
DEFAULT_EMBED_SERVER_READY_SEC = 60.0

# Default cap for corpus files read into memory during indexing (bytes). 0 in YAML = unlimited.
DEFAULT_MAX_FILE_BYTES = 5 * 1024 * 1024
DEFAULT_MAX_CHUNK_ROWS = 500_000

_TRUTHY_ALLOW_REMOTE = frozenset({"1", "true", "yes", "on"})


@dataclass(frozen=True)
class TelescopeSettings:
    """User-tunable retrieval, embedding, process limits, and optional tool paths."""

    semantic_candidate_top_k: int
    semantic_similarity_threshold: float
    semantic_granularity: str
    lexical_max_lines: int
    hybrid_report_max_rows: int
    hybrid_report_snippet_blocks: int
    hybrid_report_snippet_wrap: int
    embedding_server_url: str
    embedding_model: str
    embedding_truncate_dim: int | None
    embed_http_timeout: float
    embed_server_ready_sec: float
    subprocess_timeout: float
    http_timeout: float
    jina_grep_bin: str | None
    rg_bin: str | None
    embedding_allow_remote: bool
    max_file_bytes: int | None
    max_chunk_rows: int
    chunk_tokens: int
    min_chunk_tokens: int
    chunk_overlap_lines: int
    lexical_top_k: int
    rrf_k: int
    exact_boost: float
    card_boost: float
    stale_after_days: int
    calibrate_originals: str

    @property
    def chunk_params(self) -> chunking.ChunkParams:
        return chunking.ChunkParams(
            granularity=self.semantic_granularity,
            chunk_tokens=self.chunk_tokens,
            min_chunk_tokens=self.min_chunk_tokens,
            chunk_overlap_lines=self.chunk_overlap_lines,
        )


def parse_yaml_block(text: str, block: str) -> dict[str, str]:
    """Parse one top-level `block:` indented `key: value` section.

    Only supports flat string values under a top-level key; keys with no value
    or comment-only lines are skipped. Quoted values have surrounding single or
    double quotes stripped. This is intentionally tiny and not a YAML parser.
    """
    out: dict[str, str] = {}
    header = f"{block}:"
    in_block = False
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        stripped = line.strip()
        if stripped == header:
            in_block = True
            continue
        if not in_block:
            continue
        if stripped == "":
            continue
        if not (line.startswith(" ") or line.startswith("\t")):
            break
        if ":" in stripped:
            key, val = stripped.split(":", 1)
            out[key.strip()] = val.strip().strip("\"'")
    return out


def _parse_int(raw: str, *, default: int, minimum: int = 1) -> int:
    try:
        v = int(raw.strip())
    except ValueError:
        return default
    return v if v >= minimum else default


def _parse_float(raw: str, *, default: float, minimum: float | None = None) -> float:
    try:
        v = float(raw.strip())
    except ValueError:
        return default
    if minimum is not None and v <= minimum:
        return default
    return v


def _parse_nonneg_float(raw: str, *, default: float) -> float:
    try:
        v = float(raw.strip())
    except ValueError:
        return default
    return v if v >= 0.0 else default


def _parse_originals(raw: str, *, default: str) -> str:
    v = raw.strip().lower()
    return v if v in VALID_ORIGINALS_MODES else default


def _parse_threshold(raw: str, *, default: float) -> float:
    try:
        v = float(raw.strip())
    except ValueError:
        return default
    if not (0.0 < v < 1.0):
        return default
    return v


def _parse_granularity(raw: str, *, default: str) -> str:
    s = raw.strip().lower()
    if s in VALID_SEMANTIC_GRANULARITY:
        return s
    return default


def _parse_truncate_dim(raw: str) -> int | None:
    s = raw.strip()
    if not s:
        return None
    try:
        v = int(s)
    except ValueError:
        return None
    return v if v in vector_cache.VALID_TRUNCATE_DIMS else None


def _optional_path(raw: str) -> str | None:
    s = raw.strip()
    return s if s else None


def _parse_allow_remote(raw: str) -> bool:
    return raw.strip().lower() in _TRUTHY_ALLOW_REMOTE


def _parse_max_file_bytes(raw: str) -> int | None:
    """Return None for unlimited; positive int is max bytes (minimum 1024 when set)."""
    s = raw.strip()
    if not s:
        return DEFAULT_MAX_FILE_BYTES
    try:
        v = int(s)
    except ValueError:
        return DEFAULT_MAX_FILE_BYTES
    if v == 0:
        return None
    return max(v, 1024)


def _parse_max_chunk_rows(raw: str) -> int:
    s = raw.strip()
    if not s:
        return DEFAULT_MAX_CHUNK_ROWS
    try:
        v = int(s)
    except ValueError:
        return DEFAULT_MAX_CHUNK_ROWS
    return max(v, 1_000)


def _embedding_host_is_local(url: str) -> bool:
    p = urlparse(url.strip())
    host = (p.hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1")


def load_telescope_settings(config_yaml_text: str) -> TelescopeSettings:
    """Merge defaults with config.yaml blocks: embedding, retrieval, process, paths.

    Pure function: pass fixture YAML in tests; no disk I/O here.
    """
    emb = parse_yaml_block(config_yaml_text, "embedding")
    ret = parse_yaml_block(config_yaml_text, "retrieval")
    proc = parse_yaml_block(config_yaml_text, "process")
    paths = parse_yaml_block(config_yaml_text, "paths")
    cal = parse_yaml_block(config_yaml_text, "calibrate")

    semantic_candidate_top_k = _parse_int(
        ret.get("semantic_candidate_top_k", ""),
        default=DEFAULT_SEMANTIC_CANDIDATE_TOP_K,
        minimum=1,
    )
    semantic_similarity_threshold = _parse_threshold(
        ret.get("semantic_threshold", ""),
        default=DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD,
    )
    semantic_granularity = _parse_granularity(
        ret.get("semantic_granularity", ""),
        default=DEFAULT_SEMANTIC_GRANULARITY,
    )
    lexical_max_lines = _parse_int(
        ret.get("lexical_max_lines", ""),
        default=DEFAULT_LEXICAL_MAX_LINES,
        minimum=1,
    )
    hybrid_report_max_rows = _parse_int(
        ret.get("hybrid_report_max_rows", ""),
        default=DEFAULT_HYBRID_REPORT_MAX_ROWS,
        minimum=1,
    )
    hybrid_report_snippet_blocks = _parse_int(
        ret.get("hybrid_report_snippet_blocks", ""),
        default=DEFAULT_HYBRID_REPORT_SNIPPET_BLOCKS,
        minimum=1,
    )
    hybrid_report_snippet_wrap = _parse_int(
        ret.get("hybrid_report_snippet_wrap", ""),
        default=DEFAULT_HYBRID_REPORT_SNIPPET_WRAP,
        minimum=20,
    )

    embedding_allow_remote = _parse_allow_remote(emb.get("allow_remote", ""))
    embedding_server_url = emb.get("server_url", "").strip() or vector_cache.DEFAULT_SERVER_URL
    embedding_model = emb.get("model", "").strip() or vector_cache.DEFAULT_MODEL
    embedding_truncate_dim = _parse_truncate_dim(emb.get("truncate_dim", ""))
    if not embedding_allow_remote and not _embedding_host_is_local(embedding_server_url):
        raise ValueError(
            "embedding.server_url must point to localhost, 127.0.0.1, or ::1 unless "
            "embedding.allow_remote is set to true in config.yaml (SSRF guard)."
        )
    max_file_bytes = _parse_max_file_bytes(emb.get("max_file_bytes", ""))
    max_chunk_rows = _parse_max_chunk_rows(ret.get("max_chunk_rows", ""))
    embed_http_timeout = _parse_float(
        emb.get("embed_http_timeout", ""),
        default=DEFAULT_EMBED_HTTP_TIMEOUT_SEC,
        minimum=0.0,
    )
    if embed_http_timeout <= 0:
        embed_http_timeout = DEFAULT_EMBED_HTTP_TIMEOUT_SEC
    embed_server_ready_sec = _parse_float(
        emb.get("embed_server_ready_sec", ""),
        default=DEFAULT_EMBED_SERVER_READY_SEC,
        minimum=0.0,
    )
    if embed_server_ready_sec <= 0:
        embed_server_ready_sec = DEFAULT_EMBED_SERVER_READY_SEC

    subprocess_timeout = _parse_float(
        proc.get("subprocess_timeout", ""),
        default=DEFAULT_SUBPROCESS_TIMEOUT_SEC,
        minimum=0.0,
    )
    if subprocess_timeout <= 0:
        subprocess_timeout = DEFAULT_SUBPROCESS_TIMEOUT_SEC
    http_timeout = _parse_float(
        proc.get("http_timeout", ""),
        default=DEFAULT_HTTP_TIMEOUT_SEC,
        minimum=0.0,
    )
    if http_timeout <= 0:
        http_timeout = DEFAULT_HTTP_TIMEOUT_SEC

    jina_grep_bin = _optional_path(paths.get("jina_grep_bin", ""))
    rg_bin = _optional_path(paths.get("rg_bin", ""))

    chunk_tokens = _parse_int(
        ret.get("chunk_tokens", ""), default=DEFAULT_CHUNK_TOKENS, minimum=20
    )
    min_chunk_tokens = _parse_int(
        ret.get("min_chunk_tokens", ""), default=DEFAULT_MIN_CHUNK_TOKENS, minimum=0
    )
    if min_chunk_tokens >= chunk_tokens:
        min_chunk_tokens = min(DEFAULT_MIN_CHUNK_TOKENS, chunk_tokens // 2)
    chunk_overlap_lines = _parse_int(
        ret.get("chunk_overlap_lines", ""), default=DEFAULT_CHUNK_OVERLAP_LINES, minimum=0
    )
    lexical_top_k = _parse_int(
        ret.get("lexical_top_k", ""), default=DEFAULT_LEXICAL_TOP_K, minimum=1
    )
    rrf_k = _parse_int(ret.get("rrf_k", ""), default=DEFAULT_RRF_K, minimum=1)
    exact_boost = _parse_nonneg_float(ret.get("exact_boost", ""), default=DEFAULT_EXACT_BOOST)
    card_boost = _parse_nonneg_float(ret.get("card_boost", ""), default=DEFAULT_CARD_BOOST)
    stale_after_days = _parse_int(
        ret.get("stale_after_days", ""), default=DEFAULT_STALE_AFTER_DAYS, minimum=1
    )
    calibrate_originals = _parse_originals(
        cal.get("originals", ""), default=DEFAULT_ORIGINALS_MODE
    )

    return TelescopeSettings(
        semantic_candidate_top_k=semantic_candidate_top_k,
        semantic_similarity_threshold=semantic_similarity_threshold,
        semantic_granularity=semantic_granularity,
        lexical_max_lines=lexical_max_lines,
        hybrid_report_max_rows=hybrid_report_max_rows,
        hybrid_report_snippet_blocks=hybrid_report_snippet_blocks,
        hybrid_report_snippet_wrap=hybrid_report_snippet_wrap,
        embedding_server_url=embedding_server_url,
        embedding_model=embedding_model,
        embedding_truncate_dim=embedding_truncate_dim,
        embed_http_timeout=embed_http_timeout,
        embed_server_ready_sec=embed_server_ready_sec,
        subprocess_timeout=subprocess_timeout,
        http_timeout=http_timeout,
        jina_grep_bin=jina_grep_bin,
        rg_bin=rg_bin,
        embedding_allow_remote=embedding_allow_remote,
        max_file_bytes=max_file_bytes,
        max_chunk_rows=max_chunk_rows,
        chunk_tokens=chunk_tokens,
        min_chunk_tokens=min_chunk_tokens,
        chunk_overlap_lines=chunk_overlap_lines,
        lexical_top_k=lexical_top_k,
        rrf_k=rrf_k,
        exact_boost=exact_boost,
        card_boost=card_boost,
        stale_after_days=stale_after_days,
        calibrate_originals=calibrate_originals,
    )

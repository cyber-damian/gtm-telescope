"""Smoke tests for the shared YAML block parser and telescope settings."""

from __future__ import annotations

import pytest

from gtm_telescope import vector_cache
from gtm_telescope.config import (
    DEFAULT_EMBED_HTTP_TIMEOUT_SEC,
    DEFAULT_MAX_CHUNK_ROWS,
    DEFAULT_MAX_FILE_BYTES,
    DEFAULT_SEMANTIC_GRANULARITY,
    DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD,
    load_telescope_settings,
    parse_yaml_block,
)

SAMPLE = """
# top-level comment
llm:
  base_url: "https://api.openai.com/v1"
  model: 'gpt-4o-mini'
  api_key_env: OPENAI_API_KEY

storage:
  data_dir: "data-storage"

other:
  ignored: true
""".lstrip()


def test_parse_llm_block_strips_quotes() -> None:
    out = parse_yaml_block(SAMPLE, "llm")
    assert out == {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
    }


def test_parse_storage_block() -> None:
    out = parse_yaml_block(SAMPLE, "storage")
    assert out == {"data_dir": "data-storage"}


def test_parse_missing_block_returns_empty() -> None:
    assert parse_yaml_block(SAMPLE, "nope") == {}


def test_parse_stops_at_next_top_level_key() -> None:
    """A second top-level key must terminate the current block."""
    out = parse_yaml_block(SAMPLE, "llm")
    assert "data_dir" not in out
    assert "ignored" not in out


def test_parse_handles_tab_indent() -> None:
    text = "block:\n\tkey: value\nnext: 1\n"
    assert parse_yaml_block(text, "block") == {"key": "value"}


def test_load_telescope_settings_defaults_empty_yaml() -> None:
    s = load_telescope_settings("")
    assert s.semantic_similarity_threshold == DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD
    assert s.semantic_granularity == DEFAULT_SEMANTIC_GRANULARITY
    assert s.embedding_server_url == vector_cache.DEFAULT_SERVER_URL
    assert s.embedding_model == vector_cache.DEFAULT_MODEL
    assert s.embed_http_timeout == DEFAULT_EMBED_HTTP_TIMEOUT_SEC
    assert s.jina_grep_bin is None
    assert s.rg_bin is None
    assert s.embedding_allow_remote is False
    assert s.max_file_bytes == DEFAULT_MAX_FILE_BYTES
    assert s.max_chunk_rows == DEFAULT_MAX_CHUNK_ROWS


def test_load_telescope_settings_yaml_retrieval_block() -> None:
    yaml = """
retrieval:
  semantic_threshold: "0.5"
  semantic_granularity: line
  semantic_candidate_top_k: 7
"""
    s = load_telescope_settings(yaml)
    assert abs(s.semantic_similarity_threshold - 0.5) < 1e-9
    assert s.semantic_granularity == "line"
    assert s.semantic_candidate_top_k == 7


def test_load_telescope_settings_yaml_embedding_block() -> None:
    yaml = """
embedding:
  server_url: http://example.test:9000
  model: custom-model
  allow_remote: "true"
"""
    s = load_telescope_settings(yaml)
    assert s.embedding_server_url == "http://example.test:9000"
    assert s.embedding_model == "custom-model"
    assert s.embedding_allow_remote is True


def test_load_telescope_settings_rejects_remote_embedding_without_flag() -> None:
    yaml = """
embedding:
  server_url: http://example.test:9000
"""
    with pytest.raises(ValueError, match="allow_remote"):
        load_telescope_settings(yaml)


def test_load_telescope_settings_yaml_embedding_truncate_dim() -> None:
    yaml = """
embedding:
  truncate_dim: "256"
"""
    s = load_telescope_settings(yaml)
    assert s.embedding_truncate_dim == 256


def test_load_telescope_settings_yaml_embedding_truncate_dim_invalid() -> None:
    yaml = """
embedding:
  truncate_dim: "999"
"""
    assert load_telescope_settings(yaml).embedding_truncate_dim is None


def test_load_telescope_settings_yaml_embed_http_timeout() -> None:
    yaml = """
embedding:
  embed_http_timeout: "12.5"
"""
    assert abs(load_telescope_settings(yaml).embed_http_timeout - 12.5) < 1e-9


def test_load_telescope_settings_paths_block() -> None:
    yaml = """
paths:
  jina_grep_bin: "/opt/jina-grep"
  rg_bin: "/opt/rg"
"""
    s = load_telescope_settings(yaml)
    assert s.jina_grep_bin == "/opt/jina-grep"
    assert s.rg_bin == "/opt/rg"


def test_phase1_retrieval_defaults() -> None:
    s = load_telescope_settings("")
    assert s.semantic_granularity == "markdown"
    assert (s.chunk_tokens, s.min_chunk_tokens, s.chunk_overlap_lines) == (350, 40, 2)
    assert (s.lexical_top_k, s.rrf_k) == (15, 60)
    assert (s.exact_boost, s.card_boost) == (0.02, 0.01)
    assert s.stale_after_days == 365
    assert s.calibrate_originals == "move"
    assert s.chunk_params.granularity == "markdown"


def test_phase1_retrieval_overrides_and_validation() -> None:
    s = load_telescope_settings(
        "retrieval:\n"
        '  semantic_granularity: "token"\n'
        '  chunk_tokens: "500"\n'
        '  min_chunk_tokens: "10"\n'
        '  chunk_overlap_lines: "0"\n'
        '  lexical_top_k: "7"\n'
        '  rrf_k: "30"\n'
        '  exact_boost: "0"\n'
        '  card_boost: "0.5"\n'
        '  stale_after_days: "90"\n'
        "calibrate:\n"
        "  originals: keep\n"
    )
    assert s.semantic_granularity == "token"
    assert (s.chunk_tokens, s.min_chunk_tokens, s.chunk_overlap_lines) == (500, 10, 0)
    assert (s.lexical_top_k, s.rrf_k, s.stale_after_days) == (7, 30, 90)
    assert (s.exact_boost, s.card_boost) == (0.0, 0.5)
    assert s.calibrate_originals == "keep"


def test_phase1_invalid_values_fall_back_to_defaults() -> None:
    s = load_telescope_settings(
        "retrieval:\n"
        '  semantic_granularity: "bogus"\n'
        '  chunk_tokens: "3"\n'
        '  min_chunk_tokens: "9999"\n'
        '  chunk_overlap_lines: "-1"\n'
        '  lexical_top_k: "0"\n'
        '  rrf_k: "abc"\n'
        '  exact_boost: "-0.1"\n'
        '  stale_after_days: "0"\n'
        "calibrate:\n"
        "  originals: shred\n"
    )
    assert s.semantic_granularity == "markdown"
    assert s.chunk_tokens == 350
    assert s.min_chunk_tokens < s.chunk_tokens
    assert s.chunk_overlap_lines == 2
    assert (s.lexical_top_k, s.rrf_k, s.exact_boost, s.stale_after_days) == (15, 60, 0.02, 365)
    assert s.calibrate_originals == "move"  # never fall back to "delete"


def test_render_settings_defaults_and_overrides() -> None:
    d = load_telescope_settings("")
    assert d.render_max_chunks == 8 and d.batch_low_rrf == 0.02
    c = load_telescope_settings('retrieval:\n  render_max_chunks: "3"\n  batch_low_rrf: "0.05"\n')
    assert c.render_max_chunks == 3 and c.batch_low_rrf == 0.05


def test_render_settings_invalid_fall_back() -> None:
    c = load_telescope_settings('retrieval:\n  render_max_chunks: "0"\n  batch_low_rrf: "x"\n')
    assert c.render_max_chunks == 8 and c.batch_low_rrf == 0.02


def test_shipped_config_sets_a_model_per_provider() -> None:
    from pathlib import Path

    from gtm_telescope.llm import PROVIDERS

    text = (Path(__file__).resolve().parent.parent / "config.yaml").read_text(encoding="utf-8")
    llm = parse_yaml_block(text, "llm")
    for p in PROVIDERS:
        assert llm[f"{p.name}_model"]
    assert "base_url" not in llm

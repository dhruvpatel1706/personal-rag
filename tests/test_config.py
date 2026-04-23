"""Tests for config loading."""

from __future__ import annotations

from pathlib import Path

from personal_rag.config import Settings


def test_defaults() -> None:
    s = Settings(_env_file=None)  # don't load .env
    assert s.model == "claude-opus-4-7"
    assert s.embedding_model == "BAAI/bge-small-en-v1.5"
    assert s.chunk_size == 800
    assert 0 <= s.chunk_overlap < s.chunk_size
    assert s.top_k >= 1
    assert isinstance(s.db_path, Path)


def test_overrides() -> None:
    s = Settings(
        _env_file=None,
        anthropic_api_key="test-key",
        model="claude-sonnet-4-6",
        top_k=10,
    )
    assert s.anthropic_api_key == "test-key"
    assert s.model == "claude-sonnet-4-6"
    assert s.top_k == 10

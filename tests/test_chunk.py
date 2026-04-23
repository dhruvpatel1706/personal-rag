"""Tests for deterministic chunking."""

from __future__ import annotations

import pytest

from personal_rag.chunk import chunk_text


def test_empty_returns_empty() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n\t   ") == []


def test_short_text_returns_one_chunk() -> None:
    assert chunk_text("hello world", size=100, overlap=10) == ["hello world"]


def test_splits_on_sentence_boundaries() -> None:
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    chunks = chunk_text(text, size=35, overlap=5)
    assert len(chunks) >= 2
    # No chunk should start mid-word (sentences end with '.')
    for c in chunks:
        assert c.strip() == c


def test_long_single_sentence_hard_sliced() -> None:
    # No sentence boundary — forced hard slice
    text = "x" * 500
    chunks = chunk_text(text, size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_invalid_args() -> None:
    with pytest.raises(ValueError):
        chunk_text("hello", size=0)
    with pytest.raises(ValueError):
        chunk_text("hello", size=10, overlap=10)
    with pytest.raises(ValueError):
        chunk_text("hello", size=10, overlap=-1)


def test_chunk_count_monotonic_with_input_length() -> None:
    short = chunk_text("a. " * 30, size=30, overlap=5)
    long = chunk_text("a. " * 100, size=30, overlap=5)
    assert len(long) >= len(short)

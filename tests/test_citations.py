"""Tests for inline-citation parsing."""

from __future__ import annotations

from personal_rag.generate import extract_citations


def test_extract_unique_sorted():
    text = "First point [2]. Second point [1][3]. Third [2]."
    assert extract_citations(text, max_index=5) == [1, 2, 3]


def test_extract_ignores_out_of_range():
    text = "Claim cited as [1] and a hallucinated [7] citation."
    assert extract_citations(text, max_index=3) == [1]


def test_extract_empty_text():
    assert extract_citations("", max_index=5) == []


def test_extract_no_citations():
    assert extract_citations("plain answer with no brackets", max_index=5) == []


def test_extract_mixed_bracket_shapes_ignores_non_numeric():
    text = "see [Appendix A] and [2] and [third]"
    assert extract_citations(text, max_index=5) == [2]

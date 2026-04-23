"""Tests for generate input validation (no live API calls)."""

from __future__ import annotations

import pytest

from personal_rag.config import Settings
from personal_rag.generate import generate


def test_empty_question_raises() -> None:
    s = Settings(_env_file=None)
    with pytest.raises(ValueError, match="Empty question"):
        generate("", [], s)


def test_no_passages_returns_polite_fallback() -> None:
    s = Settings(_env_file=None)
    ans = generate("What is X?", [], s)
    assert "no relevant passages" in ans.text.lower()
    assert ans.used_passages == []

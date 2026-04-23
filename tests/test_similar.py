"""Tests for the 'similar' retrieval helpers.

We don't exercise LanceDB or the real embedding model here — that would
drag a 130MB ONNX download into CI. Instead we stub Index + embedding_dim
inside `personal_rag.similar` and feed synthetic chunks/hits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from personal_rag import similar as similar_mod


class _FakeIndex:
    """Minimal stand-in for personal_rag.index.Index.

    Exposes just the methods `similar.py` calls: get_by_id, get_by_source,
    search. search() returns a pre-programmed list of hits per query.
    """

    def __init__(
        self,
        *,
        rows: list[dict] | None = None,
        search_fn=None,
    ):
        self._rows = rows or []
        self._search_fn = search_fn

    # Index API surface the similar module uses:
    def get_by_id(self, chunk_id):
        for row in self._rows:
            if row["id"] == chunk_id:
                return dict(row)
        return None

    def get_by_source(self, source):
        return [dict(r) for r in self._rows if r["source"] == source]

    def search(self, vector, *, k):
        if self._search_fn is None:
            return []
        return self._search_fn(vector, k)


def _fake_settings(tmp_path: Path):
    @dataclass
    class S:
        embedding_model: str = "fake-model"
        db_path: Path = tmp_path
        table_name: str = "t"
        top_k: int = 5

    return S()


def _install_fake_index(monkeypatch, fake_index: _FakeIndex):
    monkeypatch.setattr(similar_mod, "embedding_dim", lambda name: 4)
    monkeypatch.setattr(similar_mod, "Index", lambda *a, **kw: fake_index)


def test_similar_to_chunk_excludes_same_source(tmp_path, monkeypatch):
    rows = [
        {"id": "a.md:0", "source": "a.md", "chunk_index": 0, "text": "ta0", "vector": [1, 0, 0, 0]},
        {"id": "a.md:1", "source": "a.md", "chunk_index": 1, "text": "ta1", "vector": [0, 1, 0, 0]},
        {"id": "b.md:0", "source": "b.md", "chunk_index": 0, "text": "tb0", "vector": [0, 0, 1, 0]},
        {"id": "c.md:0", "source": "c.md", "chunk_index": 0, "text": "tc0", "vector": [0, 0, 0, 1]},
    ]

    def search(vector, k):
        # Seed is a.md:0. Pretend the NN order is: itself, then a.md:1, then b.md:0, then c.md:0
        return [
            {"id": "a.md:0", "source": "a.md", "chunk_index": 0, "text": "ta0"},
            {"id": "a.md:1", "source": "a.md", "chunk_index": 1, "text": "ta1"},
            {"id": "b.md:0", "source": "b.md", "chunk_index": 0, "text": "tb0"},
            {"id": "c.md:0", "source": "c.md", "chunk_index": 0, "text": "tc0"},
        ][:k]

    _install_fake_index(monkeypatch, _FakeIndex(rows=rows, search_fn=search))

    result = similar_mod.similar_to_chunk(_fake_settings(tmp_path), "a.md:0", k=2)
    # Must drop both a.md rows (the seed and its sibling from the same source)
    assert [r["id"] for r in result] == ["b.md:0", "c.md:0"]


def test_similar_to_chunk_raises_on_unknown_id(tmp_path, monkeypatch):
    _install_fake_index(monkeypatch, _FakeIndex(rows=[]))
    with pytest.raises(KeyError, match="no chunk with id"):
        similar_mod.similar_to_chunk(_fake_settings(tmp_path), "missing:0", k=3)


def test_similar_to_source_aggregates_by_rr(tmp_path, monkeypatch):
    # Seed source 'a.md' has two chunks.
    rows = [
        {"id": "a.md:0", "source": "a.md", "chunk_index": 0, "text": "x", "vector": [1, 0, 0, 0]},
        {"id": "a.md:1", "source": "a.md", "chunk_index": 1, "text": "y", "vector": [0, 1, 0, 0]},
    ]

    # When queried with a.md:0's vector, b.md is rank 1 and c.md is rank 2.
    # When queried with a.md:1's vector, c.md is rank 1 and b.md is rank 2.
    # So both b and c end up with 1/61 + 1/62. Scores tie.
    def search(vector, k):
        if vector == [1, 0, 0, 0]:
            return [
                {"id": "a.md:0", "source": "a.md", "chunk_index": 0, "text": ""},
                {"id": "b.md:0", "source": "b.md", "chunk_index": 0, "text": ""},
                {"id": "c.md:0", "source": "c.md", "chunk_index": 0, "text": ""},
            ]
        return [
            {"id": "a.md:1", "source": "a.md", "chunk_index": 1, "text": ""},
            {"id": "c.md:0", "source": "c.md", "chunk_index": 0, "text": ""},
            {"id": "b.md:0", "source": "b.md", "chunk_index": 0, "text": ""},
        ]

    _install_fake_index(monkeypatch, _FakeIndex(rows=rows, search_fn=search))
    ranked = similar_mod.similar_to_source(_fake_settings(tmp_path), "a.md", k=5)
    sources = {s for s, _ in ranked}
    assert sources == {"b.md", "c.md"}
    # Scores should be equal (symmetric rank distribution)
    scores = {s: sc for s, sc in ranked}
    assert abs(scores["b.md"] - scores["c.md"]) < 1e-9
    # And both should equal 2/61 + 2/62 = 1/61 + 1/62 (each of the two
    # chunks contributed one hit at rank 1 and one at rank 2)
    assert scores["b.md"] == pytest.approx(1 / 61 + 1 / 62)


def test_similar_to_source_honors_k_limit(tmp_path, monkeypatch):
    rows = [
        {"id": "a.md:0", "source": "a.md", "chunk_index": 0, "text": "x", "vector": [1, 0, 0, 0]},
    ]

    def search(vector, k):
        return [
            {"id": "a.md:0", "source": "a.md", "chunk_index": 0, "text": ""},
            *(
                {"id": f"n{i}.md:0", "source": f"n{i}.md", "chunk_index": 0, "text": ""}
                for i in range(10)
            ),
        ]

    _install_fake_index(monkeypatch, _FakeIndex(rows=rows, search_fn=search))
    ranked = similar_mod.similar_to_source(_fake_settings(tmp_path), "a.md", k=3)
    assert len(ranked) == 3


def test_similar_to_source_missing_source_raises(tmp_path, monkeypatch):
    _install_fake_index(monkeypatch, _FakeIndex(rows=[]))
    with pytest.raises(KeyError, match="no chunks found"):
        similar_mod.similar_to_source(_fake_settings(tmp_path), "gone.md", k=3)


def test_similar_to_source_filters_self_source(tmp_path, monkeypatch):
    # If search keeps returning the same source's chunks at the top, it still
    # shouldn't leak into the ranked output.
    rows = [
        {"id": "a.md:0", "source": "a.md", "chunk_index": 0, "text": "x", "vector": [1, 0, 0, 0]},
    ]

    def search(vector, k):
        return [
            {"id": "a.md:0", "source": "a.md", "chunk_index": 0, "text": ""},
            {"id": "a.md:1", "source": "a.md", "chunk_index": 1, "text": ""},
            {"id": "b.md:0", "source": "b.md", "chunk_index": 0, "text": ""},
        ]

    _install_fake_index(monkeypatch, _FakeIndex(rows=rows, search_fn=search))
    ranked = similar_mod.similar_to_source(_fake_settings(tmp_path), "a.md", k=5)
    assert [s for s, _ in ranked] == ["b.md"]

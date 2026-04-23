"""Tests for BM25 scoring and Reciprocal Rank Fusion."""

from __future__ import annotations

from personal_rag.bm25 import bm25_rank, rrf_fuse


def test_bm25_empty_inputs_return_empty():
    assert bm25_rank([], "x", top_n=5) == []
    assert bm25_rank(["some text"], "", top_n=5) == []


def test_bm25_ranks_exact_match_highly():
    corpus = [
        "retrieval augmented generation combines embeddings with an LLM",
        "the ornithology of the northern goshawk",
        "making espresso at home requires a decent grinder",
    ]
    hits = bm25_rank(corpus, "retrieval augmented generation", top_n=3)
    # The first doc contains every query term — it should rank first.
    assert hits
    assert hits[0].corpus_index == 0
    assert hits[0].score > 0


def test_bm25_filters_zero_scores():
    corpus = ["no match here", "nothing related either"]
    hits = bm25_rank(corpus, "ornithology", top_n=5)
    # Zero relevance means no hits even though top_n asked for 5.
    assert hits == []


def test_rrf_fuses_two_ranked_lists():
    # doc_A appears rank-1 in list 1 and rank-3 in list 2. doc_B is rank-2 in
    # list 1 and rank-1 in list 2. doc_C shows up only in list 1.
    # RRF should put B or A at the top — we just assert both beat C.
    list1 = ["A", "B", "C"]
    list2 = ["B", "X", "A"]
    fused = rrf_fuse([list1, list2], top_n=5)
    fused_ids = [doc for doc, _ in fused]
    assert fused_ids.index("A") < fused_ids.index("C")
    assert fused_ids.index("B") < fused_ids.index("C")


def test_rrf_single_list_preserves_order():
    fused = rrf_fuse([["A", "B", "C"]], top_n=3)
    assert [doc for doc, _ in fused] == ["A", "B", "C"]


def test_rrf_respects_top_n():
    fused = rrf_fuse([[f"doc{i}" for i in range(20)]], top_n=5)
    assert len(fused) == 5


def test_rrf_empty_returns_empty():
    assert rrf_fuse([], top_n=5) == []

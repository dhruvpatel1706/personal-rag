"""Lightweight BM25 index over a list of text chunks.

Uses rank_bm25 (pure Python, zero deps). We rebuild per query rather than
caching the tokenized corpus: at a few-thousand-chunk scale this is cheap
(tokenization + a matrix-free Okapi scoring pass), and the simplicity beats
getting cache invalidation right when the user re-ingests a file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(s: str) -> list[str]:
    return _TOKEN.findall(s.lower())


@dataclass(frozen=True)
class BM25Hit:
    corpus_index: int  # position of the chunk in the input list
    score: float


def bm25_rank(texts: list[str], query: str, *, top_n: int) -> list[BM25Hit]:
    """Return the top-`n` (index, score) pairs for `query` over `texts`.

    Returns an empty list on empty inputs. Zero-score results are filtered.
    """
    if not texts or not query.strip():
        return []
    corpus = [_tokenize(t) for t in texts]
    q = _tokenize(query)
    if not q:
        return []

    engine = BM25Okapi(corpus)
    scores = engine.get_scores(q)

    ranked = sorted(enumerate(scores), key=lambda x: -x[1])
    out: list[BM25Hit] = []
    for idx, score in ranked[:top_n]:
        if score <= 0:
            continue
        out.append(BM25Hit(corpus_index=idx, score=float(score)))
    return out


def rrf_fuse(
    rankings: list[list[str]],
    *,
    k: int = 60,
    top_n: int,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion across multiple ranked lists of chunk IDs.

    Classic formula: score(d) = sum over lists of 1/(k + rank_in_list(d)).
    k=60 is the value from the Cormack et al. 2009 paper; it's robust
    enough that I've never tuned it.
    """
    if not rankings:
        return []
    scores: dict[str, float] = {}
    for ranked_ids in rankings:
        for rank, doc_id in enumerate(ranked_ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: -x[1])
    return fused[:top_n]

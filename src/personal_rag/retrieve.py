"""Retrieve top-k chunks for a query from the index.

Two retrieval modes:
- Dense only (default): cosine similarity over fastembed vectors in LanceDB.
- Hybrid BM25 + dense: run both, fuse rankings with Reciprocal Rank Fusion.
  Better recall on queries where exact-term match matters (rare words,
  proper nouns, code identifiers) without losing the synonym-tolerance of
  dense retrieval. Enable with `settings.hybrid = True` or `hybrid=True`.
"""

from __future__ import annotations

from personal_rag.bm25 import bm25_rank, rrf_fuse
from personal_rag.config import Settings
from personal_rag.embed import embed_query, embedding_dim
from personal_rag.index import Index

# Over-fetch multiplier: ask each retriever for more candidates than we need,
# then let fusion narrow back down to top_k.
_OVERFETCH = 3


def _dense_search(index: Index, query_vector: list[float], *, k: int) -> list[dict]:
    return index.search(query_vector, k=k)


def retrieve(
    query: str,
    settings: Settings,
    *,
    k: int | None = None,
    hybrid: bool | None = None,
) -> list[dict]:
    """Return the top-k chunks most relevant to `query`."""
    if not query.strip():
        raise ValueError("Empty query.")

    k = k or settings.top_k
    hybrid = settings.hybrid if hybrid is None else hybrid

    dim = embedding_dim(settings.embedding_model)
    index = Index(settings.db_path, settings.table_name, dim)

    query_vec = embed_query(query, model_name=settings.embedding_model)

    if not hybrid:
        return _dense_search(index, query_vec, k=k)

    # Hybrid path: over-fetch from each retriever, then fuse via RRF.
    want = _OVERFETCH * k

    dense_rows = _dense_search(index, query_vec, k=want)
    dense_ids = [r["id"] for r in dense_rows]

    # BM25 runs over every chunk in the store. At low-thousands scale this is
    # fine; if the corpus grows too big we'd cache a tokenized matrix on disk
    # and invalidate it on any write to the index.
    all_rows = index.table.to_pandas()[["id", "text", "source", "chunk_index"]].to_dict("records")
    id_to_row = {r["id"]: r for r in all_rows}

    texts = [r["text"] for r in all_rows]
    ids_in_corpus = [r["id"] for r in all_rows]
    bm25_hits = bm25_rank(texts, query, top_n=want)
    bm25_ids = [ids_in_corpus[h.corpus_index] for h in bm25_hits]

    fused = rrf_fuse([dense_ids, bm25_ids], top_n=k)
    fused_rows: list[dict] = []
    for doc_id, _score in fused:
        row = id_to_row.get(doc_id)
        if row is None:
            continue
        fused_rows.append(
            {
                "id": row["id"],
                "source": row["source"],
                "chunk_index": row["chunk_index"],
                "text": row["text"],
            }
        )
    return fused_rows

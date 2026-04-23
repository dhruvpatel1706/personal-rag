"""Retrieve top-k chunks for a query from the index."""

from __future__ import annotations

from personal_rag.config import Settings
from personal_rag.embed import embed_query, embedding_dim
from personal_rag.index import Index


def retrieve(query: str, settings: Settings, *, k: int | None = None) -> list[dict]:
    """Return the top-k chunks most relevant to `query`."""
    if not query.strip():
        raise ValueError("Empty query.")
    dim = embedding_dim(settings.embedding_model)
    index = Index(settings.db_path, settings.table_name, dim)
    vec = embed_query(query, model_name=settings.embedding_model)
    return index.search(vec, k=k or settings.top_k)

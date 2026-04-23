"""Embedding via fastembed (ONNX-based, no torch dependency)."""

from __future__ import annotations

from functools import lru_cache

from fastembed import TextEmbedding


@lru_cache(maxsize=4)
def _load_model(name: str) -> TextEmbedding:
    return TextEmbedding(model_name=name)


def embed_texts(texts: list[str], *, model_name: str) -> list[list[float]]:
    """Embed a list of texts. Returns one vector per text."""
    if not texts:
        return []
    model = _load_model(model_name)
    return [vec.tolist() for vec in model.embed(texts)]


def embed_query(query: str, *, model_name: str) -> list[float]:
    """Embed a single query string."""
    model = _load_model(model_name)
    return next(iter(model.embed([query]))).tolist()


def embedding_dim(model_name: str) -> int:
    """Return the output dimension of the embedding model."""
    model = _load_model(model_name)
    sample = next(iter(model.embed(["hello"])))
    return int(sample.shape[0])

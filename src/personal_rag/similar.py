"""Find chunks or sources similar to a given chunk/source.

Two entry points, both on top of the existing LanceDB vector search:

1. `similar_to_chunk(chunk_id)` — "what else do I have that's close to this
   paragraph?" Returns the top-k chunks from OTHER sources (never the chunk
   itself or its siblings, which would be trivially at the top).

2. `similar_to_source(source)` — "what notes are related to this note?"
   Runs the chunk-level search for every chunk in the source, aggregates
   hits by target source with reciprocal rank, and returns ranked sources.
   Reciprocal-rank aggregation beats plain similarity-sum: one chunk that
   happens to have 10 strong hits doesn't drown out the rest of the note.
"""

from __future__ import annotations

from collections import defaultdict

from personal_rag.config import Settings
from personal_rag.embed import embedding_dim
from personal_rag.index import Index

# Per-chunk over-fetch when aggregating note-level similarity. We throw away
# anything from the source note itself, so we need a little headroom.
_PER_CHUNK_FETCH_MULTIPLIER = 3


def similar_to_chunk(
    settings: Settings,
    chunk_id: str,
    *,
    k: int = 5,
) -> list[dict]:
    """Top-k chunks closest to the given chunk, excluding any from the same source.

    Returns rows shaped like the rest of the retrieval code (id, source,
    chunk_index, text) so callers can format them the same way.
    """
    dim = embedding_dim(settings.embedding_model)
    index = Index(settings.db_path, settings.table_name, dim)
    seed = index.get_by_id(chunk_id)
    if seed is None:
        raise KeyError(f"no chunk with id {chunk_id!r} in the index")

    # Over-fetch because we're going to drop same-source hits (which will include
    # the seed chunk itself, trivially #1).
    raw = index.search(seed["vector"], k=k + _PER_CHUNK_FETCH_MULTIPLIER * 5)
    cleaned: list[dict] = []
    for row in raw:
        if row["source"] == seed["source"]:
            continue
        cleaned.append(row)
        if len(cleaned) >= k:
            break
    return cleaned


def similar_to_source(
    settings: Settings,
    source: str,
    *,
    k: int = 5,
) -> list[tuple[str, float]]:
    """Rank other sources by how related they are to this one.

    Returns `[(source, score), ...]` sorted by descending score. Score is the
    sum of reciprocal ranks across all chunk-level queries — the intuition is
    "a note that shows up in the top hits of many of this note's paragraphs
    is more related than one that's the single best match for one paragraph".
    """
    dim = embedding_dim(settings.embedding_model)
    index = Index(settings.db_path, settings.table_name, dim)
    seed_chunks = index.get_by_source(source)
    if not seed_chunks:
        raise KeyError(f"no chunks found for source {source!r}")

    per_source_score: dict[str, float] = defaultdict(float)
    per_chunk_budget = max(10, _PER_CHUNK_FETCH_MULTIPLIER * k)

    for chunk in seed_chunks:
        hits = index.search(chunk["vector"], k=per_chunk_budget)
        # Reciprocal rank aggregation, filtering out the seed source itself.
        rank = 0
        for row in hits:
            if row["source"] == source:
                continue
            rank += 1
            # Classic RRF constant — 60 is the number every retrieval paper
            # uses and it's fine for ranks in the single-digit-to-hundreds.
            per_source_score[row["source"]] += 1.0 / (60 + rank)

    ranked = sorted(per_source_score.items(), key=lambda item: item[1], reverse=True)
    return ranked[:k]

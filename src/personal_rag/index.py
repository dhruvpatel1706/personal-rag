"""LanceDB wrapper: create/open a table, upsert rows, search by vector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lancedb
import pyarrow as pa


@dataclass
class Row:
    """A single indexed chunk."""

    id: str  # f"{source}:{chunk_index}"
    source: str  # original file path
    chunk_index: int
    text: str
    vector: list[float]


def _schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("source", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ]
    )


class Index:
    """Thin wrapper over a single LanceDB table."""

    def __init__(self, db_path: Path, table_name: str, dim: int):
        db_path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(db_path))
        self.table_name = table_name
        self.dim = dim
        if table_name in self.db.table_names():
            self.table = self.db.open_table(table_name)
        else:
            self.table = self.db.create_table(table_name, schema=_schema(dim))

    def upsert(self, rows: list[Row]) -> int:
        """Replace any rows sharing the same `source`, then insert `rows`."""
        if not rows:
            return 0
        sources = {r.source for r in rows}
        for source in sources:
            self.table.delete(f"source = '{source.replace(chr(39), chr(39) * 2)}'")
        self.table.add(
            [
                {
                    "id": r.id,
                    "source": r.source,
                    "chunk_index": r.chunk_index,
                    "text": r.text,
                    "vector": r.vector,
                }
                for r in rows
            ]
        )
        return len(rows)

    def search(self, query_vector: list[float], *, k: int) -> list[dict]:
        """Return the top-`k` chunks by cosine similarity, plus distance."""
        return (
            self.table.search(query_vector)
            .limit(k)
            .select(["id", "source", "chunk_index", "text"])
            .to_list()
        )

    def count(self) -> int:
        return self.table.count_rows()

    def sources(self) -> list[str]:
        """List all distinct sources currently indexed."""
        return sorted(
            {row["source"] for row in self.table.to_pandas()[["source"]].to_dict("records")}
        )

    def remove_source(self, source: str) -> int:
        """Remove every chunk coming from `source`. Returns count before delete."""
        escaped = source.replace(chr(39), chr(39) * 2)
        before = self.count()
        self.table.delete(f"source = '{escaped}'")
        after = self.count()
        return before - after

    def get_by_id(self, chunk_id: str) -> dict | None:
        """Return the full row (including vector) for a chunk id, or None.

        Used by the `similar` flow — we need the chunk's own vector to query
        against the rest of the index.
        """
        df = self.table.to_pandas()
        rows = df[df["id"] == chunk_id]
        if rows.empty:
            return None
        row = rows.iloc[0]
        return {
            "id": row["id"],
            "source": row["source"],
            "chunk_index": int(row["chunk_index"]),
            "text": row["text"],
            "vector": list(row["vector"]),
        }

    def get_by_source(self, source: str) -> list[dict]:
        """All chunks with this source, vectors included, ordered by chunk index."""
        df = self.table.to_pandas()
        rows = df[df["source"] == source].sort_values("chunk_index")
        return [
            {
                "id": r["id"],
                "source": r["source"],
                "chunk_index": int(r["chunk_index"]),
                "text": r["text"],
                "vector": list(r["vector"]),
            }
            for _, r in rows.iterrows()
        ]

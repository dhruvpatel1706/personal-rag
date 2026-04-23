"""Read files from disk, chunk, embed, upsert into the index."""

from __future__ import annotations

import io
from pathlib import Path

from pypdf import PdfReader

from personal_rag.chunk import chunk_text
from personal_rag.config import Settings
from personal_rag.embed import embed_texts, embedding_dim
from personal_rag.index import Index, Row

SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".pdf"}


class IngestError(RuntimeError):
    """Raised when a file can't be read or indexed."""


def _read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(path.read_bytes()))
        parts = [(page.extract_text() or "").strip() for page in reader.pages]
        return "\n\n".join(p for p in parts if p)
    raise IngestError(f"Unsupported file type: {suffix} ({path})")


def _iter_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target] if target.suffix.lower() in SUPPORTED_SUFFIXES else []
    return sorted(
        p for p in target.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


def ingest(target: Path, settings: Settings) -> dict:
    """Ingest one file or a whole directory. Returns summary stats."""
    files = _iter_files(target)
    if not files:
        raise IngestError(f"No supported files found under {target}")

    dim = embedding_dim(settings.embedding_model)
    index = Index(settings.db_path, settings.table_name, dim)

    total_chunks = 0
    per_file: dict[str, int] = {}
    for path in files:
        try:
            text = _read_file(path)
        except Exception as exc:
            raise IngestError(f"Failed to read {path}: {exc}") from exc
        if not text.strip():
            continue
        chunks = chunk_text(text, size=settings.chunk_size, overlap=settings.chunk_overlap)
        if not chunks:
            continue
        vectors = embed_texts(chunks, model_name=settings.embedding_model)
        rows = [
            Row(
                id=f"{path}:{i}",
                source=str(path),
                chunk_index=i,
                text=chunk,
                vector=vec,
            )
            for i, (chunk, vec) in enumerate(zip(chunks, vectors))
        ]
        index.upsert(rows)
        per_file[str(path)] = len(chunks)
        total_chunks += len(chunks)

    return {
        "files_ingested": len(per_file),
        "chunks_total": total_chunks,
        "per_file": per_file,
    }

"""FastAPI server — wraps ingest + retrieve + generate behind HTTP endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from personal_rag import __version__
from personal_rag.config import get_settings
from personal_rag.generate import generate
from personal_rag.ingest import IngestError, ingest
from personal_rag.retrieve import retrieve

app = FastAPI(
    title="personal-rag",
    version=__version__,
    description="Local RAG over your documents. Retrieve + generate with Claude.",
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=25)


class Passage(BaseModel):
    source: str
    chunk_index: int
    text: str


class QueryResponse(BaseModel):
    answer: str
    passages: list[Passage]


class IngestRequest(BaseModel):
    path: str = Field(..., description="Local file or directory path.")


class IngestResponse(BaseModel):
    files_ingested: int
    chunks_total: int
    per_file: dict[str, int]


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "version": __version__,
        "model": settings.model,
        "embedding_model": settings.embedding_model,
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    settings = get_settings()
    passages = retrieve(req.question, settings, k=req.top_k)
    try:
        answer = generate(req.question, passages, settings)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return QueryResponse(
        answer=answer.text,
        passages=[
            Passage(source=p["source"], chunk_index=p["chunk_index"], text=p["text"])
            for p in passages
        ],
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest_endpoint(req: IngestRequest) -> IngestResponse:
    settings = get_settings()
    path = Path(req.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {req.path}")
    try:
        result = ingest(path, settings)
    except IngestError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return IngestResponse(**result)

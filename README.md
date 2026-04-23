# personal-rag

**Ask Claude questions against your own documents — no data leaves your machine except the final call.**

Ingests your markdown, text, and PDFs into a local LanceDB index. Embeddings run on-device via `fastembed` (ONNX, no torch, no GPU needed). Only the retrieved top-k chunks and your question are sent to Claude for the final answer. No cloud vector DB, no background syncing, no telemetry.

```
$ personal-rag ingest ~/notes
╭───────────── Ingest complete ──────────────╮
│ Indexed 47 files into 832 chunks.          │
╰────────────────────────────────────────────╯

$ personal-rag ask "what's the rationale for paper-only trading in algo-trader?"
╭─ Answer ───────────────────────────────────────────────────────────────────╮
│ The operator is on a US work visa that restricts self-employment,          │
│ and the IRS treats frequent trading as a business activity, so v1 of       │
│ the system is paper-only. Re-enabling live trading is gated by two         │
│ coordinated code changes (`docs/policy.md` + `src/execution/broker.py`)    │
│ that must land in the same reviewed PR.                                    │
│                                                                            │
│ Sources: Passage 1, Passage 2                                              │
╰────────────────────────────────────────────────────────────────────────────╯

                     Retrieved passages
  ┌────┬────────────────────────────────┬───────┬──────────────────┐
  │  # │ source                         │ chunk │ preview          │
  ├────┼────────────────────────────────┼───────┼──────────────────┤
  │  1 │ ~/notes/algo-trader-policy.md  │   3   │ Paper-only...    │
  │  2 │ ~/notes/algo-trader-policy.md  │   5   │ Re-enabling...   │
  │  3 │ ...                            │  ...  │ ...              │
  └────┴────────────────────────────────┴───────┴──────────────────┘
```

---

## What it does (and doesn't)

- ✅ **Runs locally.** Embeddings are computed on your machine with `fastembed` (ONNX-based, ~130 MB model, CPU-only is fine). LanceDB stores vectors in a local directory. Only the final question + top-k retrieved chunks hit the Claude API.
- ✅ **Incremental.** Re-ingesting a file replaces its chunks in-place. No duplicates.
- ✅ **Deterministic.** Same file → same chunks → same embeddings. Reproducible debugging.
- ✅ **Typed config.** `pydantic-settings` reads from `.env`, validates types, supports CLI overrides.
- ❌ **Not a vector hub.** Not meant to scale to millions of documents. Tens of thousands is fine.
- ❌ **Not OCR.** Scanned PDFs return empty text. Use a separate OCR pass first.
- ❌ **Not a chat app.** v0.1 is a CLI + HTTP API. No UI.

## Install

```bash
git clone https://github.com/dhruvpatel1706/personal-rag.git
cd personal-rag
pip install -e .
```

Requires Python 3.10+.

## Configure

```bash
cp .env.example .env
# add your ANTHROPIC_API_KEY
```

Get a key at [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys).

## Use

### CLI

```bash
# Ingest a file or a directory (recursive)
personal-rag ingest ~/notes
personal-rag ingest ~/papers/attention-is-all-you-need.pdf

# Ask questions
personal-rag ask "what does Ashish Vaswani argue about self-attention?"
personal-rag ask "what was my rationale for the 6% portfolio heat cap?" -k 8

# Without the retrieved passages table
personal-rag ask "summarize my notes on MLOps" --no-sources
```

### HTTP

```bash
personal-rag serve --port 8000

# In another shell:
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "what are the three phases?", "top_k": 5}'
```

Endpoints: `GET /health`, `POST /query`, `POST /ingest`. OpenAPI docs at `/docs`.

## How it works

1. **Ingest** — `ingest.py` walks the input path, reads supported files (`.md`, `.txt`, `.pdf`), chunks each at `CHUNK_SIZE` chars with `CHUNK_OVERLAP` overlap on sentence boundaries, embeds every chunk with fastembed, and upserts rows into LanceDB at `DB_PATH`.
2. **Retrieve** — `retrieve.py` embeds the question with the same model and runs a cosine similarity search on the LanceDB table. Returns the top-k chunks.
3. **Generate** — `generate.py` sends the question + retrieved context to Claude (default `claude-opus-4-7`) with adaptive thinking. The system prompt is cached with `ephemeral` cache_control so repeated asks reuse the prefix.

## Design choices

- **fastembed, not sentence-transformers.** fastembed is ONNX-backed, skips the 1GB PyTorch dependency, and runs fine on CPU. The default model (`BAAI/bge-small-en-v1.5`, 384 dim) is competitive with MiniLM and a little better on retrieval benchmarks.
- **LanceDB, not pgvector / Qdrant / Chroma.** LanceDB is embedded (no server), file-based, single-binary, and columnar. For a personal index it's the path of least operational pain.
- **Upsert semantics keyed on source path.** Re-ingesting the same file is safe — old chunks are deleted, new ones inserted. No accidental duplicate explosion.
- **Prompt caching on the system prompt.** The retrieval-QA instructions don't change between calls. `cache_control: ephemeral` makes the 2nd-through-nth call meaningfully cheaper.
- **Adaptive thinking.** Lets Claude decide how much reasoning a given question warrants; no fixed `budget_tokens` to tune.

## Development

```bash
pip install -e ".[dev]"
pytest                                                      # unit tests
black --check src tests
isort --check-only --profile black src tests
flake8 src tests --max-line-length=100 --ignore=E501,W503,E203

# One-shot smoke: ingest and ask against this repo's README
personal-rag ingest README.md
personal-rag ask "what does this project avoid doing?"
```

CI runs on Python 3.10 / 3.11 / 3.12.

## Roadmap

- [ ] v0.2 — hybrid BM25 + dense retrieval for better recall on rare terms
- [ ] v0.3 — per-chunk citations in the answer text (not just a trailing list)
- [ ] v0.4 — incremental indexing via a watch mode (`personal-rag watch ~/notes`)
- [ ] v0.5 — simple web UI (FastAPI + HTMX)

## License

MIT. See [LICENSE](LICENSE).

# RAGify

A Retrieval-Augmented Generation (RAG) system built from scratch to learn how RAG pipelines work end-to-end — file upload → embedding → hybrid retrieval → re-ranking → LLM answer generation, restricted strictly to the uploaded context.

Built with **Python**, **FastAPI**, **ChromaDB**, **Postgres (Neon)**, **sentence-transformers**, and **NVIDIA NIM (build.nvidia.com)** free-tier LLMs.

---

## Goal

- Upload a document → chunk it (structure-aware) → embed it → store in a vector database + full-text search database
- Ask a question → retrieve relevant chunks via hybrid search (semantic + keyword) → re-rank → send to an LLM
- LLM answers **only** using the retrieved context (no hallucination / no going out-of-context)

---

## Tech Stack

| Component         | Choice                                                          | Why                                                                                      |
| ----------------- | --------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Backend framework | FastAPI (fully async)                                           | Async, fast, easy to build APIs                                                          |
| Vector DB         | ChromaDB Cloud, cosine similarity                               | Managed, zero local infra                                                                |
| Keyword search DB | PostgreSQL (Neon, serverless free tier)                         | Full-text search (`tsvector`/`tsquery`) for exact-term matches vector search misses      |
| Embedding model   | `sentence-transformers` (`all-MiniLM-L6-v2`), local, normalized | No rate limits, free, fast for bulk chunking                                             |
| Re-ranker         | `cross-encoder/ms-marco-MiniLM-L-6-v2`                          | Scores (question, chunk) pairs jointly for higher precision than cosine similarity alone |
| LLM               | NVIDIA NIM — `meta/llama-3.2-11b-vision-instruct`               | Free tier; swapped from 3.1-8b after reliability issues                                  |
| File parsing      | `pypdf` (.pdf), `python-docx` (.docx), built-in decode (.txt)   | Lightweight, sufficient for text-based documents                                         |
| Frontend          | Plain HTML/JS, served via FastAPI StaticFiles                   | Kept focus on RAG pipeline, not frontend tooling                                         |
| Env management    | `venv`                                                          | Simple, standard                                                                         |

---

## Architecture

```
RAGify/
├── app/
│ ├── main.py # FastAPI app entrypoint — /upload, /ask, /ask/stream, frontend route
│ ├── ingestion.py # File parsing + structure-aware chunking + embedding + storing + dedup
│ ├── retrieval.py # Hybrid retrieval: vector + keyword search, RRF fusion, cross-encoder re-rank
│ ├── postgres_search.py # Postgres full-text search (index + keyword search)
│ └── llm.py # NVIDIA NIM API call (async streaming via httpx), prompt construction
├── static/
│ └── index.html # Browser frontend (upload + ask + ask-streaming)
├── eval/
│ ├── eval_set.json # Regression test questions against the Attention Is All You Need paper
│ └── run_eval.py # Runs retrieval against eval_set.json, reports accuracy
├── tests/ # Manual verification scripts (Postgres connection, RRF logic, pipeline stages)
├── .env # API keys + DB connection strings (NEVER commit this)
├── .gitignore
├── requirements.txt
└── README.md
```

### Pipeline

**Ingestion:** `file upload → parse text (.txt/.pdf/.docx) → detect section boundaries (numbered headers) → chunk each section (word-based, with overlap) → embed each chunk (local model, normalized) → dedup (file-scoped content-hash IDs) → store in ChromaDB (vectors) AND Postgres (full-text)`

**Query (non-streaming, `/ask`):** `question → [vector search (ChromaDB, cosine, top 20) + keyword search (Postgres, full-text, top 20)] → merge via Reciprocal Rank Fusion → re-rank fused candidates with cross-encoder → top 3 → build prompt → call NVIDIA LLM → return {answer, sources}`

**Query (streaming, `/ask/stream`):** Same hybrid retrieval + prompt steps, but the LLM's response is streamed token-by-token to the browser as it's generated.

All blocking calls (embedding, Postgres queries via `asyncio.run`, cross-encoder inference) run inside `run_in_threadpool` so they never block FastAPI's event loop.

---

## Progress Tracker

### Phase 1 — Core Concepts ✅

- [x] Embeddings, chunking, vector databases

### Phase 2 — Environment Setup ✅

- [x] Python venv, NVIDIA NIM API key, dependencies, `.env`, folder structure

### Phase 3 — Ingestion Pipeline ✅

- [x] File upload endpoint, parsing (.txt/.pdf/.docx), chunking, embedding, storage

### Phase 4 — Query Pipeline ✅

- [x] `/ask` endpoint, embed question, retrieve, build prompt, call LLM

### Phase 5 — Guardrails ✅

- [x] Context-only system prompt, similarity threshold fallback, input validation

### Phase 6 — Polish / Extend ✅

- [x] .docx support, dedup, minimal frontend, streaming responses
- [x] Migrated to async `httpx` streaming + `run_in_threadpool` for all blocking calls
- [x] Migrated ChromaDB → Chroma Cloud
- [x] Normalized embeddings on both ingestion and query sides (fixed cosine-distance mismatch bug)
- [x] File-scoped chunk IDs (`filename#chunk_i_hash`) — fixes cross-file ID collisions
- [x] **Cross-encoder re-ranking** — wide vector retrieval (top 20) + cross-encoder re-rank → top 3
- [x] **Eval harness** (`eval/run_eval.py`) — 12-question regression suite against a real document
- [x] **Hybrid search** — Postgres full-text search alongside vector search, merged via Reciprocal Rank Fusion
- [x] **Structure-aware chunking** — detects numbered section headers (e.g. "5.3 Optimizer") so chunks don't mix unrelated subsections
- [ ] Multi-turn chat history
- [ ] Logging + structured error handling
- [ ] Rate limiting
- [ ] Automated test suite (pytest)
- [ ] Deploy live (Render/Railway/etc.)

---

## Setup Instructions

```bash
# 1. Clone / open project folder, switch to the working branch
git checkout release/dev

# 2. Activate virtual environment
source venv/Scripts/activate   # Git Bash on Windows
# or venv\Scripts\activate.bat on CMD

# 3. Install dependencies (always use python -m pip to avoid venv/pip PATH mismatches)
python -m pip install -r requirements.txt

# 4. Add keys to .env
NVIDIA_API_KEY=your_key_here
CHROMA_API_KEY=your_key_here
CHROMA_TENANT=your_tenant_here
CHROMA_DATABASE=your_database_here
DATABASE_URL=your_neon_connection_string_here

# 5. Create the Postgres chunks table (run once, in Neon's SQL Editor)
CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    source TEXT NOT NULL,
    search_vector TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);
CREATE INDEX idx_search_vector ON chunks USING GIN(search_vector);

# 6. Run the server
uvicorn app.main:app --reload

# 7. Open in browser
# http://127.0.0.1:8000/

# 8. Run the eval harness (optional, verifies retrieval quality)
python -m eval.run_eval
```

---

## Key Decisions & Notes

- Migrated from local `PersistentClient` to **ChromaDB Cloud**, and from no keyword search to **Postgres (Neon) full-text search** — enables hybrid retrieval without managing local infra.
- Configured ChromaDB to use **cosine similarity** instead of default L2 — L2 has no fixed range, making threshold filtering meaningless.
- **Normalized embeddings on both ingestion and query sides.** Originally only the query side normalized — silently distorted cosine distance calculations until fixed.
- **Switched chunk ID scheme** from pure content-hash (`md5(text)`) to file-scoped (`filename#chunk_i_hash10`). The old scheme caused identical text across different files to silently overwrite each other's metadata. Tradeoff: lost automatic cross-file deduplication in exchange for correct per-file traceability.
- **Known limitation:** uploading a file with the same name as an existing one deletes the old file's chunks entirely, in both ChromaDB and Postgres, with no collision protection. Safe only because filenames are currently assumed unique per upload.
- **Added cross-encoder re-ranking** (`cross-encoder/ms-marco-MiniLM-L-6-v2`): retrieve top 20 candidates cheaply via vector similarity, then re-score each (question, chunk) pair jointly for much higher precision before keeping the final top 3.
- **Added hybrid search**: Postgres full-text search runs alongside ChromaDB vector search; results are merged via Reciprocal Rank Fusion (RRF) before re-ranking. Chosen over Elasticsearch (real recurring cost, more operational complexity) and `rank_bm25` (no incremental indexing — full corpus rebuild on every upload, doesn't scale).
- **Postgres full-text search uses `'english'` config** (stemming + stopwords), not `'simple'` — appropriate for the current English-only corpus; will need per-language handling (`'simple'` config, or language detection) if non-English documents are added.
- **Postgres query uses OR-based matching** (`to_tsquery` with keywords joined by `|`), not the default AND-based matching (`plainto_tsquery`). AND-based matching caused zero results whenever a question's exact wording (e.g. "Transformer") didn't appear in the same chunk as the answer (e.g. "Adam optimizer") — a real bug found via the eval harness.
- **asyncpg connections are opened and closed per-call**, not pooled. A persistent connection pool created inside one `asyncio.run()` call became invalid once that call's temporary event loop closed, causing `InterfaceError: cannot perform operation` on subsequent calls. Since `retrieve_relevant_chunks` is sync and wrapped in `run_in_threadpool` (each call getting its own `asyncio.run()`), per-call connections are the correct fix given this architecture.
- **Added structure-aware chunking**: detects numbered section headers (e.g. "5.3 Optimizer") via regex and never lets a chunk span two detected sections. Fixes a real failure found via the eval harness, where a chunk mixing "5.2 Hardware and Schedule" with "5.3 Optimizer" scored -5.14 on the cross-encoder (strongly irrelevant) despite containing the correct answer — irrelevant surrounding content diluted the chunk's apparent relevance. Falls back to word-count-only chunking for text with no detected headers (e.g. Abstract, Introduction). **Known limitation:** only catches `number.number Header` patterns; single-number headers ("1 Introduction") and unheaded sections are not detected.
- **Built an eval harness** (`eval/eval_set.json` + `eval/run_eval.py`) — a small (12-question), growing regression suite, not full document coverage. Used to diagnose both the AND/OR query bug and the chunk topic-mixing issue with real evidence rather than guesswork. Current score: 12/12 (100%) on the Attention Is All You Need paper.
- Rejected `meta/llama-3.2-90b-vision-instruct` for cost reasons; later switched the default model to `meta/llama-3.2-11b-vision-instruct` after the original `meta/llama-3.1-8b-instruct` had reliability issues.
- Migrated `requests` → `httpx` for true async LLM streaming, and wrapped all blocking calls (`embed_chunks`, `store_chunks`, `retrieve_relevant_chunks`) in `run_in_threadpool` — without this, a slow embedding/rerank call would freeze the server for every concurrent user.

---

## Learnings Log

- Embeddings capture meaning, not exact words — cosine similarity: "cat/mat" vs "feline/rug" scored 0.556, vs "stock market" scored 0.097.
- Chunk overlap matters — 200-word chunks with 50-word overlap on 1000 words produced 7 chunks, matching manual math.
- Distance metrics matter: cosine distance must be explicitly configured for threshold-based filtering to work; L2 has no fixed range.
- `python -m pip` / `python -m module` avoids Windows venv PATH mismatches — confirmed the hard way when bare `pip install` silently installed into the global Python instead of the active venv, causing a `ModuleNotFoundError` that looked like a code bug but was an environment issue.
- Generators (`yield`) don't run until iterated — the same generator function can power both a "wait for everything" flow and a "forward pieces immediately" flow.
- A bi-encoder (embedding model) and a cross-encoder (re-ranker) solve different problems: bi-encoders embed question and chunk _separately_ for fast approximate search across a whole collection; cross-encoders score them _jointly_ for precision, but are too slow to run against everything — hence retrieve-wide-then-rerank-narrow as the standard pattern.
- `asyncpg` connection pools are bound to the event loop that created them — reusing a pool across multiple separate `asyncio.run()` calls (each with its own temporary event loop) causes `InterfaceError`/`Event loop is closed`, since the pool's underlying connections are tied to an event loop that no longer exists by the second call.
- Postgres full-text search defaults to AND logic between query words (`plainto_tsquery`) — a question and its answer can both be about the same topic yet share zero common significant words, causing zero search results even though the right content exists. OR-based matching (`to_tsquery` with `|`) fixes this at the cost of needing the cross-encoder to filter out the resulting looser matches.
- An eval harness doesn't need per-document coverage — a small (~20-40 question), growing, representative sample is enough to catch systemic retrieval/chunking problems, and it should grow from real observed failures over time rather than being written exhaustively upfront.
- A cross-encoder can score a chunk very negatively even when it contains the literal correct answer, if the chunk also contains a lot of unrelated content (e.g. two merged subsections) — this is a chunking problem disguised as a retrieval problem, only diagnosable by inspecting actual chunk text and actual scores rather than guessing from pass/fail alone.

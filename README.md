# RAGify

**Live:** https://ragify-zb6i.onrender.com (hosted on Render)

A Retrieval-Augmented Generation (RAG) system built from scratch to learn how RAG pipelines work end-to-end — file upload → embedding → hybrid retrieval → re-ranking → LLM answer generation, restricted strictly to the uploaded context, with conversation memory for follow-up questions.

Built with **Python**, **FastAPI**, **ChromaDB**, **Postgres (Neon)**, and two interchangeable model stacks selected by one environment variable:

- **dev** — local `sentence-transformers` models + **NVIDIA NIM (build.nvidia.com)** free-tier LLMs
- **prod** — **Gemini API** for embeddings, re-ranking, and answers (no local ML models)

---

## Goal

- Upload a document → chunk it (structure-aware) → embed it → store in a vector database + full-text search database
- Ask a question → rewrite follow-ups using conversation memory → retrieve relevant chunks via hybrid search (semantic + keyword) → re-rank → send to an LLM
- LLM answers **only** using the retrieved context (no hallucination / no going out-of-context)

---

## Tech Stack

| Component           | Dev                                                             | Prod                                                                  | Why                                                                                 |
| ------------------- | --------------------------------------------------------------- | --------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| Backend framework   | FastAPI (fully async)                                           | same                                                                  | Async, fast, easy to build APIs                                                     |
| Vector DB           | ChromaDB Cloud, cosine similarity                               | same                                                                  | Managed, zero local infra                                                           |
| Keyword search DB   | PostgreSQL (Neon, serverless free tier)                         | same                                                                  | Full-text search (`tsvector`/`tsquery`) for exact-term matches vector search misses |
| Embedding model     | `all-MiniLM-L6-v2` (`sentence-transformers`), local, normalized | Gemini `gemini-embedding-001` (API)                                   | Dev: no rate limits, free. Prod: no local model in memory                           |
| Re-ranker           | Cross-encoder `ms-marco-MiniLM-L-6-v2`, local                   | Gemini `gemini-3.5-flash-lite`, listwise (LLM returns ranked indices) | Scores relevance more precisely than cosine similarity alone                        |
| Answer LLM          | NVIDIA NIM — `meta/llama-3.2-11b-vision-instruct`               | Gemini `gemini-3.5-flash-lite`                                        | Free tiers; same interface in both modes                                            |
| Conversation memory | In-process (`app/memory.py`)                                    | same                                                                  | Follow-up questions + rolling summary, no extra infra                               |
| File parsing        | `pypdf` (.pdf), `python-docx` (.docx), built-in decode (.txt)   | same                                                                  | Lightweight, sufficient for text-based documents                                    |
| Frontend            | Plain HTML/JS, served via FastAPI StaticFiles                   | same                                                                  | Kept focus on RAG pipeline, not frontend tooling                                    |
| Env management      | `venv` + `.env`                                                 | platform environment variables                                        | Simple, standard                                                                    |

Model names are copied from the code; confirm them against your providers' current model lists when you change them.

---

## Environments (dev vs prod)

Controlled by the `ENVIRONMENT` variable. **Only the exact value `prod` switches to prod**; anything else (including unset) runs dev.

|                         | dev                                                                        | prod                                                                                                                                                              |
| ----------------------- | -------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Requirements file       | `requirements-dev.txt` — full `chromadb`, `sentence-transformers`, PyTorch | `requirements.txt` (the file the Dockerfile installs) — thin `chromadb-client`, `google-genai`; no PyTorch (local-model imports are lazy, so prod never loads it) |
| LLM retry/backoff       | Yes (3 attempts, 1s then 2s waits)                                         | No                                                                                                                                                                |
| Extra required env vars | `NVIDIA_API_KEY`, `NVIDIA_URL`                                             | `GEMINI_API_KEY`                                                                                                                                                  |

**Keep the two environments' data separate:**

- The embedding models have different dimensions, so dev and prod cannot share a Chroma collection. Set a different `CHROMA_COLLECTION_NAME` for each.
- Use a separate Postgres database (or Neon branch) per environment. The keyword index is keyed by filename, and re-uploading a file deletes that filename's rows, so a shared database lets one environment wipe the other's data.

---

## Architecture

```
RAGify/
├── app/
│   ├── main.py             # FastAPI entrypoint — /upload, /ask, /ask/stream, /memory/{id}/summary, frontend route
│   ├── ingestion.py        # File parsing + structure-aware chunking + embedding + storing (Chroma + Postgres)
│   ├── retrieval.py        # Hybrid retrieval: vector + keyword search, RRF fusion, re-rank
│   ├── postgres_search.py  # Postgres full-text search (index + OR-based keyword search, asyncpg)
│   ├── services.py         # Environment-selected embedder + re-ranker (dev: local, prod: Gemini)
│   ├── llm.py              # Prompt construction, answer streaming (NVIDIA or Gemini), query rewriting
│   ├── memory.py           # In-process conversation memory (recent turns + rolling summary)
│   ├── summarizer.py       # Background task that condenses older turns into the summary
│   ├── env_check.py        # Startup validation of required environment variables
│   └── logging_config.py   # One-time logging setup
├── static/
│   └── index.html          # Browser frontend (upload + ask + ask-streaming)
├── eval/
│   ├── eval_set.json       # Regression test questions against the Attention Is All You Need paper
│   └── run_eval.py         # Runs retrieval against eval_set.json, reports accuracy
├── tests/                  # Manual verification scripts (embedding, LLM, Postgres connection, RRF logic, pipeline stages)
├── .env                    # API keys + DB connection strings (NEVER commit this)
├── .gitignore
├── requirements.txt
└── README.md
```

### Pipeline

**Ingestion:** `file upload → parse text (.txt/.pdf/.docx) → detect section boundaries (numbered headers) → chunk each section (200 words, 50-word overlap, sentence-aware) → embed each chunk (MiniLM in dev, Gemini in prod) → file-scoped chunk IDs → store in ChromaDB (vectors) AND Postgres (full-text)`

**Query (non-streaming, `/ask`):** `question + conversation_id → load memory (summary + recent turns) → rewrite follow-up into a standalone search query → [vector search (ChromaDB, cosine, top 20, similarity ≥ 0.3) + keyword search (Postgres, full-text, top 20)] → merge via Reciprocal Rank Fusion → re-rank fused candidates → top 3 → build prompt (rules + summary + recent turns + chunks + question) → LLM → save turn → return {answer, sources}`

**Query (streaming, `/ask/stream`):** Same memory, retrieval, and prompt steps, but the LLM's response is streamed to the browser as it's generated. The turn is saved to memory once the stream finishes.

**Conversation memory:** Held in process memory only. Up to 500 conversations, expiring after 1 hour of inactivity. The last 4 turns are kept verbatim; older turns are condensed into a rolling summary by a background task (`summarizer.py`), with a version check so a stale summary can't overwrite a newer one. The rewritten query is used for retrieval; the user's original wording goes into the final prompt.

All blocking calls (embedding, Postgres queries via `asyncio.run`, re-ranking) run inside `run_in_threadpool` so they never block FastAPI's event loop.

---

## API

| Method | Path                                | Parameters                    | Description                                                   |
| ------ | ----------------------------------- | ----------------------------- | ------------------------------------------------------------- |
| `GET`  | `/`                                 |                               | Serves the frontend                                           |
| `POST` | `/upload`                           | multipart `file`              | Ingest a document (max 10 MB)                                 |
| `POST` | `/ask`                              | `question`, `conversation_id` | Full answer as JSON: `{question, answer, sources}`            |
| `POST` | `/ask/stream`                       | `question`, `conversation_id` | Answer streamed as `text/plain`                               |
| `GET`  | `/memory/{conversation_id}/summary` |                               | Current rolling summary + number of recent turns held (max 4) |

`conversation_id` is required. Any unguessable string works; `ConversationMemory.new_conversation_id()` can generate one, but no endpoint exposes it, so the client creates its own ID and reuses it for the whole chat.

```bash
curl -X POST -F "file=@paper.pdf" http://127.0.0.1:8000/upload

curl -X POST "http://127.0.0.1:8000/ask?question=What%20optimizer%20was%20used%3F&conversation_id=my-chat-1"

curl -N -X POST "http://127.0.0.1:8000/ask/stream?question=Why%20that%20one%3F&conversation_id=my-chat-1"
```

Error codes on `/ask`: `400` empty question / unsupported or empty file, `413` file too large, `500` retrieval or unexpected failure, `502` provider HTTP error, `503` provider unreachable, `504` provider timeout. The 502/503/504 mappings come from `httpx` exceptions (the NVIDIA path); Gemini SDK errors currently surface as a generic `500`. On `/ask/stream` the status code is already sent when streaming starts, so errors appear as an `[Error: ...]` line in the stream.

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

### Phase 6 — Retrieval Quality ✅

- [x] .docx support, dedup, minimal frontend, streaming responses
- [x] Migrated to async `httpx` streaming + `run_in_threadpool` for all blocking calls
- [x] Migrated ChromaDB → Chroma Cloud
- [x] Normalized embeddings on both ingestion and query sides (fixed cosine-distance mismatch bug)
- [x] File-scoped chunk IDs (`filename#chunk_i_hash`) — fixes cross-file ID collisions
- [x] **Cross-encoder re-ranking** — wide vector retrieval (top 20) + re-rank → top 3
- [x] **Eval harness** (`eval/run_eval.py`) — 12-question regression suite against a real document
- [x] **Hybrid search** — Postgres full-text search alongside vector search, merged via Reciprocal Rank Fusion
- [x] **Structure-aware chunking** — detects numbered section headers (e.g. "5.3 Optimizer") so chunks don't mix unrelated subsections

### Phase 7 — Conversations & Reliability ✅

- [x] **Multi-turn chat** — conversation memory, follow-up query rewriting, rolling summary of older turns
- [x] **Structured logging** across upload, retrieval, and answer paths
- [x] **Startup environment validation** — clear error naming any missing variable
- [x] **Retry with exponential backoff** on the NVIDIA call (only before any output is sent)
- [x] Error handling around retrieval and LLM calls with correct HTTP status codes

### Phase 8 — Dev/Prod & Deployment ✅

- [x] Dev/prod split behind one interface (`app/services.py`, `ENVIRONMENT` switch)
- [x] Gemini embeddings, listwise re-ranking, answers, query rewriting, and summarization for prod
- [x] Production deployment

### Next

- [ ] Authentication + rate limiting
- [ ] Persistent conversation memory (survives restarts, shared across workers)
- [ ] Batch embedding requests in prod
- [ ] Broader section-header detection + larger, multi-document eval set
- [ ] Automated test suite (pytest)

---

## Setup Instructions

```bash
# 1. Clone / open project folder, switch to the working branch
git checkout release/dev

# 2. Activate virtual environment
source venv/Scripts/activate   # Git Bash on Windows
# or venv\Scripts\activate.bat on CMD

# 3. Install dependencies (always use python -m pip to avoid venv/pip PATH mismatches)
#    Dev uses requirements-dev.txt (local models). The prod image installs requirements.txt.
#    To try ENVIRONMENT=prod locally, also run: python -m pip install google-genai
python -m pip install -r requirements-dev.txt

# 4. Add config to .env  (KEY=value — no spaces around "=", no quotes)
ENVIRONMENT=dev
NVIDIA_API_KEY=your_key_here
NVIDIA_URL=https://integrate.api.nvidia.com/v1/chat/completions
CHROMA_API_KEY=your_key_here
CHROMA_TENANT=your_tenant_here
CHROMA_DATABASE=your_database_here
CHROMA_COLLECTION_NAME=ragify_docs_dev
DATABASE_URL=your_neon_connection_string_here

# 5. Create the Postgres chunks table (run once per database, in Neon's SQL Editor)
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

### Environment variables

| Variable                 | Dev              | Prod     | Notes                                                           |
| ------------------------ | ---------------- | -------- | --------------------------------------------------------------- |
| `ENVIRONMENT`            | `dev` (or unset) | `prod`   | Exact value `prod` switches modes                               |
| `CHROMA_API_KEY`         | required         | required | Chroma Cloud                                                    |
| `CHROMA_TENANT`          | required         | required |                                                                 |
| `CHROMA_DATABASE`        | required         | required |                                                                 |
| `CHROMA_COLLECTION_NAME` | optional         | optional | Defaults to `ragify_docs`; use a different name per environment |
| `DATABASE_URL`           | required         | required | Postgres (Neon); use a separate database per environment        |
| `NVIDIA_API_KEY`         | required         |          |                                                                 |
| `NVIDIA_URL`             | required         |          | `https://integrate.api.nvidia.com/v1/chat/completions`          |
| `GEMINI_API_KEY`         |                  | required |                                                                 |

On startup, `validate_env()` checks the variables required for the current mode and names any that are missing.

---

## Deployment (prod)

- **Platform:** Render (Docker web service)
- **Live URL:** https://ragify-zb6i.onrender.com
- **Image:** built from the `Dockerfile` — `python:3.12-slim`, installs `requirements.txt` (no PyTorch), runs as a non-root user
- **Start command:** set in the Dockerfile — `uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1` (binds to the `$PORT` Render provides, 8000 locally)
- **Environment variables:** set in the Render dashboard — the prod column above, with `ENVIRONMENT=prod`
- **Workers:** exactly one. Conversation memory lives in process memory, so with multiple workers or instances a follow-up can land on a process that has never seen the conversation. One worker also keeps memory use inside small plans' limits.
- **Free-plan note:** if the service is on Render's free plan, it sleeps when idle, so the first request after a pause is slow and in-memory conversation history is lost when it restarts.

Run the production image locally:

```bash
docker build -t ragify .
docker run --env-file <your-prod-env-file> -e ENVIRONMENT=prod -p 8000:8000 ragify
```

---

## Key Decisions & Notes

- Migrated from local `PersistentClient` to **ChromaDB Cloud**, and from no keyword search to **Postgres (Neon) full-text search** — enables hybrid retrieval without managing local infra.
- Configured ChromaDB to use **cosine similarity** instead of default L2 — L2 has no fixed range, making threshold filtering meaningless.
- **Normalized embeddings on both ingestion and query sides.** Originally only the query side normalized — silently distorted cosine distance calculations until fixed.
- **Switched chunk ID scheme** from pure content-hash (`md5(text)`) to file-scoped (`filename#chunk_i_hash10`). The old scheme caused identical text across different files to silently overwrite each other's metadata. Tradeoff: lost automatic cross-file deduplication in exchange for correct per-file traceability.
- **Known limitation:** uploading a file with the same name as an existing one deletes the old file's chunks entirely, in both ChromaDB and Postgres, with no collision protection. Safe only because filenames are currently assumed unique per upload.
- **Added re-ranking:** retrieve top 20 candidates cheaply, then re-score each (question, chunk) pair jointly for much higher precision before keeping the final top 3. Dev uses a local cross-encoder; prod uses a Gemini listwise re-ranker (the LLM returns a ranked list of candidate indices, falling back to RRF order on failure).
- **Added hybrid search**: Postgres full-text search runs alongside ChromaDB vector search; results are merged via Reciprocal Rank Fusion (RRF) before re-ranking. Chosen over Elasticsearch (real recurring cost, more operational complexity) and `rank_bm25` (no incremental indexing — full corpus rebuild on every upload, doesn't scale).
- **Postgres full-text search uses `'english'` config** (stemming + stopwords), not `'simple'` — appropriate for the current English-only corpus; will need per-language handling (`'simple'` config, or language detection) if non-English documents are added.
- **Postgres query uses OR-based matching** (`to_tsquery` with keywords joined by `|`), not the default AND-based matching (`plainto_tsquery`). AND-based matching caused zero results whenever a question's exact wording (e.g. "Transformer") didn't appear in the same chunk as the answer (e.g. "Adam optimizer") — a real bug found via the eval harness.
- **asyncpg connections are opened and closed per-call**, not pooled. A persistent connection pool created inside one `asyncio.run()` call became invalid once that call's temporary event loop closed, causing `InterfaceError: cannot perform operation` on subsequent calls. Since `retrieve_relevant_chunks` and `store_chunks` are sync and wrapped in `run_in_threadpool` (each call getting its own `asyncio.run()`), per-call connections are the correct fix given this architecture.
- **Added structure-aware chunking**: detects numbered section headers (e.g. "5.3 Optimizer") via regex and never lets a chunk span two detected sections. Fixes a real failure found via the eval harness, where a chunk mixing "5.2 Hardware and Schedule" with "5.3 Optimizer" scored -5.14 on the cross-encoder (strongly irrelevant) despite containing the correct answer — irrelevant surrounding content diluted the chunk's apparent relevance. Falls back to word-count-only chunking for text with no detected headers (e.g. Abstract, Introduction). **Known limitation:** only catches `number.number Header` patterns; single-number headers ("1 Introduction") and unheaded sections are not detected.
- **Built an eval harness** (`eval/eval_set.json` + `eval/run_eval.py`) — a small (12-question), growing regression suite, not full document coverage. Used to diagnose both the AND/OR query bug and the chunk topic-mixing issue with real evidence rather than guesswork. Score history on the dev stack: 9/12 (75%) with hybrid search + re-ranking → 11/12 (91.7%) after structure-aware chunking → 12/12 (100%) after correcting one test assertion's expected phrase. Re-run against the prod configuration before trusting prod numbers.
- **Conversation memory is in-process, with a rolling summary.** The last 4 turns stay verbatim; older turns are condensed by a background task, committed with a version check so a slow summarization can't overwrite a newer summary. Follow-up questions are rewritten into standalone search queries (using the last 3 turns + summary), falling back to the original question if rewriting fails. Chosen over a database-backed store to avoid extra infrastructure; the cost is that memory is lost on restart and isn't shared across workers.
- **Dev/prod split behind one interface** (`app/services.py`): the embedder and re-ranker are chosen once at import. Local-model imports are lazy, so prod never loads PyTorch and stays light on memory.
- **Startup validation runs before any module that creates network clients.** `ingestion.py` connects to Chroma at import time, so `main.py` calls `load_dotenv()` and `validate_env()` first, giving a clear error naming the missing variable instead of a confusing library crash.
- **Retry only before any output has been sent.** A mid-stream retry would re-yield text already delivered and duplicate the answer. Only timeouts and connection errors are retried; 4xx/5xx responses (e.g. a bad API key) are not.
- Rejected `meta/llama-3.2-90b-vision-instruct` for cost reasons; later switched the dev default to `meta/llama-3.2-11b-vision-instruct` after the original `meta/llama-3.1-8b-instruct` had reliability issues.
- Migrated `requests` → `httpx` for true async LLM streaming, and wrapped all blocking calls (`embed_chunks`, `store_chunks`, `retrieve_relevant_chunks`) in `run_in_threadpool` — without this, a slow embedding/rerank call would freeze the server for every concurrent user.

---

## Known Limitations

- **Same filename = overwrite** in both Chroma and Postgres (see above).
- **Section detection is narrow:** only `N.N Title` headers; everything else falls back to word-count chunking, which can still mix topics in a chunk.
- **English-only keyword search** (`'english'` full-text config).
- **Conversation memory is in-process:** lost on restart, not shared across workers, expires after an hour. If more than 20 turns pile up unsummarized (for example, if the summarizer keeps failing), `add_turn` raises.
- **No authentication or rate limiting** on any endpoint, including `/upload` and `/memory/{id}/summary`.
- **Prod re-ranker sees only the first 400 characters of each chunk,** so an answer late in a long chunk can be ranked down.
- **Prod embeddings are requested one chunk per API call,** so large uploads are slow and rate-limit-prone.
- **The 0.3 similarity threshold was tuned on MiniLM;** Gemini embeddings have a different similarity distribution, so re-check it in prod.
- **Retry/backoff exists only on the NVIDIA (dev) path;** Gemini errors surface as a generic `500`.
- **Text-based files only:** scanned PDFs are not OCR'd; max file size is 10 MB.
- **Small eval set:** 12 questions, one document, measured on the dev stack.

---

## Learnings Log

- Embeddings capture meaning, not exact words — cosine similarity: "cat/mat" vs "feline/rug" scored 0.556, vs "stock market" scored 0.097.
- Chunk overlap matters — 200-word chunks with 50-word overlap on 1000 words produced 7 chunks, matching manual math.
- Distance metrics matter: cosine distance must be explicitly configured for threshold-based filtering to work; L2 has no fixed range.
- `python -m pip` / `python -m module` avoids Windows venv PATH mismatches — confirmed the hard way when bare `pip install` silently installed into the global Python instead of the active venv, causing a `ModuleNotFoundError` that looked like a code bug but was an environment issue.
- Generators (`yield`) don't run until iterated — the same generator function can power both a "wait for everything" flow and a "forward pieces immediately" flow.
- A bi-encoder (embedding model) and a cross-encoder (re-ranker) solve different problems: bi-encoders embed question and chunk _separately_ for fast approximate search across a whole collection; cross-encoders score them _jointly_ for precision, but are too slow to run against everything — hence retrieve-wide-then-rerank-narrow as the standard pattern.
- `asyncpg` connection pools are bound to the event loop that created them — reusing a pool across multiple separate `asyncio.run()` calls (each with its own temporary event loop) causes `InterfaceError`/`Event loop is closed`, since the pool's underlying connections are tied to an event loop that no longer exists by the second call.
- Postgres full-text search defaults to AND logic between query words (`plainto_tsquery`) — a question and its answer can both be about the same topic yet share zero common significant words, causing zero search results even though the right content exists. OR-based matching (`to_tsquery` with `|`) fixes this at the cost of needing the re-ranker to filter out the resulting looser matches.
- An eval harness doesn't need per-document coverage — a small (~20-40 question), growing, representative sample is enough to catch systemic retrieval/chunking problems, and it should grow from real observed failures over time rather than being written exhaustively upfront.
- A cross-encoder can score a chunk very negatively even when it contains the literal correct answer, if the chunk also contains a lot of unrelated content (e.g. two merged subsections) — this is a chunking problem disguised as a retrieval problem, only diagnosable by inspecting actual chunk text and actual scores rather than guessing from pass/fail alone.
- A failing eval case can be a test-authoring problem (the expected phrase doesn't match the paper's wording) rather than a retrieval bug — check the retrieved text before changing code or the test.
- `import` executes a module's top-level code, so import order matters when modules create network clients at import time (this is why environment validation runs before the other `app.*` imports).
- `load_dotenv()` doesn't override variables already set in the shell, and uvicorn's auto-reload doesn't watch `.env` — after editing `.env`, restart the server.
- Retrying a streaming call is only safe if nothing has been yielded yet; otherwise the retry duplicates text the user already received.
- Dev and prod stacks need separate data stores: embedding dimensions differ, and delete-by-filename in a shared database lets one environment erase the other's data.

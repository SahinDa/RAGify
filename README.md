# RAGify

**Live:** https://ragify-zb6i.onrender.com (hosted on Render)

A Retrieval-Augmented Generation (RAG) system built from scratch to learn how RAG pipelines work end-to-end: file upload → embedding → hybrid retrieval → re-ranking → LLM answer generation, restricted strictly to the uploaded context, with conversation memory for follow-up questions.

The same codebase runs in two modes, selected by one environment variable:

- **dev**: local `sentence-transformers` models + **NVIDIA NIM** free-tier LLMs
- **prod**: **Gemini API** for embeddings, re-ranking, and answers (no local ML models)

---

## Goal

- Upload a document → chunk it (structure-aware) → embed it → store it in a vector database and a full-text search database
- Ask a question → rewrite follow-ups using conversation memory → retrieve via hybrid search (semantic + keyword) → re-rank → send to an LLM
- The LLM answers **only** from the retrieved context, and says so when the documents don't contain the answer

---

## Architecture

### Overview

```
Browser (static/index.html)
   │
   ▼
FastAPI app  (Render · Docker · 1 worker · per-IP rate limits · logging)
   │
   ├─ /upload ─▶ parse ─▶ section-aware chunking ─▶ embed ─┬─▶ Chroma Cloud    (vectors)
   │                                                       └─▶ Neon Postgres   (full-text)
   │
   └─ /ask ────▶ memory ─▶ query rewrite ─▶ hybrid retrieval ─▶ re-rank ─▶ LLM ─▶ answer
                                              ├─ Chroma Cloud   (semantic)
                                              └─ Neon Postgres  (keyword)

Embedder, re-ranker, and LLM depend on ENVIRONMENT:
   dev  → local MiniLM + cross-encoder, NVIDIA NIM
   prod → Gemini API
```

### Components

```
RAGify/
├── app/
│   ├── main.py             # FastAPI entrypoint: routes, rate limits, /health
│   ├── ingestion.py        # parsing, section detection, chunking, embedding, storage (Chroma + Postgres)
│   ├── retrieval.py        # hybrid retrieval, Reciprocal Rank Fusion, re-ranking
│   ├── postgres_search.py  # full-text indexing + OR-based keyword search (asyncpg)
│   ├── services.py         # environment-selected embedder and re-ranker
│   ├── llm.py              # prompt building, answer streaming (NVIDIA or Gemini), query rewriting
│   ├── memory.py           # in-process conversation memory
│   ├── summarizer.py       # background rolling-summary generation
│   ├── env_check.py        # startup validation of required environment variables
│   └── logging_config.py   # one-time logging setup
├── static/
│   └── index.html          # browser frontend (upload, ask, streaming ask)
├── eval/
│   ├── eval_set.json       # regression questions (Attention Is All You Need paper)
│   └── run_eval.py         # runs retrieval against the eval set, reports accuracy
├── tests/                  # manual verification scripts (embedding, LLM, Postgres, RRF, pipeline stages)
├── Dockerfile              # production image
├── .env                    # secrets and config (never commit)
└── requirements*.txt
```

### Ingestion flow

`upload → parse (.txt/.pdf/.docx) → detect numbered section headers → split into sections → chunk each section (200 words, 50-word overlap, sentence-aware) → embed → store in Chroma (vectors) and Postgres (full-text)`

- Chunks never span two detected sections (e.g. "5.3 Optimizer"). Text with no detected headers falls back to word-count chunking.
- Chunk IDs are `filename#chunk_<index>_<hash10>`. Re-uploading a file with the same name first deletes its old chunks in both stores.

### Query flow

```
question + conversation_id
  → load memory (rolling summary + recent turns)
  → rewrite a follow-up into a standalone search query (falls back to the original on failure)
  → vector search   (Chroma, top 20, cosine similarity >= 0.3)  ─┐
  → keyword search  (Postgres, top 20, OR-matched terms)         ─┴→ Reciprocal Rank Fusion (k=60)
  → re-rank the fused candidates → top 3
  → build prompt (rules + summary + recent turns + chunks + question)
  → LLM answer (context only)
  → save the turn → summarize older turns in the background
```

The rewritten query drives retrieval; the user's original wording goes into the final prompt. `/ask/stream` follows the same flow and streams the answer as it is generated.

### Conversation memory

Held in process memory only: up to 500 conversations, expiring after 1 hour of inactivity. The last 4 turns are kept verbatim. Older turns are condensed into a rolling summary by a background task (`summarizer.py`), committed with a version check so a slow summarization can't overwrite a newer one.

### Technology by environment

| Component                       | Dev                                                                       | Prod                                                                                                 | Why                                                       |
| ------------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| Backend                         | FastAPI (fully async)                                                     | same                                                                                                 | Async, fast, easy to build APIs                           |
| Vector DB                       | Chroma Cloud, cosine similarity                                           | same                                                                                                 | Managed, zero local infra                                 |
| Keyword search                  | PostgreSQL (Neon free tier), full-text `tsvector`                         | same                                                                                                 | Exact-term matches that vector search misses              |
| Embeddings                      | `all-MiniLM-L6-v2`, local, normalized (384-d)                             | Gemini `gemini-embedding-001`                                                                        | Dev: free, no rate limits. Prod: no local model in memory |
| Re-ranker                       | Cross-encoder `ms-marco-MiniLM-L-6-v2`, local                             | Gemini `gemini-3.5-flash-lite`, listwise (LLM returns ranked indices)                                | More precise than cosine similarity alone                 |
| Answer LLM                      | NVIDIA NIM `meta/llama-3.2-11b-vision-instruct`                           | Gemini `gemini-3.5-flash-lite`                                                                       | Free tiers, same interface in both modes                  |
| Query rewriting + summarization | NVIDIA NIM                                                                | Gemini                                                                                               | Same models as the answer LLM                             |
| Conversation memory             | In-process                                                                | same                                                                                                 | No extra infrastructure                                   |
| Rate limiting                   | `slowapi`, in-memory, per client IP                                       | same                                                                                                 | Protects quotas and compute                               |
| File parsing                    | `pypdf`, `python-docx`, built-in decode                                   | same                                                                                                 | Lightweight, enough for text-based documents              |
| Frontend                        | Plain HTML/JS via FastAPI StaticFiles                                     | same                                                                                                 | Keeps the focus on the RAG pipeline                       |
| LLM retry/backoff               | Yes (3 attempts: 1s, 2s waits)                                            | No                                                                                                   |                                                           |
| Dependencies                    | `requirements-dev.txt`: full `chromadb`, `sentence-transformers`, PyTorch | `requirements.txt` (installed by the Dockerfile): thin `chromadb-client`, `google-genai`, no PyTorch | Local-model imports are lazy, so prod never loads PyTorch |
| Extra env vars                  | `NVIDIA_API_KEY`, `NVIDIA_URL`                                            | `GEMINI_API_KEY`                                                                                     |                                                           |

`ENVIRONMENT=prod` switches to the prod column; any other value (or unset) runs dev. Model names are taken from the code; confirm them against your providers' current model lists when you change them.

**Keep the two environments' data separate.** The embedding models have different dimensions, so dev and prod cannot share a Chroma collection: set a different `CHROMA_COLLECTION_NAME` for each. Use a separate Postgres database (or Neon branch) per environment too, because the keyword index is keyed by filename and re-uploading a file deletes that filename's rows.

---

## API

| Method | Path                                | Parameters                    | Rate limit (per IP) | Description                                                            |
| ------ | ----------------------------------- | ----------------------------- | ------------------- | ---------------------------------------------------------------------- |
| `GET`  | `/`                                 |                               | none                | Serves the frontend                                                    |
| `POST` | `/upload`                           | multipart `file`              | 5/min               | Ingest a document (max 10 MB)                                          |
| `POST` | `/ask`                              | `question`, `conversation_id` | 15/min              | Full answer as JSON: `{question, answer, sources}`                     |
| `POST` | `/ask/stream`                       | `question`, `conversation_id` | 15/min              | Answer streamed as `text/plain`                                        |
| `GET`  | `/memory/{conversation_id}/summary` |                               | 30/min              | Current rolling summary + number of recent turns held (max 4)          |
| `GET`  | `/health`                           |                               | none                | `{"status": "healthy", "message": "RAGify service is up and running"}` |

`conversation_id` is required. Any unguessable string works; the client creates one and reuses it for the whole chat.

```bash
curl -X POST -F "file=@paper.pdf" http://127.0.0.1:8000/upload

curl -X POST "http://127.0.0.1:8000/ask?question=What%20optimizer%20was%20used%3F&conversation_id=my-chat-1"

curl -N -X POST "http://127.0.0.1:8000/ask/stream?question=Why%20that%20one%3F&conversation_id=my-chat-1"
```

**Errors:** `400` empty question / unsupported or empty file, `413` file too large, `429` rate limit exceeded, `500` retrieval or unexpected failure, `502` provider HTTP error, `503` provider unreachable, `504` provider timeout. The 502/503/504 mappings come from `httpx` exceptions (the NVIDIA path); Gemini SDK errors currently surface as a generic `500`. On `/ask/stream` the status code is already sent when streaming starts, so errors appear as an `[Error: ...]` line in the stream.

### Rate limiting

Per-IP limits use `slowapi` with in-memory counters. Behind Render's reverse proxy the socket address is the proxy's, so the client IP is read from the `X-Forwarded-For` header. Exceeding a limit returns `429`.

- `/upload`, 5/min: every upload runs parsing, chunking, embedding, and dual-store ingestion.
- `/ask` and `/ask/stream`, 15/min: stops automated request loops and protects LLM and embedding quotas.
- `/memory/{conversation_id}/summary`, 30/min.
- `/health` is not limited, so the uptime cron job is never throttled.

Limits apply to the deployed service; local development runs are not throttled. Counters live in process memory, so they reset on restart and are per worker, which is fine for the single-worker deployment.

### Health check

`GET /health` touches no database or model, so it is cheap to call. A cron job calls it every 10 minutes so Render's free plan doesn't spin the service down after 15 minutes without traffic (see Deployment).

---

## Project Phases

### Phase 1: Core Concepts ✅

- [x] Embeddings, chunking, vector databases

### Phase 2: Environment Setup ✅

- [x] Python venv, NVIDIA NIM API key, dependencies, `.env`, folder structure

### Phase 3: Ingestion Pipeline ✅

- [x] File upload endpoint, parsing (.txt/.pdf/.docx), chunking, embedding, storage

### Phase 4: Query Pipeline ✅

- [x] `/ask` endpoint, embed question, retrieve, build prompt, call LLM

### Phase 5: Guardrails ✅

- [x] Context-only system prompt, similarity threshold fallback, input validation

### Phase 6: Retrieval Quality ✅

- [x] .docx support, dedup, minimal frontend, streaming responses
- [x] Async `httpx` streaming + `run_in_threadpool` for all blocking calls
- [x] Migrated to Chroma Cloud
- [x] Normalized embeddings on both ingestion and query sides (fixed a cosine-distance mismatch)
- [x] File-scoped chunk IDs (`filename#chunk_i_hash`)
- [x] Re-ranking: wide vector retrieval (top 20) → re-rank → top 3
- [x] Eval harness: 12-question regression suite against a real document
- [x] Hybrid search: Postgres full-text alongside vector search, merged with Reciprocal Rank Fusion
- [x] Structure-aware chunking (numbered section headers)

### Phase 7: Conversations & Reliability ✅

- [x] Multi-turn chat: conversation memory, follow-up query rewriting, rolling summary
- [x] Structured logging across upload, retrieval, and answer paths
- [x] Startup environment validation with a clear error naming any missing variable
- [x] Retry with exponential backoff on the NVIDIA call (only before any output is sent)
- [x] Error handling around retrieval and LLM calls with correct HTTP status codes
- [x] Per-IP rate limiting (`slowapi`, proxy-aware) and a `/health` endpoint

### Phase 8: Dev/Prod & Deployment ✅

- [x] Dev/prod split behind one interface (`app/services.py`, `ENVIRONMENT` switch)
- [x] Gemini embeddings, listwise re-ranking, answers, query rewriting, and summarization for prod
- [x] Dockerized and deployed on Render, with a cron-based keep-alive

---

## Setup

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

# 4. Add config to .env  (KEY=value, no spaces around "=", no quotes)
ENVIRONMENT=dev
NVIDIA_API_KEY=your_key_here
NVIDIA_URL=https://integrate.api.nvidia.com/v1/chat/completions
CHROMA_API_KEY=your_key_here
CHROMA_TENANT=your_tenant_here
CHROMA_DATABASE=your_database_here
CHROMA_COLLECTION_NAME=ragify_docs_dev
DATABASE_URL=your_neon_connection_string_here

# 5. Create the Postgres chunks table (once per database, in Neon's SQL Editor)
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
```

### Environment variables

| Variable                 | Dev              | Prod     | Notes                                                           |
| ------------------------ | ---------------- | -------- | --------------------------------------------------------------- |
| `ENVIRONMENT`            | `dev` (or unset) | `prod`   | Exact value `prod` switches modes                               |
| `CHROMA_API_KEY`         | required         | required | Chroma Cloud                                                    |
| `CHROMA_TENANT`          | required         | required |                                                                 |
| `CHROMA_DATABASE`        | required         | required |                                                                 |
| `CHROMA_COLLECTION_NAME` | optional         | optional | Defaults to `ragify_docs`; use a different name per environment |
| `DATABASE_URL`           | required         | required | Neon Postgres; separate database per environment                |
| `NVIDIA_API_KEY`         | required         |          |                                                                 |
| `NVIDIA_URL`             | required         |          | `https://integrate.api.nvidia.com/v1/chat/completions`          |
| `GEMINI_API_KEY`         |                  | required |                                                                 |

On startup, `validate_env()` checks the variables required for the current mode and names any that are missing.

---

## Deployment

- **Platform:** Render (Docker web service)
- **Live URL:** https://ragify-zb6i.onrender.com
- **Image:** built from the `Dockerfile`: `python:3.12-slim`, installs `requirements.txt` (no PyTorch), runs as a non-root user
- **Start command:** set in the Dockerfile: `uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1`
- **Environment variables:** set in the Render dashboard (the prod column above, with `ENVIRONMENT=prod`)
- **Workers:** exactly one. Conversation memory and rate-limit counters live in process memory, so with multiple workers a follow-up can land on a process that has never seen the conversation. One worker also keeps memory use inside small plans' limits.
- **Keep-alive:** Render's free plan spins a service down after 15 minutes without requests. A cron job calls `GET /health` every 10 minutes, so the service stays awake. A restart or redeploy still clears in-memory conversation history.

Run the production image locally:

```bash
docker build -t ragify .
docker run --env-file <your-prod-env-file> -e ENVIRONMENT=prod -p 8000:8000 ragify
```

---

## Evaluation

`eval/eval_set.json` holds 12 questions against the _Attention Is All You Need_ paper: direct facts, exact-term questions (numbers, names), and one unanswerable question that checks the fallback. `run_eval.py` runs retrieval only (no LLM call). A question counts as a hit when the right source file is returned **and** the expected keyword appears in the retrieved text.

```bash
python -m eval.run_eval
```

| Change                                    | Score (dev stack) |
| ----------------------------------------- | ----------------- |
| Hybrid search + re-ranking                | 9/12 (75%)        |
| + structure-aware chunking                | 11/12 (91.7%)     |
| + corrected keyword in one test assertion | 12/12 (100%)      |

It is a regression suite, not full coverage, and it was measured on the dev stack. Re-run it against the prod configuration before trusting prod retrieval quality.

---

## Key Decisions

- **Hybrid retrieval over vector-only.** Embeddings capture meaning but miss exact terms (numbers, names). Postgres full-text covers those, and Reciprocal Rank Fusion merges the two ranked lists without needing comparable scores.
- **Postgres full-text over Elasticsearch and `rank_bm25`.** Elasticsearch adds a service, a recurring cost, and dual-write complexity. `rank_bm25` can't update incrementally, so every upload would rebuild the whole index. Neon's free tier plus `tsvector` gives incremental indexing for far less effort.
- **OR-based keyword matching (`to_tsquery` with `|`), not `plainto_tsquery`.** AND-matching returned nothing whenever a question's wording (e.g. "Transformer") didn't appear in the same chunk as the answer (e.g. "Adam optimizer"). The re-ranker cleans up the looser matches. Found through the eval harness.
- **Retrieve wide, then re-rank narrow.** Cosine search is fast but imprecise; a re-ranker scores (question, chunk) pairs jointly. Retrieve 20, keep 3.
- **Structure-aware chunking.** A chunk that mixed "5.2 Hardware and Schedule" with "5.3 Optimizer" scored -5.14 on the cross-encoder despite containing the correct answer. Splitting at detected section headers fixed it (75% → 91.7% on the eval).
- **One `asyncpg` connection per call, not a pool.** `retrieve_relevant_chunks` and `store_chunks` are sync functions run via `run_in_threadpool`, each calling `asyncio.run()`. A pool created in one call's temporary event loop is invalid in the next, causing `InterfaceError` / `Event loop is closed`.
- **Blocking work runs in `run_in_threadpool`** (embedding, re-ranking, Chroma and Postgres calls) so a slow request can't freeze the event loop for everyone else.
- **Normalized embeddings on both ingestion and query sides**, with Chroma set to cosine distance. Normalizing only one side silently distorted similarity scores.
- **File-scoped chunk IDs** (`filename#chunk_i_hash`) instead of pure content hashes. Identical text in two files used to overwrite each other's metadata. Trade-off: no cross-file deduplication.
- **Dev/prod split behind one interface** (`app/services.py`). The embedder and re-ranker are chosen once at import, and local-model imports are lazy, so prod never loads PyTorch.
- **Startup validation runs before any module that creates network clients.** `ingestion.py` connects to Chroma at import time, so `main.py` calls `load_dotenv()` and `validate_env()` first and fails with a clear message instead of a library crash.
- **Retry only before any output has been sent.** A mid-stream retry would re-yield text the user already received. Only timeouts and connection errors are retried; 4xx/5xx responses (e.g. a bad API key) are not.
- **Conversation memory is in-process, with a rolling summary.** Chosen over a database-backed store to avoid extra infrastructure; the cost is that memory is lost on restart and isn't shared across workers.
- **Per-IP rate limiting with in-memory counters.** For a public demo the things worth protecting are upstream quotas and compute, and per-IP throttling does that with no extra infrastructure. It is only correct with a single worker, which the deployment already requires. The client IP comes from `X-Forwarded-For` because behind a reverse proxy the socket address is the proxy's.
- **A cheap, unthrottled `/health` endpoint** that touches no database or model, used by the keep-alive cron job.
- **Models:** rejected `meta/llama-3.2-90b-vision-instruct` for cost; the dev default moved from `meta/llama-3.1-8b-instruct` to `meta/llama-3.2-11b-vision-instruct` after reliability issues.

---

## Architecture Decision Record: Authentication & Multi-Tenancy

**Status:** Accepted for this release.
**Decision:** No user accounts and no per-user data namespaces. Abuse and cost are mitigated with per-IP rate limits instead.

**Why:**

- **Frictionless evaluation.** Recruiters, hiring managers, and engineers can try live document Q&A immediately, with no sign-up, password, or OAuth step.
- **No per-session namespaces to orphan.** Giving each anonymous visitor a private namespace would leave abandoned browser tabs' embeddings and SQL rows stranded in Chroma and Neon until a background TTL reaper removed them. One shared corpus avoids building that machinery.
- **Focus on the core competency.** The project's goal is RAG pipeline engineering (hybrid retrieval, RRF, re-ranking, contextual memory), not standard account CRUD.
- **Quota protection by throttling.** Upstream quotas and compute are guarded by per-IP rate limits rather than account identity.

**Consequences accepted:**

- All visitors share one corpus. Anything uploaded is searchable by everyone, so do not upload private documents to the live instance.
- Uploads accumulate in the free-tier stores until someone purges them; a shared corpus does not clean itself up.
- Re-uploading an existing filename replaces that file's chunks for everyone.
- Rate limits throttle, but they cannot stop a determined actor using many IPs, and the Gemini quota is shared globally.

**Revisit when:** the app needs private documents, multiple users, or traffic beyond demos. That would mean authenticated users, an owner field in the Chroma metadata and the Postgres `chunks` table, owner-scoped queries, and a cleanup policy.

---

## Known Limitations

### Main limitations

- **Keyword search is tuned for English.** The Postgres full-text index uses the `english` configuration (stemming, stopwords), and the query builder's stopword list is English. Documents in other languages still get vector search, but keyword matching is weaker, and languages written without spaces (such as Chinese or Japanese) are not supported by this tokenization. The dev embedding model (MiniLM) is English-oriented as well. Supporting more languages would mean per-language text-search configs or language detection at ingestion.
- **No authentication and no per-user data isolation**, by design (see the ADR above). All visitors share one corpus.

### Other limitations

- **Same filename = overwrite** in both Chroma and Postgres, with no collision protection.
- **Section detection is narrow.** Only `N.N Title` headers are detected. Single-number headers ("1 Introduction"), Markdown headers, and unheaded text (abstracts, plain notes) fall back to word-count chunking, which can still mix topics inside a chunk.
- **Conversation memory is in-process:** lost on restart, not shared across workers, expires after an hour. If more than 20 turns pile up unsummarized (for example, if the summarizer keeps failing), `add_turn` raises.
- **Rate limits are a throttle, not a security boundary:** counters are in-process and keyed on the proxy-provided client IP.
- **Prod re-ranker sees only the first 400 characters of each chunk**, so an answer late in a long chunk can be ranked down.
- **Prod embeddings are requested one chunk per API call**, so large uploads are slow and quota-hungry.
- **The 0.3 similarity threshold was tuned on MiniLM.** Gemini embeddings have a different similarity distribution, so re-check it in prod.
- **Retry/backoff exists only on the NVIDIA (dev) path;** Gemini errors surface as a generic `500`.
- **Text-based files only:** scanned PDFs are not OCR'd; max file size is 10 MB.
- **Small eval set:** 12 questions, one document, measured on the dev stack.

---

## Learnings Log

- Embeddings capture meaning, not exact words: cosine similarity for "cat/mat" vs "feline/rug" was 0.556, vs "stock market" 0.097.
- Chunk overlap matters: 200-word chunks with 50-word overlap on 1000 words produced 7 chunks, matching manual math.
- Distance metrics matter: cosine distance must be configured explicitly for threshold filtering to work; L2 has no fixed range.
- `python -m pip` / `python -m module` avoids Windows venv PATH mismatches. Bare `pip install` once installed into the global Python instead of the venv, producing a `ModuleNotFoundError` that looked like a code bug.
- Generators (`yield`) don't run until iterated, so one generator function can power both a "wait for everything" flow and a "forward pieces immediately" flow.
- A bi-encoder (embedding model) and a cross-encoder (re-ranker) solve different problems: bi-encoders embed question and chunk _separately_ for fast approximate search; cross-encoders score them _jointly_ for precision but are too slow to run against everything. Hence retrieve wide, re-rank narrow.
- `asyncpg` pools are bound to the event loop that created them; reusing one across separate `asyncio.run()` calls breaks.
- Postgres full-text defaults to AND between query words. A question and its answer can share no significant words, so AND-matching can return nothing even when the right content exists.
- An eval harness doesn't need per-document coverage. A small, growing, representative sample catches systemic retrieval and chunking problems, and it should grow from real failures.
- A cross-encoder can score a chunk very negatively even when it contains the literal answer, if the chunk also holds unrelated content. That is a chunking problem disguised as a retrieval problem, visible only by inspecting real chunk text and real scores.
- A failing eval case can be a test-authoring problem (the expected phrase doesn't match the paper's wording) rather than a retrieval bug. Check the retrieved text before changing code or the test.
- `import` executes a module's top-level code, so import order matters when modules create network clients at import time.
- `load_dotenv()` doesn't override variables already set in the shell, and uvicorn's auto-reload doesn't watch `.env`. Restart the server after editing it.
- Retrying a streaming call is only safe if nothing has been yielded yet; otherwise the retry duplicates text the user already received.
- Dev and prod stacks need separate data stores: embedding dimensions differ, and delete-by-filename in a shared database lets one environment erase the other's data.
- Behind a reverse proxy the socket address is the proxy's, so per-IP rate limiting has to read the forwarded-for header; in-memory limit counters are only correct with a single worker.

# RAGify

**Live:** https://ragify-zb6i.onrender.com (hosted on Render)

A Retrieval-Augmented Generation (RAG) system built from scratch to learn how RAG pipelines work end-to-end: file upload → embedding → hybrid retrieval → re-ranking → LLM answer generation, restricted strictly to the uploaded context, with conversation memory for follow-up questions and **tenant-isolated document storage**.

The app runs in a **hybrid public-demo + private multi-tenant** model:

- **Guests (not logged in)** use a shared public demo corpus, so anyone can try it with no sign-up.
- **Logged-in users** get a strictly isolated private corpus that only they can upload to, search, and ask questions about.

The same codebase runs in two modes, selected by one environment variable (`ENVIRONMENT`):

- **dev: for people who want to run and build on it themselves.** Clone the repo, download the open-source models onto your own machine (`all-MiniLM-L6-v2` for embeddings and a `ms-marco` cross-encoder for re-ranking), and experiment, test, and extend the pipeline locally. It needs PyTorch and `sentence-transformers`, so it is heavier, but embedding and re-ranking are free with no rate limits.
- **prod: the live, hosted version that visitors actually use.** Embeddings and re-ranking go through the **Gemini API**, so the server needs no local ML models and no PyTorch. This is what runs on Render at the URL above, and it is what anyone testing the live demo is using.

Both modes use the **Gemini API for answers, query rewriting, and summaries**. The switch only changes where embeddings and re-ranking run.

---

## Goal

- Upload a document → hash it (skip exact duplicates) → chunk it (structure-aware) → embed it → store it in a vector database and a full-text search database, tagged with its owner
- Ask a question → rewrite follow-ups using conversation memory → retrieve via **owner-scoped** hybrid search (semantic + keyword) → re-rank → send to an LLM
- The LLM answers **only** from the retrieved context, and says so when the documents don't contain the answer
- One user's private documents are never retrievable by another user or by guests

---

## Architecture

### Overview

```
Browser (static/index.html)
   │
   ▼
FastAPI app  (Render · Docker · 1 worker · per-IP rate limits · logging)
   │
   ├─ /auth/* ──▶ signup · login · logout · me   (bcrypt + JWT in cookie / Bearer header)
   │
   │  every /upload, /ask, /ask/stream call resolves ownership first:
   │     guest       → (is_public=True,  user_id=None)   → shared demo corpus
   │     logged in   → (is_public=False, user_id=<id>)   → private corpus
   │
   ├─ /upload ─▶ size check ─▶ SHA-256 duplicate check ─▶ parse ─▶ section-aware chunking ─▶ embed
   │                                                                ├─▶ Neon Postgres  (documents + chunks, full-text)
   │                                                                └─▶ Chroma Cloud   (vectors + owner metadata)
   │
   └─ /ask ────▶ memory ─▶ query rewrite ─▶ scoped hybrid retrieval ─▶ re-rank ─▶ LLM ─▶ answer
                                              ├─ Chroma Cloud   (semantic, owner filter)
                                              └─ Neon Postgres  (keyword, owner filter)

Embedder and re-ranker depend on ENVIRONMENT; the answer LLM is Gemini in both:
   dev  → local MiniLM + cross-encoder      · Gemini (answers, rewriting, summaries)
   prod → Gemini embeddings + Gemini rerank · Gemini (answers, rewriting, summaries)
```

### Components

```
RAGify/
├── app/
│   ├── __init__.py
│   ├── main.py             # FastAPI entrypoint: lifespan, routes, rate limits, /health
│   ├── ingestion.py        # parsing, file hashing, section detection, chunking, embedding, storage (Chroma + Postgres)
│   ├── retrieval.py        # scoped hybrid retrieval, Reciprocal Rank Fusion, re-ranking (native async)
│   ├── postgres_search.py  # duplicate check, atomic document+chunk indexing, scoped keyword search (asyncpg)
│   ├── services.py         # environment-selected embedder and re-ranker (lazy local-model imports)
│   ├── llm.py              # prompt building, Gemini async streaming (google-genai), query rewriting
│   ├── memory.py           # in-process conversation memory
│   ├── summarizer.py       # background rolling-summary generation
│   ├── env_check.py        # startup validation of required environment variables
│   ├── logging_config.py   # one-time logging setup
│   ├── auth/
│   │   ├── __init__.py     # public exports (auth_router, get_optional_user, resolve_ownership, ...)
│   │   ├── router.py       # /auth/signup, /auth/login, /auth/logout, /auth/me
│   │   ├── deps.py         # get_optional_user, get_current_user, DemoUser, resolve_ownership, require_admin
│   │   ├── schemas.py      # Pydantic request/response models
│   │   └── security.py     # bcrypt hashing, JWT creation/validation
│   └── db/
│       ├── database.py     # SQLAlchemy engine/session (auth) + shared asyncpg pool (search)
│       ├── models.py       # SQLAlchemy User model, UserRole enum
│       └── schema.sql      # Postgres schema: users, documents, chunks
├── static/
│   └── index.html          # browser frontend (upload, ask, streaming ask)
├── eval/
│   ├── eval_set.json       # regression questions (Attention Is All You Need paper)
│   └── run_eval.py         # runs retrieval against the eval set, reports accuracy
├── tests/                  # manual verification scripts (embedding, LLM, Postgres, RRF); some predate the Gemini/multi-tenant changes
├── Dockerfile              # production image
├── .env                    # secrets and config (never commit)
└── requirements*.txt
```

### Data model (PostgreSQL / Neon)

```
users (1) ───────< documents (1) ───────< chunks
                    parent record          child records + tsvector
```

**`users`** stores registered accounts: `id`, `email` (unique), `first_name`, `last_name`, `hashed_password` (bcrypt), `role` (1 = user, 2 = admin), `created_at`.

**`documents`** is the file registry (parent table):

| Column        | Notes                                                                        |
| ------------- | ---------------------------------------------------------------------------- |
| `id`          | UUID stored as text, primary key                                             |
| `filename`    |                                                                              |
| `file_hash`   | SHA-256 hex digest of the raw uploaded bytes                                 |
| `is_public`   | `TRUE` for the shared demo corpus                                            |
| `user_id`     | Nullable FK to `users.id` (`ON DELETE CASCADE`); `NULL` for public documents |
| `chunk_count` |                                                                              |
| `created_at`  |                                                                              |

**Ownership `CHECK` constraint** (on both `documents` and `chunks`):

```sql
CHECK ((is_public = TRUE AND user_id IS NULL) OR (is_public = FALSE AND user_id IS NOT NULL))
```

A row is either public with no owner, or private with an owner. Nothing in between can be stored.

**Scoped deduplication indexes** block duplicate uploads per workspace without leaking across tenants (full SQL in the Setup section's Database schema):

```sql
CREATE UNIQUE INDEX uq_documents_public_hash ON documents (file_hash)          WHERE is_public = TRUE;
CREATE UNIQUE INDEX uq_documents_user_hash   ON documents (user_id, file_hash) WHERE is_public = FALSE;
```

**`chunks`** is the child table used for full-text search:

- `id` is `{document_id}:{chunk_index}` (previously `filename#chunk_i_hash`)
- `document_id` FK to `documents.id` with `ON DELETE CASCADE`, plus `chunk_index`, `is_public`, `user_id`
- `text`, `source` (filename), and `search_vector` (`TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED`, GIN-indexed)

**Cascading cleanup:** deleting or replacing a `documents` row automatically deletes its `chunks` rows in Postgres. (Chroma vectors are purged explicitly by the application; see Known Limitations.)

The Chroma side stores each chunk under the same `{document_id}:{chunk_index}` ID, with metadata `document_id`, `source`, `chunk_index`, `char_length`, `is_public`, and `user_id` (empty string for public documents).

### Authentication & ownership

Auth lives in `app/auth/` and is mounted through `auth_router`.

- **Passwords:** hashed with `bcrypt`. **Sessions:** signed JWT (`HS256`, 24h default) sent as the `ragify_token` cookie or an `Authorization: Bearer` header.
- **`get_optional_user`** resolves the caller to a `User` if the token is valid, otherwise to a virtual `DemoUser` (guest). Invalid, expired, or unknown-user tokens fall back to guest rather than erroring.
- **`resolve_ownership(user)`** is the single choke point that maps a caller to a storage scope:
  - `DemoUser` → `(is_public=True, user_id=None)`
  - `User` → `(is_public=False, user_id=str(user.id))`
- **`get_current_user`** is the strict variant (401 for guests), used by `/auth/logout`. **`require_admin`** gates admin-only operations (role 2).

`/upload`, `/ask`, and `/ask/stream` all call `resolve_ownership` and pass the result down into both search stores.

**Tenant filters applied to both stores:**

| Store      | Public scope             | Private scope                                            |
| ---------- | ------------------------ | -------------------------------------------------------- |
| ChromaDB   | `{"is_public": True}`    | `{"$and": [{"is_public": False}, {"user_id": user_id}]}` |
| PostgreSQL | `WHERE is_public = TRUE` | `WHERE is_public = FALSE AND user_id = $2`               |

### Ingestion flow

```
upload
  → size validation (non-empty, ≤ 10 MB)
  → SHA-256 of raw bytes → duplicate check in the caller's scope  ──(hit)──▶ return status "duplicate", stop
  → parse (.txt/.pdf/.docx)
  → detect numbered section headers → split into sections
  → chunk each section (200 words, 50-word overlap, sentence-aware)
  → embed
  → Postgres (one transaction): delete caller's previous version of the same filename,
                                insert documents row, bulk-insert chunks
  → Chroma: purge superseded document vectors, upsert new chunks with ownership metadata
```

- **Duplicate short-circuit.** Identical bytes already stored in the caller's scope return `"status": "duplicate"` immediately: no PDF/DOCX parsing, no embedding API calls, no database writes.
- **Scoped overwrite.** Uploading a modified file under an existing filename replaces only the caller's previous version. Other users' files with the same name are untouched.
- Chunks never span two detected sections (e.g. "5.3 Optimizer"). Text with no detected headers falls back to word-count chunking.
- PDF extraction is null-safe: image-only pages return `None` from `pypdf`, which is coerced to an empty string.

### Query flow

```
question + conversation_id (+ optional auth token)
  → resolve ownership (guest → public corpus, user → private corpus)
  → load memory (rolling summary + recent turns)
  → rewrite a follow-up into a standalone search query (falls back to the original on failure)
  → vector search   (Chroma, owner-filtered, top 20, cosine similarity >= 0.3)  ─┐
  → keyword search  (Postgres, owner-filtered, top 20, OR-matched terms)          ─┴→ Reciprocal Rank Fusion (k=60)
  → re-rank the fused candidates → top 3
  → build prompt (rules + summary + recent turns + chunks + question)
  → LLM answer (context only)
  → save the turn → summarize older turns in the background
```

The rewritten query drives retrieval; the user's original wording goes into the final prompt. `/ask/stream` follows the same flow and streams the answer as it is generated.

### Async model

The request path runs natively on FastAPI's main event loop:

- **Async I/O stays on the event loop with `await`:** `asyncpg` queries (shared pool), the `google-genai` async client (`client.aio`), `retrieve_relevant_chunks`, and `store_chunks`.
- **Blocking synchronous work is offloaded with `run_in_threadpool`:** `parse_file`, `embed_chunks`, ChromaDB queries/upserts/deletes, and local re-ranking.
- **Never wrap an `async def` in `run_in_threadpool`.** It would return an unawaited coroutine instead of running it.
- A single `asyncpg` pool (`get_db_pool()`, created lazily) is shared across requests and closed on shutdown through FastAPI's `lifespan` hook (`close_db_pool()`).

### Conversation memory

Held in process memory only: up to 500 conversations, expiring after 1 hour of inactivity. The last 4 turns are kept verbatim. Older turns are condensed into a rolling summary by a background task (`summarizer.py`), committed with a version check so a slow summarization can't overwrite a newer one.

### Technology by environment

| Component                       | Dev                                                                                         | Prod                                                                                     | Why                                                       |
| ------------------------------- | ------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| Backend                         | FastAPI (async)                                                                             | same                                                                                     | Async, fast, easy to build APIs                           |
| Auth                            | `bcrypt` + `PyJWT`, SQLAlchemy sessions                                                     | same                                                                                     | Simple, self-contained accounts                           |
| Vector DB                       | Chroma Cloud, cosine similarity                                                             | same                                                                                     | Managed, zero local infra                                 |
| Keyword search + registry       | PostgreSQL (Neon free tier), `documents` + `chunks`, `tsvector`, shared `asyncpg` pool      | same                                                                                     | Exact-term matches that vector search misses              |
| Embeddings                      | `all-MiniLM-L6-v2`, local, normalized (384-d)                                               | Gemini `gemini-embedding-001`                                                            | Dev: free, no rate limits. Prod: no local model in memory |
| Re-ranker                       | Cross-encoder `ms-marco-MiniLM-L-6-v2`, local                                               | Gemini `gemini-3.5-flash-lite`, listwise (LLM returns ranked indices)                    | More precise than cosine similarity alone                 |
| Answer LLM                      | Gemini `gemini-2.5-flash-lite` (override with `GEMINI_MODEL`), temp 0.2, 512 max tokens     | Gemini `gemini-2.5-flash-lite` (override with `GEMINI_MODEL`), temp 0.1, 1024 max tokens | One SDK and one streaming path for both modes             |
| Query rewriting + summarization | Gemini (`google-genai` async client)                                                        | same                                                                                     | Same model as the answer LLM                              |
| Conversation memory             | In-process                                                                                  | same                                                                                     | No extra infrastructure                                   |
| Rate limiting                   | `slowapi`, in-memory, per client IP                                                         | same                                                                                     | Protects quotas and compute                               |
| File parsing                    | `pypdf` (null-safe), `python-docx`, built-in decode                                         | same                                                                                     | Lightweight, enough for text-based documents              |
| Frontend                        | Plain HTML/JS via FastAPI StaticFiles                                                       | same                                                                                     | Keeps the focus on the RAG pipeline                       |
| Dependencies                    | `requirements-dev.txt`: full `chromadb`, `sentence-transformers`, PyTorch, plus shared deps | `requirements.txt` (installed by the Dockerfile): thin `chromadb-client`, no PyTorch     | Local-model imports are lazy, so prod never loads PyTorch |
| Extra env vars                  | `GEMINI_API_KEY`                                                                            | `GEMINI_API_KEY`                                                                         |                                                           |

**Which mode should I use?**

| If you want to...                                                       | Use                                                                 |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------- |
| Download the models, run everything on your own machine, test and build | `dev` (`requirements-dev.txt`, `ENVIRONMENT=dev` or unset)          |
| Try the app as a user, or deploy it for others                          | `prod` (the live site, or the Docker image with `ENVIRONMENT=prod`) |

`ENVIRONMENT=prod` switches embeddings and re-ranking to the Gemini column; any other value (or unset) runs dev. Model names are taken from the code; confirm them against Google's current model list when you change them.

**Keep the two environments' data separate.** The embedding models have different dimensions, so dev and prod cannot share a Chroma collection: set a different `CHROMA_COLLECTION_NAME` for each. Use a separate Postgres database (or Neon branch) per environment too, so one environment's filename-scoped replacement never erases the other's data.

---

## API

| Method | Path                                | Parameters                                           | Auth     | Rate limit (per IP) | Description                                                            |
| ------ | ----------------------------------- | ---------------------------------------------------- | -------- | ------------------- | ---------------------------------------------------------------------- |
| `GET`  | `/`                                 |                                                      | none     | none                | Serves the frontend                                                    |
| `POST` | `/auth/signup`                      | JSON `email`, `password`, `first_name`, `last_name?` | none     | none                | Create an account; sets the session cookie, returns token + user       |
| `POST` | `/auth/login`                       | JSON `email`, `password`                             | none     | none                | Authenticate; sets the session cookie, returns token + user            |
| `POST` | `/auth/logout`                      |                                                      | required | none                | Clears the session cookie                                              |
| `GET`  | `/auth/me`                          |                                                      | optional | none                | `{authenticated, is_public, user}`; guests get the `DemoUser`          |
| `POST` | `/upload`                           | multipart `file`                                     | optional | 5/min               | Ingest a document (max 10 MB) into the caller's scope                  |
| `POST` | `/ask`                              | `question`, `conversation_id`                        | optional | 15/min              | Full answer as JSON: `{question, answer, sources}`                     |
| `POST` | `/ask/stream`                       | `question`, `conversation_id`                        | optional | 15/min              | Answer streamed as `text/plain`                                        |
| `GET`  | `/memory/{conversation_id}/summary` |                                                      | optional | 30/min              | Current rolling summary + number of recent turns held (max 4)          |
| `GET`  | `/health`                           |                                                      | none     | none                | `{"status": "healthy", "message": "RAGify service is up and running"}` |

"Optional" auth means the endpoint works for guests (public corpus) and logged-in users (private corpus). The token is read from the `ragify_token` cookie or an `Authorization: Bearer <token>` header.

`/upload` responses:

```json
{ "document_id": "…", "filename": "paper.pdf", "chunks_created": 42, "status": "success",   "owner": "you@example.com" }
{ "document_id": "…", "filename": "paper.pdf", "chunks_created": 42, "status": "duplicate", "owner": "you@example.com" }
```

`conversation_id` is required. Any unguessable string works; the client creates one and reuses it for the whole chat.

```bash
# Guest (public demo corpus)
curl -X POST -F "file=@paper.pdf" http://127.0.0.1:8000/upload

# Create an account and log in (stores the cookie)
curl -c cookies.txt -X POST http://127.0.0.1:8000/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"email":"you@example.com","password":"a-strong-password","first_name":"Ada"}'

# Logged in (private corpus)
curl -b cookies.txt -X POST -F "file=@paper.pdf" http://127.0.0.1:8000/upload
curl -b cookies.txt -X POST "http://127.0.0.1:8000/ask?question=What%20optimizer%20was%20used%3F&conversation_id=my-chat-1"
curl -b cookies.txt -N -X POST "http://127.0.0.1:8000/ask/stream?question=Why%20that%20one%3F&conversation_id=my-chat-1"
```

**Errors:** `400` empty question / unsupported or empty file / duplicate account email, `401` bad credentials or login required, `413` file too large, `422` invalid input, `429` rate limit exceeded, `500` retrieval or unexpected failure. On `/ask/stream` the status code is already sent when streaming starts, so errors appear as an `[Error: ...]` line in the stream. Gemini SDK errors currently surface as a generic `500` (the `502/503/504` mappings in `/ask` come from `httpx` exceptions and no longer fire now that the NVIDIA/`httpx` client is gone).

### Rate limiting

Per-IP limits use `slowapi` with in-memory counters. Behind Render's reverse proxy the socket address is the proxy's, so the client IP is read from the `X-Forwarded-For` header. Exceeding a limit returns `429`.

- `/upload`, 5/min: every upload runs hashing, parsing, chunking, embedding, and dual-store ingestion.
- `/ask` and `/ask/stream`, 15/min: stops automated request loops and protects LLM and embedding quotas.
- `/memory/{conversation_id}/summary`, 30/min.
- `/auth/*` and `/health` are not limited.

Limits apply to the deployed service; local development runs are not throttled. Counters live in process memory, so they reset on restart and are per worker, which is fine for the single-worker deployment.

### Health check

`GET /health` touches no database or model, so it is cheap to call. A cron job calls it every 10 minutes so Render's free plan doesn't spin the service down after 15 minutes without traffic (see Deployment).

---

## Project Phases

### Phase 1: Core Concepts ✅

- [x] Embeddings, chunking, vector databases

### Phase 2: Environment Setup ✅

- [x] Python venv, API keys, dependencies, `.env`, folder structure

### Phase 3: Ingestion Pipeline ✅

- [x] File upload endpoint, parsing (.txt/.pdf/.docx), chunking, embedding, storage

### Phase 4: Query Pipeline ✅

- [x] `/ask` endpoint, embed question, retrieve, build prompt, call LLM

### Phase 5: Guardrails ✅

- [x] Context-only system prompt, similarity threshold fallback, input validation

### Phase 6: Retrieval Quality ✅

- [x] .docx support, minimal frontend, streaming responses
- [x] Migrated to Chroma Cloud
- [x] Normalized embeddings on both ingestion and query sides (fixed a cosine-distance mismatch)
- [x] Re-ranking: wide vector retrieval (top 20) → re-rank → top 3
- [x] Eval harness: 12-question regression suite against a real document
- [x] Hybrid search: Postgres full-text alongside vector search, merged with Reciprocal Rank Fusion
- [x] Structure-aware chunking (numbered section headers)

### Phase 7: Conversations & Reliability ✅

- [x] Multi-turn chat: conversation memory, follow-up query rewriting, rolling summary
- [x] Structured logging across upload, retrieval, and answer paths
- [x] Startup environment validation with a clear error naming any missing variable
- [x] Error handling around retrieval and LLM calls with correct HTTP status codes
- [x] Per-IP rate limiting (`slowapi`, proxy-aware) and a `/health` endpoint

### Phase 8: Dev/Prod & Deployment ✅

- [x] Dev/prod split behind one interface (`app/services.py`, `ENVIRONMENT` switch)
- [x] Gemini embeddings and listwise re-ranking for prod
- [x] Dockerized and deployed on Render, with a cron-based keep-alive

### Phase 9: Auth, Multi-Tenancy & Async Refactor ✅

- [x] User accounts (`/auth/signup`, `/auth/login`, `/auth/logout`, `/auth/me`) with bcrypt + JWT
- [x] Hybrid model: guests share a public demo corpus, logged-in users get a private one (`resolve_ownership`)
- [x] `documents` parent table + `chunks` child table with `ON DELETE CASCADE` and an ownership `CHECK` constraint
- [x] Tenant filters on both ChromaDB and PostgreSQL queries
- [x] SHA-256 file deduplication with scoped unique indexes; duplicate uploads short-circuit before parsing/embedding
- [x] Scoped overwrite: re-uploading a filename replaces only the caller's previous version in both stores
- [x] UUID-anchored chunk IDs (`{document_id}:{chunk_index}`)
- [x] Shared `asyncpg` pool with FastAPI `lifespan` shutdown; `retrieve_relevant_chunks` and `store_chunks` rewritten as native `async def`
- [x] Unified Gemini LLM pipeline (`google-genai` async client) for dev and prod; NVIDIA NIM/`httpx` client removed
- [x] Null-safe PDF text extraction

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
python -m pip install -r requirements-dev.txt

# 4. Add config to .env  (KEY=value, no spaces around "=", no quotes)
ENVIRONMENT=dev
GEMINI_API_KEY=your_key_here
# GEMINI_MODEL=gemini-2.5-flash-lite      # optional override
JWT_SECRET_KEY=a_long_random_string       # required in practice; see warning below
CHROMA_API_KEY=your_key_here
CHROMA_TENANT=your_tenant_here
CHROMA_DATABASE=your_database_here
CHROMA_COLLECTION_NAME=ragify_docs_dev
DATABASE_URL=your_neon_connection_string_here

# 5. Create the Postgres tables (once per database, in Neon's SQL Editor)
#    Copy-paste the SQL from the "Database schema" section below.

# 6. Run the server
uvicorn app.main:app --reload

# 7. Open in browser
# http://127.0.0.1:8000/
```

> **Set `JWT_SECRET_KEY`.** `app/auth/security.py` falls back to a hard-coded placeholder secret if the variable is missing, which would let anyone forge tokens. Always set a long random value in every environment.

### Database schema

Run this once per database (Neon SQL Editor). It creates all three tables, their indexes, the ownership constraints, and the scoped deduplication indexes. The same SQL lives in `app/db/schema.sql`; make sure the two `CREATE UNIQUE INDEX` lines are in that file too.

```sql
-- 1. Users (role: 1 = USER, 2 = ADMIN)
CREATE TABLE users (
    id              TEXT PRIMARY KEY,
    email           VARCHAR(255) UNIQUE NOT NULL,
    first_name      VARCHAR(100) NOT NULL,
    last_name       VARCHAR(100),
    hashed_password VARCHAR(255) NOT NULL,
    role            INTEGER DEFAULT 1 NOT NULL CHECK (role IN (1, 2)),
    created_at      TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);
CREATE INDEX idx_users_email ON users(email);

-- 2. Documents (file registry; parent of chunks)
CREATE TABLE documents (
    id          TEXT PRIMARY KEY,                               -- UUID
    filename    TEXT NOT NULL,
    file_hash   TEXT,                                           -- SHA-256 of raw file bytes
    chunk_count INTEGER DEFAULT 0 NOT NULL,
    user_id     TEXT REFERENCES users(id) ON DELETE CASCADE,    -- NULL for public documents
    is_public   BOOLEAN DEFAULT FALSE NOT NULL,
    created_at  TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    CONSTRAINT chk_documents_ownership CHECK (
        (is_public = TRUE  AND user_id IS NULL) OR
        (is_public = FALSE AND user_id IS NOT NULL)
    )
);
CREATE INDEX idx_documents_user_id   ON documents(user_id);
CREATE INDEX idx_documents_is_public ON documents(is_public);
CREATE INDEX idx_documents_file_hash ON documents(file_hash);

-- Scoped deduplication: one copy of identical bytes per workspace, no cross-tenant leaks
CREATE UNIQUE INDEX uq_documents_public_hash ON documents (file_hash)          WHERE is_public = TRUE;
CREATE UNIQUE INDEX uq_documents_user_hash   ON documents (user_id, file_hash) WHERE is_public = FALSE;

-- 3. Chunks (child of documents; keyword / full-text store)
CREATE TABLE chunks (
    id            TEXT PRIMARY KEY,                             -- "{document_id}:{chunk_index}"
    document_id   TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index   INTEGER DEFAULT 0 NOT NULL,
    user_id       TEXT REFERENCES users(id) ON DELETE CASCADE,
    is_public     BOOLEAN DEFAULT FALSE NOT NULL,
    text          TEXT NOT NULL,
    source        TEXT NOT NULL,                                -- original filename
    search_vector TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    CONSTRAINT chk_chunks_ownership CHECK (
        (is_public = TRUE  AND user_id IS NULL) OR
        (is_public = FALSE AND user_id IS NOT NULL)
    )
);
CREATE INDEX idx_chunks_document_id ON chunks(document_id);
CREATE INDEX idx_chunks_user_id     ON chunks(user_id);
CREATE INDEX idx_chunks_is_public   ON chunks(is_public);
CREATE INDEX idx_search_vector      ON chunks USING GIN(search_vector);
```

**Upgrading an existing database** that still has the old single `chunks(id, text, source, search_vector)` table: the new layout is not backward compatible (chunk IDs and the parent table changed). Drop the old `chunks` table (`DROP TABLE chunks;`), run the SQL above, and clear or re-create your Chroma collection so IDs and metadata match, then re-upload your documents.

### Environment variables

| Variable                      | Dev              | Prod     | Notes                                                                                                                        |
| ----------------------------- | ---------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `ENVIRONMENT`                 | `dev` (or unset) | `prod`   | Exact value `prod` switches embeddings and re-ranking to Gemini                                                              |
| `GEMINI_API_KEY`              | required         | required | Answers, query rewriting, summaries (both modes); embeddings/re-rank (prod)                                                  |
| `GEMINI_MODEL`                | optional         | optional | Model _name_ (e.g. `gemini-3.5-flash-lite`); defaults to `gemini-2.5-flash-lite`. Never put a model name in `GEMINI_API_KEY` |
| `JWT_SECRET_KEY`              | required         | required | Signs session tokens; never use the fallback                                                                                 |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | optional         | optional | Defaults to 1440 (24 hours)                                                                                                  |
| `CHROMA_API_KEY`              | required         | required | Chroma Cloud                                                                                                                 |
| `CHROMA_TENANT`               | required         | required |                                                                                                                              |
| `CHROMA_DATABASE`             | required         | required |                                                                                                                              |
| `CHROMA_COLLECTION_NAME`      | optional         | optional | Defaults to `ragify_docs`; use a different name per environment                                                              |
| `DATABASE_URL`                | required         | required | Neon Postgres; used by both SQLAlchemy (auth) and `asyncpg` (search)                                                         |

`NVIDIA_API_KEY` and `NVIDIA_URL` are no longer used. On startup, `validate_env()` checks the variables required for the current mode and names any that are missing; keep its list in sync with the table above.

---

## Deployment

- **Platform:** Render (Docker web service)
- **Live URL:** https://ragify-zb6i.onrender.com
- **Image:** built from the `Dockerfile`: `python:3.12-slim`, installs `requirements.txt` (no PyTorch), runs as a non-root user
- **Start command:** set in the Dockerfile: `uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1`
- **Environment variables:** set in the Render dashboard (the table above, with `ENVIRONMENT=prod`)
- **Workers:** exactly one. Conversation memory and rate-limit counters live in process memory, so with multiple workers a follow-up can land on a process that has never seen the conversation. One worker also keeps memory use inside small plans' limits.
- **Shutdown:** the `lifespan` hook closes the shared `asyncpg` pool cleanly.
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

It is a regression suite, not full coverage, and it was measured on the dev stack before the multi-tenant changes. Retrieval now requires an ownership scope (`is_public`, `user_id`), so the harness must query the public corpus (or a test user's scope). Re-run it, and re-run it against the prod configuration, before trusting retrieval quality.

---

## Key Decisions

- **Hybrid retrieval over vector-only.** Embeddings capture meaning but miss exact terms (numbers, names). Postgres full-text covers those, and Reciprocal Rank Fusion merges the two ranked lists without needing comparable scores.
- **Postgres full-text over Elasticsearch and `rank_bm25`.** Elasticsearch adds a service, a recurring cost, and dual-write complexity. `rank_bm25` can't update incrementally, so every upload would rebuild the whole index. Neon's free tier plus `tsvector` gives incremental indexing for far less effort.
- **OR-based keyword matching (`to_tsquery` with `|`), not `plainto_tsquery`.** AND-matching returned nothing whenever a question's wording (e.g. "Transformer") didn't appear in the same chunk as the answer (e.g. "Adam optimizer"). The re-ranker cleans up the looser matches. Found through the eval harness.
- **Retrieve wide, then re-rank narrow.** Cosine search is fast but imprecise; a re-ranker scores (question, chunk) pairs jointly. Retrieve 20, keep 3.
- **Structure-aware chunking.** A chunk that mixed "5.2 Hardware and Schedule" with "5.3 Optimizer" scored -5.14 on the cross-encoder despite containing the correct answer. Splitting at detected section headers fixed it (75% → 91.7% on the eval).
- **Shared `asyncpg` pool on the main event loop.** The earlier design used one connection per call because `retrieve_relevant_chunks` and `store_chunks` were sync functions run via `run_in_threadpool`, each calling `asyncio.run()`; a pool created in one call's temporary event loop broke in the next (`InterfaceError` / `Event loop is closed`). Making both functions native `async def` lets one pool live on FastAPI's loop and be closed in the `lifespan` hook.
- **Offload only blocking sync work.** Parsing, embedding, Chroma calls, and local re-ranking go through `run_in_threadpool` so a slow request can't freeze the event loop. Async I/O (`asyncpg`, `client.aio`) is awaited directly. Wrapping an `async def` in `run_in_threadpool` returns an unawaited coroutine.
- **Parent `documents` table with UUID-anchored chunks** (`{document_id}:{chunk_index}`) replaces filename-based chunk IDs. Replacing a document is one `DELETE` on the parent plus `ON DELETE CASCADE`, identical text in two documents can't collide, and every chunk traces to exactly one owner.
- **Ownership is enforced in the database, not just in code.** The `CHECK ((is_public AND user_id IS NULL) OR (NOT is_public AND user_id IS NOT NULL))` constraint rejects any attempt to pass a guest sentinel (like `"demo-user"`) as a user ID, which forces explicit ownership resolution (`resolve_ownership`) at the API boundary.
- **Hybrid public-demo + private tenancy.** Guests can evaluate the app instantly with no sign-up; anyone who needs privacy creates an account and gets an isolated corpus. The same `(is_public, user_id)` pair drives filters in both Chroma and Postgres, so the two stores can't disagree about visibility.
- **Hash raw bytes before parsing or embedding.** A SHA-256 check scoped to the caller's workspace prevents duplicate records and saves embedding API quota on accidental re-uploads. Scoped unique indexes enforce it at the database level without leaking across tenants.
- **Scoped overwrite by filename.** A modified file with an existing name atomically replaces only the caller's previous version (Postgres transaction first, then Chroma purge). Trade-off: the two stores aren't updated in one transaction (see Known Limitations).
- **One LLM SDK for both environments.** `google-genai`'s async client now handles answer streaming, query rewriting, and summarization in dev and prod, replacing separate `httpx` SSE and Gemini implementations. The `ENVIRONMENT=prod` switch is kept only for embeddings and re-ranking, so heavy local ML dependencies stay behind lazy imports in `app/services.py` and prod never loads PyTorch.
- **Normalized embeddings on both ingestion and query sides**, with Chroma set to cosine distance. Normalizing only one side silently distorted similarity scores.
- **Startup validation runs before any module that creates network clients.** `ingestion.py` connects to Chroma at import time, so `main.py` calls `load_dotenv()` and `validate_env()` first and fails with a clear message instead of a library crash.
- **Conversation memory is in-process, with a rolling summary.** Chosen over a database-backed store to avoid extra infrastructure; the cost is that memory is lost on restart and isn't shared across workers.
- **Per-IP rate limiting with in-memory counters.** For a public demo the things worth protecting are upstream quotas and compute, and per-IP throttling does that with no extra infrastructure. It is only correct with a single worker, which the deployment already requires. The client IP comes from `X-Forwarded-For` because behind a reverse proxy the socket address is the proxy's.
- **A cheap, unthrottled `/health` endpoint** that touches no database or model, used by the keep-alive cron job.

---

## Architecture Decision Record: Authentication & Multi-Tenancy

**Status:** Accepted. Supersedes the earlier "no accounts, one shared corpus" decision.
**Decision:** Hybrid model. Unauthenticated visitors use a shared public demo corpus; authenticated users get a strictly isolated private corpus. Abuse and cost are still mitigated with per-IP rate limits.

**Why:**

- **Frictionless evaluation stays.** Recruiters, hiring managers, and engineers can try live document Q&A immediately, with no sign-up, password, or OAuth step.
- **Real privacy when needed.** Users who want to upload their own documents can sign up and be certain other users and guests cannot retrieve them.
- **No per-session namespaces to orphan.** Anonymous visitors share one public corpus instead of each getting a throwaway namespace, which would leave abandoned embeddings and SQL rows behind in Chroma and Neon until a TTL reaper removed them.
- **Isolation is enforced at every layer.** Ownership is resolved once at the API boundary, applied as filters in both Chroma and Postgres, and backed by a database `CHECK` constraint.
- **Quota protection by throttling.** Upstream quotas and compute are guarded by per-IP rate limits rather than account identity.

**Consequences accepted:**

- The public corpus is shared and writable by every guest. Anything uploaded while logged out is searchable by everyone, so do not upload private documents without logging in.
- Re-uploading an existing filename in the public corpus replaces that file for all guests.
- Public uploads accumulate in the free-tier stores until someone purges them.
- Rate limits throttle, but they cannot stop a determined actor using many IPs, and the Gemini quota is shared globally.
- Auth is deliberately minimal: no email verification, password reset, or account/document deletion endpoints yet.

---

## Known Limitations

The main limitation is listed first. The rest are split into **intentional design choices** (deliberate trade-offs that match the project's scope) and **real limitations** (bugs, risks, or gaps that should be fixed or re-evaluated).

### Main limitations

- **Keyword search is tuned for English.** The Postgres full-text index uses the `english` configuration (stemming, stopwords), and the query builder's stopword list is English. Documents in other languages still get vector search, but keyword matching is weaker, and languages written without spaces (such as Chinese or Japanese) are not supported by this tokenization. The dev embedding model (MiniLM) is English-oriented as well. Supporting more languages would mean per-language text-search configs or language detection at ingestion.

### By design (intentional trade-offs)

- **Shared public demo corpus for guests.** Visitors can try the app without signing up. Anything uploaded while logged out is visible to every guest, so private documents should only be uploaded after logging in (see the ADR above).
- **Same filename = overwrite within a scope.** A modified file with an existing name replaces the caller's previous version, which avoids building a document-management UI.
- **Deduplication is exact-bytes only (SHA-256).** This is a fast-path check. Semantic or cross-format deduplication (the same text as a PDF and a DOCX) is out of scope.
- **Logout only clears the cookie.** JWTs are stateless, so a copied token stays valid until it expires. Revocation would need a token blacklist, which is not built.
- **Conversation memory and rate-limit counters are in-process.** This avoids running Redis on a single-worker deployment. Both reset on restart and are not shared across workers.
- **Text-based files only.** Scanned or image-only PDFs are not OCR'd (their pages are skipped safely), and uploads are capped at 10 MB.

### Real limitations

**Security**

- **`JWT_SECRET_KEY` has an insecure fallback.** If the variable is unset, tokens are signed with a hard-coded placeholder, and anyone could forge them. It should fail at startup instead.
- **Cookie flags are wrong for production.** The session cookie is not `HttpOnly`, so an XSS bug could read it. `router.py` also only sets `Secure` when `ENVIRONMENT == "production"`, while the rest of the app uses `prod`, so `Secure` is never enabled on Render.
- **No login throttling or password rules.** `/auth/login` and `/auth/signup` are not rate limited and accept any password, which leaves login open to brute force.
- **`/memory/{conversation_id}/summary` has no ownership check.** Anyone who knows a `conversation_id` can read that conversation's summary, because conversations aren't tied to a user.

**Data integrity**

- **Postgres and Chroma writes are not atomic, and there is no Chroma cleanup on user deletion.** Postgres commits first, then Chroma is purged and upserted. A failure in between leaves the stores out of sync, and deleting a user cascades their Postgres rows but leaves orphan vectors in Chroma.

**Retrieval quality**

- **Section detection is narrow.** Only `N.N Title` headers are detected. Markdown `#` headers and single-number headers ("1 Introduction") fall back to word-count chunking, which can mix topics inside a chunk.
- **The prod re-ranker truncates each chunk to 400 characters.** An answer in the second half of a 200-word chunk can be ranked down or ignored.
- **The 0.3 similarity threshold was tuned on MiniLM.** It was never calibrated for `gemini-embedding-001`, whose similarity distribution differs.

**Reliability & evaluation**

- **Conversation memory raises if more than 20 turns pile up unsummarized.** If the background summarizer keeps failing, `add_turn` throws an unhandled error instead of degrading gracefully.
- **Prod embeddings are requested one chunk per API call.** Large uploads are slow and burn Gemini requests-per-minute quota; batch embedding would fix both.
- **No retry/backoff on Gemini errors, and the eval set is small and pre-tenancy.** Transient `429`/`503` errors fail immediately (surfacing as a generic `500`, or an `[Error: ...]` line when streaming). The 12-question, single-document eval was measured on the dev stack before multi-tenancy, so retrieval quality on the prod stack and on scoped data has not been re-evaluated.

---

## Learnings Log

- Embeddings capture meaning, not exact words: cosine similarity for "cat/mat" vs "feline/rug" was 0.556, vs "stock market" 0.097.
- Chunk overlap matters: 200-word chunks with 50-word overlap on 1000 words produced 7 chunks, matching manual math.
- Distance metrics matter: cosine distance must be configured explicitly for threshold filtering to work; L2 has no fixed range.
- `python -m pip` / `python -m module` avoids Windows venv PATH mismatches. Bare `pip install` once installed into the global Python instead of the venv, producing a `ModuleNotFoundError` that looked like a code bug.
- Generators (`yield`) don't run until iterated, so one generator function can power both a "wait for everything" flow and a "forward pieces immediately" flow.
- A bi-encoder (embedding model) and a cross-encoder (re-ranker) solve different problems: bi-encoders embed question and chunk _separately_ for fast approximate search; cross-encoders score them _jointly_ for precision but are too slow to run against everything. Hence retrieve wide, re-rank narrow.
- **`asyncpg` pool lifecycle & native async orchestration:** pools are bound to the event loop that created them. Earlier, sync functions spun up temporary event loops with `asyncio.run()` inside `run_in_threadpool`, which broke pool reuse. Refactoring `retrieve_relevant_chunks` and `store_chunks` into native `async def` functions on FastAPI's main loop lets one pool be shared across requests and closed cleanly through the `lifespan` hook.
- **Never wrap `async def` functions in `run_in_threadpool`.** It returns an unawaited coroutine object instead of executing it. Keep async I/O (`asyncpg`, `client.aio`) on the main loop with `await`, and offload only blocking synchronous libraries (`pypdf`, `python-docx`, `chromadb`, local `SentenceTransformer` / `CrossEncoder`) to worker threads.
- **Database `CHECK` constraints prevent sentinel-ID bugs.** Enforcing `(is_public = TRUE AND user_id IS NULL) OR (is_public = FALSE AND user_id IS NOT NULL)` catches any attempt to pass a guest sentinel ID (like `"demo-user"`) as a user foreign key, forcing explicit ownership resolution at the API boundary.
- **Hash raw bytes before parsing or embedding.** Computing a SHA-256 `file_hash` and checking scoped uniqueness before `parse_file` and `embed_chunks` prevents duplicate records and saves embedding quota on accidental re-uploads.
- **Unifying the LLM SDK simplifies multi-environment maintenance.** Using `google-genai`'s async client in both dev and prod removed the need to maintain two streaming and retry implementations (`httpx` SSE vs. Gemini SDK), while heavy local ML dependencies stay behind lazy imports in `app/services.py`.
- **`pypdf` returns `None` on non-text pages.** Scanned or image-only pages return `None` from `page.extract_text()`; coercing with `(page.extract_text() or "")` prevents a `TypeError` during string concatenation.
- Postgres full-text defaults to AND between query words. A question and its answer can share no significant words, so AND-matching can return nothing even when the right content exists.
- An eval harness doesn't need per-document coverage. A small, growing, representative sample catches systemic retrieval and chunking problems, and it should grow from real failures.
- A cross-encoder can score a chunk very negatively even when it contains the literal answer, if the chunk also holds unrelated content. That is a chunking problem disguised as a retrieval problem, visible only by inspecting real chunk text and real scores.
- A failing eval case can be a test-authoring problem (the expected phrase doesn't match the paper's wording) rather than a retrieval bug. Check the retrieved text before changing code or the test.
- `import` executes a module's top-level code, so import order matters when modules create network clients at import time.
- `load_dotenv()` doesn't override variables already set in the shell, and uvicorn's auto-reload doesn't watch `.env`. Restart the server after editing it.
- Retrying a streaming call is only safe if nothing has been yielded yet; otherwise the retry duplicates text the user already received.
- Dev and prod stacks need separate data stores: embedding dimensions differ, and filename-scoped replacement in a shared database lets one environment erase the other's data.
- Behind a reverse proxy the socket address is the proxy's, so per-IP rate limiting has to read the forwarded-for header; in-memory limit counters are only correct with a single worker.

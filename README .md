# RAGify

A Retrieval-Augmented Generation (RAG) system built from scratch to learn how RAG pipelines work end-to-end — file upload → embedding → vector storage → retrieval → LLM answer generation, restricted strictly to the uploaded context.

Built with **Python**, **FastAPI**, **ChromaDB**, **sentence-transformers**, and **NVIDIA NIM (build.nvidia.com)** free-tier LLMs.

---

## Goal

- Upload a document → chunk it → embed it → store in a vector database
- Ask a question → retrieve the most relevant chunks → send to an LLM
- LLM answers **only** using the retrieved context (no hallucination / no going out-of-context)

---

## Tech Stack

| Component | Choice | Why |
|---|---|---|
| Backend framework | FastAPI | Async, fast, easy to build APIs |
| Vector DB | ChromaDB (PersistentClient, cosine similarity) | Local, zero-setup, great for learning |
| Embedding model | `sentence-transformers` (`all-MiniLM-L6-v2`), local | No rate limits, free, fast for bulk chunking |
| LLM | NVIDIA NIM — `meta/llama-3.1-8b-instruct` (dev) | Free tier, fast enough for iteration; swap to `meta/llama-3.1-70b-instruct` for quality |
| File parsing | `pypdf` (.pdf), `python-docx` (.docx), built-in decode (.txt) | Lightweight, sufficient for text-based documents |
| Frontend | Plain HTML/JS, served via FastAPI StaticFiles | Kept focus on RAG pipeline, not frontend tooling |
| Env management | `venv` | Simple, standard |

---

## Architecture

```
RAGify/
├── app/
│   ├── main.py          # FastAPI app entrypoint — /upload, /ask, /ask/stream, frontend route
│   ├── ingestion.py      # File parsing + chunking + embedding + storing + dedup
│   ├── retrieval.py      # Query embedding + top-k retrieval + similarity threshold filtering
│   └── llm.py            # NVIDIA NIM API call (streaming), prompt construction
├── static/
│   └── index.html        # Browser frontend (upload + ask + ask-streaming)
├── chroma_db/             # Local Chroma persistent storage (gitignored)
├── .env                   # API keys (NEVER commit this)
├── .gitignore
├── requirements.txt
└── README.md
```

### Pipeline

**Ingestion:**
`file upload → parse text (.txt/.pdf/.docx) → chunk (word-based, with overlap) → embed each chunk (local model) → dedup (content-hash + filename cleanup) → store (vector + text + metadata) in Chroma`

**Query (non-streaming, `/ask`):**
`question → embed → retrieve top-k similar chunks (cosine similarity, threshold-filtered) → build prompt (system instructions + context + question) → call NVIDIA LLM → return {answer, sources}`

**Query (streaming, `/ask/stream`):**
Same retrieval + prompt steps, but the LLM's response is streamed token-by-token to the browser as it's generated (no sources returned in this mode).

---

## Progress Tracker

### Phase 1 — Core Concepts ✅
- [x] Embeddings (text → meaning as vectors, semantic similarity)
- [x] Chunking (why size + overlap matter)
- [x] Vector databases (why not a normal DB, why store text + vector together)

### Phase 2 — Environment Setup ✅
- [x] Python venv created
- [x] NVIDIA NIM API key generated
- [x] Install dependencies
- [x] Create `.env` file
- [x] Create folder structure

### Phase 3 — Ingestion Pipeline ✅
- [x] File upload endpoint (`/upload`)
- [x] Parse file (.txt / .pdf / .docx)
- [x] Chunking function (word-based, size + overlap configurable)
- [x] Call embedding model on each chunk
- [x] Store chunks + embeddings + metadata in Chroma

### Phase 4 — Query Pipeline ✅
- [x] `/ask` endpoint
- [x] Embed incoming question
- [x] Retrieve top-k chunks from Chroma
- [x] Build final prompt (system + context + question)
- [x] Call NVIDIA LLM, return answer

### Phase 5 — Guardrails ✅
- [x] System prompt to restrict answers to context only
- [x] Similarity score threshold → "I don't know" fallback when nothing relevant is retrieved
- [x] Input validation / error handling

### Phase 6 — Polish / Extend 🔄
- [x] .docx file support
- [x] Content-hash + filename-based deduplication
- [x] Minimal frontend for testing (upload + ask)
- [x] Streaming responses (`/ask/stream`, dual Ask/Ask-Streaming buttons)
- [ ] Multi-turn chat history
- [ ] Reranking retrieved chunks
- [ ] Deploy live (Render/Railway/etc.)

---

## Setup Instructions

```bash
# 1. Clone / open project folder
# 2. Activate virtual environment
source venv/Scripts/activate   # Git Bash on Windows
# or venv\Scripts\activate.bat on CMD

# 3. Install dependencies
python -m pip install -r requirements.txt

# 4. Add NVIDIA API key to .env
NVIDIA_API_KEY=your_key_here

# 5. Run the server
uvicorn app.main:app --reload

# 6. Open in browser
# http://127.0.0.1:8000/
```

---

## Key Decisions & Notes

- Chose **ChromaDB** (PersistentClient) over Pinecone/Qdrant/FAISS for local, zero-infra learning setup.
- Configured ChromaDB collection to use **cosine similarity** (`hnsw:space: cosine`) instead of default L2 distance — L2 has no fixed range, made threshold filtering meaningless. Similarity threshold set to `0.3` based on real testing — may need tuning per use case.
- Chose local `sentence-transformers` (`all-MiniLM-L6-v2`) for embeddings over NVIDIA NIM embedding endpoint — avoids rate limits during bulk ingestion, saves NVIDIA free credits for LLM calls only.
- Chose `meta/llama-3.1-8b-instruct` for development speed; can swap to `meta/llama-3.1-70b-instruct` for quality later — just a `.env`/model-string change.
- Rejected `meta/llama-3.2-90b-vision-instruct` — vision model, unnecessary overhead for text-only RAG.
- Using `pypdf` for PDF text extraction and `python-docx` for Word docs — simple, lightweight, sufficient for text-based documents (not scanned/image PDFs, which would need OCR).
- Implemented dual deduplication: content-hash IDs (MD5 of chunk text) prevent identical content from being stored twice across different filenames; filename-based delete-before-insert handles cleanup when a file is edited and re-uploaded under the same name.
- Consolidated LLM calling into a single `call_llm_stream()` generator function, reused by both `/ask` (via `"".join()`) and `/ask/stream` (via `StreamingResponse`) — avoids duplicate logic and duplicate bugs.
- Fixed edge case: NVIDIA occasionally sends stream chunks with an empty `choices` list (metadata/keep-alive events) — now skipped safely instead of crashing.
- Chose plain HTML/JS over Streamlit/React for the frontend — kept focus on learning the RAG pipeline itself rather than frontend tooling. Served directly via FastAPI's `StaticFiles`/`FileResponse`.
- Frontend has two buttons: **"Ask"** (non-streaming, shows sources) and **"Ask (Streaming)"** (progressive text, no sources) — sources remain the primary trust mechanism for RAGify; streaming is a UX bonus, not a replacement.
- **Known limitation:** dedup does not track "this content appears in multiple source files" — the most recently uploaded file's metadata wins if identical content is re-uploaded under a different filename.

---

## Learnings Log

- Embeddings capture meaning, not exact words — proven with cosine similarity: "cat/mat" vs "feline/rug" scored 0.556, vs "stock market" scored 0.097.
- Chunk overlap matters — 200-word chunks with 50-word overlap on 1000 words produced 7 chunks, matching manual math (step size = chunk_size - overlap).
- Distance metrics matter: ChromaDB's default L2 distance has no fixed range, making `1 - distance` a meaningless "similarity" score. Cosine distance (bounded 0-2) must be explicitly configured for threshold-based filtering to work.
- `python -m pip` / `python -m module` avoids Windows venv PATH mismatches between `python` and `pip` binaries.
- Running a script directly (`python app/file.py`) vs as a module (`python -m app.file`) changes what's on Python's import path — matters once files start importing from each other.
- Generators (`yield`) don't run until iterated — the same generator function can power both a "wait for everything" flow (`"".join()`) and a "forward pieces immediately" flow (`StreamingResponse`).


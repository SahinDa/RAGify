import os
import asyncpg
from typing import Optional
import re
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def build_or_query(question: str) -> str:
    """
    Converts a natural-language question into an OR-joined tsquery string,
    so a chunk matching ANY significant word is returned, not just chunks
    matching ALL words.
    """
    words = re.findall(r"\w+", question.lower())
    stopwords = {"what", "is", "are", "was", "were", "the", "a", "an", "used",
                 "to", "in", "of", "for", "how", "many", "does", "do", "on",
                 "which", "and", "or"}
    keywords = [w for w in words if w not in stopwords]
    if not keywords:
        return question  # fallback: don't break if everything got filtered out
    return " | ".join(keywords)

async def check_duplicate_document(
    file_hash: str,
    is_public: bool,
    user_id: Optional[str],
) -> Optional[dict]:
    """
    Checks if a document with the exact SHA-256 file_hash already exists
    within the caller's ownership scope.
    """
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        if is_public:
            row = await conn.fetchrow(
                """
                SELECT id::text AS document_id, filename, chunk_count
                FROM documents
                WHERE is_public = TRUE AND file_hash = $1
                LIMIT 1
                """,
                file_hash,
            )
        else:
            row = await conn.fetchrow(
                """
                SELECT id::text AS document_id, filename, chunk_count
                FROM documents
                WHERE is_public = FALSE AND user_id = $1 AND file_hash = $2
                LIMIT 1
                """,
                user_id,
                file_hash,
            )
        return dict(row) if row else None
    finally:
        await conn.close()
   
        
async def index_document_and_chunks(
    document_id: str,
    filename: str,
    file_hash: str,
    chunks: list[str],
    ids: list[str],
    is_public: bool,
    user_id: Optional[str],
) -> list[str]:
    """
    Atomically replaces any prior version of the same filename within the same
    ownership scope, inserts the new parent `documents` record, and bulk-inserts
    all child `chunks` with ownership flags.
    Returns any superseded document_ids so caller can purge stale ChromaDB vectors.
    """
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        async with conn.transaction():
            # Remove prior versions of the same filename in the same scope
            if is_public:
                old_rows = await conn.fetch(
                    """
                    DELETE FROM documents
                    WHERE is_public = TRUE AND filename = $1
                    RETURNING id::text AS id
                    """,
                    filename,
                )
            else:
                old_rows = await conn.fetch(
                    """
                    DELETE FROM documents
                    WHERE is_public = FALSE AND user_id = $1 AND filename = $2
                    RETURNING id::text AS id
                    """,
                    user_id,
                    filename,
                )

            stale_doc_ids = [r["id"] for r in old_rows]

            await conn.execute(
                """
                INSERT INTO documents (id, filename, file_hash, is_public, user_id, chunk_count)
                VALUES ($1::uuid, $2, $3, $4, $5, $6)
                """,
                document_id,
                filename,
                file_hash,
                is_public,
                user_id,
                len(chunks),
            )

            await conn.executemany(
                """
                INSERT INTO chunks (id, document_id, chunk_index, text, source, is_public, user_id)
                VALUES ($1, $2::uuid, $3, $4, $5, $6, $7)
                """,
                [
                    (cid, document_id, idx, text, filename, is_public, user_id)
                    for idx, (cid, text) in enumerate(zip(ids, chunks))
                ],
            )

            return stale_doc_ids
    finally:
        await conn.close()


async def keyword_search(question: str,top_k: int = 20) -> list[str]:
    """
    Full-text search using Postgres tsvector/tsquery ranking.
    Returns chunk IDs ranked by relevance (best first).
    """
    
    query_string = build_or_query(question)
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        rows = await conn.fetch(
            """
            SELECT id, ts_rank(search_vector,query) AS rank
            FROM chunks, to_tsquery('english', $1) query
            WHERE search_vector @@query
            ORDER BY rank DESC
            LIMIT $2
            """,
            query_string, top_k
        )

        return [row["id"] for row in rows]
    
    finally:
        await conn.close()
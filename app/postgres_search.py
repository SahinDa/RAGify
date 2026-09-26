import os
import asyncpg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

_pool = None

async def get_pool():
    """
    Returns a reusable connection pool to Neon Postgres.
    Created once, reused across calls - avoids opening a new connection every time.
    """
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(DATABASE_URL)

    return _pool


async def index_chunks(chunks: list[str], ids: list[str], source_filename: str):
    """
    Inserts chunks into Postgres for full-text search.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM chunks WHERE source = $1", source_filename)
        await conn.executemany(
            "INSERT INTO chunks (id,text,source) VALUES ($1, $2, $3)",
            [(cid, text, source_filename) for cid, text in zip(ids,chunks)]
        )



async def keyword_search(question: str,top_k: int = 20) -> list[str]:
    """
    Full-text search using Postgres tsvector/tsquery ranking.
    Returns chunk IDs ranked by relevance (best first).
    """

    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, ts_rank(search_vector, plainto_tsquery('english',$1)) AS rank
            FROM chunks
            WHERE search_vector @@ plainto_tsquery('english', $1)
            ORDER BY rank DESC
            LIMIT $2
            """,
            question, top_k
        )

        return [row["id"] for row in rows]
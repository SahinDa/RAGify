import os
import asyncpg
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


async def index_chunks(chunks: list[str], ids: list[str], source_filename: str):
    """
    Inserts chunks into Postgres for full-text search.
    """
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        await conn.execute("DELETE FROM chunks WHERE source = $1", source_filename)
        await conn.executemany(
            "INSERT INTO chunks (id,text,source) VALUES ($1, $2, $3)",
            [(cid, text, source_filename) for cid, text in zip(ids,chunks)]
        )
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
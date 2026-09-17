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
            """
            question, top_k
        )

        return [row["id"] for row in rows]
from app.ingestion import collection
from typing import Optional
from fastapi.concurrency import run_in_threadpool
from app.postgres_search import keyword_search
from app.services import embedder, reranker
import logging
logger = logging.getLogger(__name__)


def _build_chroma_where(is_public: bool, user_id: Optional[str]) -> dict:
    """Builds the strict tenant isolation filter for ChromaDB queries."""
    if is_public:
        return {"is_public": True}
    return {
        "$and": [
            {"is_public": False},
            {"user_id": user_id},
        ]
    }
 
def _vector_search_sync(
    question: str,
    where_filter: dict,
    candidate_k: int,
    similarity_threshold: float,
) -> tuple[dict, list[str]]:
    """Synchronous ChromaDB vector search helper executed in a worker thread."""
    question_embedding = [embedder.embed_query(question)]

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=candidate_k,
        where=where_filter,
    )

    documents = results["documents"][0] if results["documents"] else []
    distances = results["distances"][0] if results["distances"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    vector_ids = results["ids"][0] if results["ids"] else []

    chunk_lookup = {}
    filtered_vector_ids = []
    for chunk_id, doc, dist, meta in zip(vector_ids, documents, distances, metadatas):
        similarity = 1.0 - dist
        chunk_lookup[chunk_id] = {"text": doc, "metadata": meta}
        if similarity >= similarity_threshold:
            filtered_vector_ids.append(chunk_id)

    return chunk_lookup, filtered_vector_ids
    
async def retrieve_relevant_chunks(
    question: str,
    is_public: bool,
    user_id: Optional[str],
    top_k: int = 3,
    candidate_k: int = 20,
    similarity_threshold: float = 0.3,
):
    """
    Tenant-isolated hybrid retrieval pipeline:
    1. Scoped Vector search (ChromaDB) - top candidate_k, cosine pre-filtered.
    2. Scoped Keyword search (Postgres full-text) - top candidate_k.
    3. Merge both ranked lists via Reciprocal Rank Fusion.
    4. Re-rank the fused candidates with cross-encoder / Gemini listwise reranker.
    5. Return top_k.
    """
    logger.info(f"Retrieving for question: '{question}' (is_public={is_public}, user_id={user_id})")

    where_filter = _build_chroma_where(is_public=is_public, user_id=user_id)

    # Stage 1: Scoped Vector Search (offloaded to threadpool so event loop stays free)
    chunk_lookup, filtered_vector_ids = await run_in_threadpool(
        _vector_search_sync,
        question,
        where_filter,
        candidate_k,
        similarity_threshold,
    )
    logger.info(f"Vector search: {len(filtered_vector_ids)}/{candidate_k} passed threshold")

    # Stage 2: Scoped Keyword Search (Postgres)
    keyword_ids = await keyword_search(
        question=question,
        is_public=is_public,
        user_id=user_id,
        top_k=candidate_k,
    )

    missing_ids = [cid for cid in keyword_ids if cid not in chunk_lookup]
    if missing_ids:
        fetched = await run_in_threadpool(
            collection.get,
            ids=missing_ids,
            include=["documents", "metadatas"],
        )
        for cid, doc, meta in zip(fetched["ids"], fetched["documents"], fetched["metadatas"]):
            chunk_lookup[cid] = {"text": doc, "metadata": meta}

    logger.info(f"Keyword search: {len(keyword_ids)} results")

    # Stage 3: Merge rankings via RRF
    fused_ids = reciprocal_rank_fusion(filtered_vector_ids, keyword_ids)
    if not fused_ids:
        return []

    logger.info(f"Fused candidates: {len(fused_ids)}")

    # Stage 4: Re-rank
    candidates = [chunk_lookup[cid] for cid in fused_ids if cid in chunk_lookup]
    final_results = await run_in_threadpool(reranker.rerank, question, candidates, top_k)

    logger.info(f"Returning top {len(final_results)} after re-ranking")
    return final_results

def reciprocal_rank_fusion(vector_ids: list[str], keyword_ids: list[str], k: int = 60) -> list[str]:
    """
    Merges two ranked ID lists (vector search results, keyword search results)
    into one fused ranking using Reciprocal Rank Fusion.

    Formula: for each list, a chunk at rank position `r` (0-indexed) contributes
    a score of 1 / (k + r + 1). Scores are summed across both lists, so a chunk
    that ranks well in BOTH lists ends up highest overall.
    """
    scores = {}

    for rank, chunk_id in enumerate(vector_ids):
        scores[chunk_id] = scores.get(chunk_id,0) + 1 / (k + rank + 1)

    for rank, chunk_id in enumerate(keyword_ids):
        scores[chunk_id] = scores.get(chunk_id,0) + 1 / (k + rank + 1)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_id for chunk_id, score in fused]    

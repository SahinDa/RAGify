import asyncio
from app.ingestion import embedding_model, collection
from app.postgres_search import keyword_search
from sentence_transformers import CrossEncoder

# Load once at module level — same pattern as embedding_model, avoids reload per request
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


def retrieve_relevant_chunks(
    question: str,
    top_k: int = 3, 
    candidate_k: int = 20,
    similarity_threshold: float = 0.3
):
    """
    Hybrid retrieval pipeline:
    1. Vector search (ChromaDB) - top candidate_k, cosine pre-filtered.
    2. Keyword search (Postgres full-text) - top candidate_k.
    3. Merge both ranked lists via Reciprocal Rank Fusion.
    4. Re-rank the fused candidated with a cross-encoder.
    5. Return top_k.

    Returns:
        List of dicts: [{"text": ..., "score": ..., "metadata": ...}, ...]
        `score` here is the cross-encoder relevance score, not cosine similarity.
    """
    # Stage 1: Vector Search 
    question_embedding = embedding_model.encode([question],normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=candidate_k
    )

    
    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]
    vector_ids = results["ids"][0]

    chunk_lookup = {}
    filtered_vector_ids = []
    for chunk_id, doc, dist, meta in zip(vector_ids,documents, distances, metadatas):
        similarity = 1 - dist  # rough conversion; good enough for thresholding
        chunk_lookup[chunk_id] = {"text":doc, "metadata": meta}
        if similarity >= similarity_threshold:
            filtered_vector_ids.append(chunk_id)

    
    # Stage 2: Keyword search (Postgres)
    keyword_ids = asyncio.run(keyword_search(question,top_k=candidate_k))

    missing_ids = [cid for cid in keyword_ids if cid not in chunk_lookup]
    if missing_ids:
        fetched = collection.get(ids=missing_ids,include=["documents","metadatas"])
        for cid, doc, meta in zip(fetched["ids"], fetched["documents"], fetched["metadatas"]):
            chunk_lookup[cid] = {"text": doc, "metadata": meta}

    # Stage 3: Merge rankings via RRF
    fused_ids = reciprocal_rank_fusion(filtered_vector_ids,keyword_ids)

    if not fused_ids:
        return []


    # Stage 4: Cross-encoder re-rank
    candidates = [chunk_lookup[cid] for cid in fused_ids if cid in chunk_lookup] 
    pairs = [[question , c["text"]] for c in candidates]
    rerank_scores = reranker.predict(pairs)

    for c , score in zip(candidates, rerank_scores):
        c["score"] = float(score)

    candidates.sort(key=lambda c: c["score"], reverse=True)    

    return candidates[:top_k]


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

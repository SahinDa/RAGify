from app.ingestion import embedding_model, collection
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
    Two-stage retrieval:
    1.Embed the question, pull `candidate_k` candidates from chromaDB (wide net,
    fast bi-encoded search), pre-filtered by cosine similarity_threshold.

    2.Re-rank those candidates with a cross-encoded (slower, but scores question+chunk
    together for much higher precision), keep top_k.

    Returns:
        List of dicts: [{"text": ..., "score": ..., "metadata": ...}, ...]
        `score` here is the cross-encoder relevance score, not cosine similarity.
    """
    question_embedding = embedding_model.encode([question],normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=candidate_k
    )

    

    # Chroma returns distances (lower = more similar) for the default metric.
    # We convert to a similarity-like score for easier thresholding.
    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    # Stage 1: cheap consine pre-filter
    candidates = []
    for doc, dist, meta in zip(documents, distances, metadatas):
        similarity = 1 - dist  # rough conversion; good enough for thresholding
        if similarity >= similarity_threshold:
            candidates.append({"text":doc,"metadata": meta})


    if not candidates:
        return []        

    # Stage 2 : cross-encoded re-rank  - scores (question, chunk) pairs jointly
    pairs = [[question , c["text"]] for c in candidates]
    rerank_scores = reranker.predict(pairs)

    for c , score in zip(candidates, rerank_scores):
        c["score"] = float(score)

    candidates.sort(key=lambda c: c["score"], reverse=True)    

    return candidates[:top_k]
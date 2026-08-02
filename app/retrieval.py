from app.ingestion import embedding_model, collection


def retrieve_relevant_chunks(question: str, top_k: int = 3, similarity_threshold: float = 0.3):
    """
    Embeds the question, searches ChromaDB for top-k similar chunks,
    and filters out chunks below the similarity threshold.

    Returns:
        List of dicts: [{"text": ..., "score": ..., "metadata": ...}, ...]
    """
    question_embedding = embedding_model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k
    )

    relevant_chunks = []

    # Chroma returns distances (lower = more similar) for the default metric.
    # We convert to a similarity-like score for easier thresholding.
    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    for doc, dist, meta in zip(documents, distances, metadatas):
        similarity = 1 - dist  # rough conversion; good enough for thresholding
        if similarity >= similarity_threshold:
            relevant_chunks.append({
                "text": doc,
                "score": similarity,
                "metadata": meta
            })

    return relevant_chunks
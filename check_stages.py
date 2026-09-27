import asyncio
from app.ingestion import embedding_model, collection
from app.postgres_search import keyword_search

question = "What optimizer was used to train the Transformer?"
candidate_k = 20
similarity_threshold = 0.3

# Stage 1: Vector search
question_embedding = embedding_model.encode([question], normalize_embeddings=True).tolist()
results = collection.query(query_embeddings=question_embedding, n_results=candidate_k)

vector_ids = results["ids"][0]
distances = results["distances"][0]

print("=== Vector search results ===")
for vid, dist in zip(vector_ids, distances):
    similarity = 1 - dist
    marker = "chunk_22" in vid
    print(f"{'>>> ' if marker else '    '}{vid} | similarity={similarity:.4f}")

# Stage 2: Keyword search
keyword_ids = asyncio.run(keyword_search(question, top_k=candidate_k))
print("\n=== Keyword search results ===")
for kid in keyword_ids:
    marker = "chunk_22" in kid
    print(f"{'>>> ' if marker else '    '}{kid}")
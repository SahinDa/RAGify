from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "The cat sat on the mat.",
    "A feline rested on the rug.",
    "Stock market crashed today."
]

embeddings = model.encode(sentences)

print("Shape of embeddings:", embeddings.shape)
print("First embedding (first 5 values):", embeddings[0][:5])


# Compare sentence 1 vs 2 (should be similar - both about cat/mat)
sim_1_2 = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

# Compare sentence 1 vs 3 (should be dissimilar - cat vs stock market)
sim_1_3 = cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]

print(f"\nSimilarity (cat/mat vs feline/rug): {sim_1_2:.4f}")
print(f"Similarity (cat/mat vs stock market): {sim_1_3:.4f}")

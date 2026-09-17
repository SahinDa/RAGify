import asyncio
from app.postgres_search import index_chunks , keyword_search

async def main():
    # --- Test 1 : Insert
    test_chunks = [
        "The Transformer uses multi-head self-attention.",
        "The model dimension d_model is 512 in the base configuration."
    ]
    test_ids = [
        "test_file.pdf#chunk_0_abc123",
        "test_file.pdf#chunk_1_def456"
    ]

    await index_chunks(test_chunks, test_ids, source_filename="test_file.pdf")
    print("Insert done — check your Neon table now.")

    # --- Test 2: Search
    results = await keyword_search("What is d_model", top_k=20)
    print("Search results (chunk IDs):",results)

if __name__ == "__main__":
    asyncio.run(main())
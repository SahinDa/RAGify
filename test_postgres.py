import asyncio
from app.postgres_search import index_chunks

async def main():
    test_chunks = [
        "The Transformer uses multi-head self-attention.",
        "The model dimension d_model is 512 in the base configuration."
    ]
    test_ids = [
        "test_file.pdf#chunk_0_abc123",
        "test_file.pdf#chunk_1_def456"
    ]

    await index_chunks(test_chunks, test_ids, source_filename="test_file.pdf")
    print("Done — check your Neon table now.")

if __name__ == "__main__":
    asyncio.run(main())
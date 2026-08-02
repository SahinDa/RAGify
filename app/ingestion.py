import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model once (expensive to load, so we do it at module level)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Persistent Chroma client - saves data to disk in chroma_db/ folder
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Create (or get existing) collection - think of this like a "table" for our chunks
collection = chroma_client.get_or_create_collection(name="ragify_docs")

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """
    Splits text into overlapping chunks based on word count.

    Args:
        text: The full document text
        chunk_size: Number of words per chunk
        overlap: Number of words to overlap between consecutive chunks

    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)

        # Stop if this chunk reached the end of the document
        if i + chunk_size >= len(words):
            break

    return chunks

def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """
    Converts a list of text chunks into embedding vectors.
    """
    embeddings = embedding_model.encode(chunks)
    return embeddings.tolist()  # Chroma expects plain lists, not numpy arrays 

def store_chunks(chunks: list[str], embeddings: list[list[float]], source_filename: str):
    """
    Stores chunks + their embeddings + metadata into ChromaDB.
    """
    ids = [f"{source_filename}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": source_filename, "chunk_index": i} for i in range(len(chunks))]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas
    )
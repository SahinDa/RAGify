import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import io

# Load embedding model once (expensive to load, so we do it at module level)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Persistent Chroma client - saves data to disk in chroma_db/ folder
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Create (or get existing) collection - think of this like a "table" for our chunks
collection = chroma_client.get_or_create_collection(
    name="ragify_docs",
    metadata={"hnsw:space": "cosine"} 
    )

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
    Deletes any existing chunks from the same filename first, to avoid duplicates.
    """
    
    # Remove old chunks from this same file, if any exist (re-upload = replace, not duplicate)
    collection.delete(where={"source": source_filename})
        
    ids = [f"{source_filename}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": source_filename, "chunk_index": i} for i in range(len(chunks))]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas
    )
    

def parse_file(filename: str, content: bytes) -> str:
    """
    Extracts plain text from uploaded file bytes, based on file extension.

    Args:
        filename: original filename (used to detect extension)
        content: raw file bytes

    Returns:
        Extracted plain text as a string
    """
    if filename.lower().endswith(".txt"):
        return content.decode("utf-8")

    elif filename.lower().endswith(".pdf"):
        pdf_file = io.BytesIO(content)  # treat bytes like a file, in memory
        reader = PdfReader(pdf_file)

        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"

        return text

    else:
        raise ValueError(f"Unsupported file type: {filename}")


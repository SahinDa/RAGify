import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import io
import hashlib
import docx
import re

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
    Splits text into overlapping chunks, breaking at sentence boundaries
    instead of cutting mid-sentence. chunk_size/overlap are in words.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks = []
    current_chunk_sentences = []
    current_word_count = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())

        if current_word_count + sentence_word_count > chunk_size and current_chunk_sentences:
            # finalize current chunk
            chunks.append(" ".join(current_chunk_sentences))

            # build overlap: carry the last few sentences forward
            overlap_sentences = []
            overlap_word_count = 0
            for s in reversed(current_chunk_sentences):
                w = len(s.split())
                if overlap_word_count + w > overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_word_count += w

            current_chunk_sentences = overlap_sentences
            current_word_count = overlap_word_count

        # this MUST be inside the outer loop — runs for every sentence
        current_chunk_sentences.append(sentence)
        current_word_count += sentence_word_count

    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    return chunks

def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """
    Converts a list of text chunks into embedding vectors.
    """
    embeddings = embedding_model.encode(chunks)
    return embeddings.tolist()  # Chroma expects plain lists, not numpy arrays 

def store_chunks(chunks: list[str], embeddings: list[list[float]], source_filename: str):
    """
    Stores chunks + embeddings + metadata into ChromaDB.
    - Deletes old chunks tied to this exact filename first (handles edited re-uploads,
      so stale/removed content doesn't linger forever).
    - Uses content-hash IDs so identical text is never duplicated,
      even if it appears under a different filename.
    """
    
    # Step 1: clear out old chunks from this filename (in case the file was edited)
    collection.delete(where={"source": source_filename})

    # 2. Generate collision-safe compound IDs
    ids = [get_chunk_id(source_filename, i, chunk) for i, chunk in enumerate(chunks)]
    metadatas = [
        {"source": source_filename, "chunk_index": i, "char_length": len(chunks[i])}
        for i in range(len(chunks))
    ]
    
    # Step 3: upsert - overwrites if identical content already exists under any ID
    collection.upsert(
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
        pdf_file = io.BytesIO(content)
        reader = PdfReader(pdf_file)

        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"

        return text

    elif filename.lower().endswith(".docx"):
        docx_file = io.BytesIO(content)
        document = docx.Document(docx_file)

        text = ""

        # Walk paragraphs and tables in document order
        for element in document.element.body:
            if element.tag.endswith("}p"):
                para = docx.text.paragraph.Paragraph(element, document)
                if para.text.strip():
                    text += para.text + "\n"

            elif element.tag.endswith("}tbl"):
                table = docx.table.Table(element, document)
                text += "\n[TABLE]\n"

                for row in table.rows:
                    row_text = " | ".join(
                        cell.text.strip() for cell in row.cells
                    )
                    text += row_text + "\n"

                text += "[/TABLE]\n"

        return text

    else:
        raise ValueError(f"Unsupported file type: {filename}")
def get_chunk_id(source_filename: str, chunk_index: int,text: str) -> str:
    """
    Generates a unique, deterministic ID tied to the file, chunk index, and content hash.
    Prevents identical text across different documents from overwriting each other.
    """
    content_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
    return f"{source_filename}#chunk_{chunk_index}_{content_hash}"

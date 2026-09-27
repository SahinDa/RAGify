import chromadb
import os
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import io
import hashlib
import docx
import re
from app.postgres_search import index_chunks
import asyncio
from dotenv import load_dotenv

load_dotenv()
# Load embedding model once (expensive to load, so we do it at module level)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

#Local Chromadb Setup
# Persistent Chroma client - saves data to disk in chroma_db/ folder
# chroma_client = chromadb.PersistentClient(path="./chroma_db")

#Remote Chromadb Setup
chroma_client = chromadb.CloudClient(
    api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE")
)

# Create (or get existing) collection - think of this like a "table" for our chunks
collection = chroma_client.get_or_create_collection(
    name="ragify_docs",
    metadata={"hnsw:space": "cosine"} 
    )

SECTION_PATTERN = re.compile(r'(\d+\.\d+\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,4})\n')

def detect_sections(text: str) -> list[str]:
    """
    Splits text into sections at detected numbered headers (e.g. "5.3 Optimizer").
    Each returned section includes its own header at the start.
    If no headers are found, returns the whole text as a single section.
    """
    matches = list(SECTION_PATTERN.finditer(text))

    if not matches:
        return [text]

    sections = []
    start = 0
    for match in matches:
        header_start = match.start()
        if header_start > start:
            sections.append(text[start:header_start])
        start = header_start

    sections.append(text[start:])

    return [s for s in sections if s.strip()]


def _chunk_section(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """
    Splits a single section of text into overlapping chunks, breaking at sentence
    boundaries instead of cutting mid-sentence. chunk_size/overlap are in words.
    (This is your original chunk_text logic, unchanged, now operating on one
    section at a time instead of the whole document.)
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
            chunks.append(" ".join(current_chunk_sentences))

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

        current_chunk_sentences.append(sentence)
        current_word_count += sentence_word_count

    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    return chunks


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
      """
      Splits text into overlapping chunks. First detects structural section
      boundaries (numbered headers like "5.3 Optimizer") so a chunk never spans
      two unrelated subsections. Within each section, applies word-count +
      sentence-boundary chunking as before. Falls back to whole-document
      chunking if no section headers are detected.
      """
      sections = detect_sections(text)
  
      all_chunks = []
      for section in sections:
          all_chunks.extend(_chunk_section(section, chunk_size, overlap))
  
      return all_chunks

def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """
    Converts a list of text chunks into normalized embedding vectors (L2 norm = 1.0).
    Ensures calibrated cosine distance calculations in ChromaDB.
    """
    embeddings = embedding_model.encode(chunks,normalize_embeddings=True)
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

    asyncio.run(index_chunks(chunks,ids,source_filename))
    

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

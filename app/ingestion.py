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
from fastapi import FastAPI, UploadFile, HTTPException
from app.ingestion import parse_file, chunk_text, embed_chunks, store_chunks

app = FastAPI(title="RAGify")


@app.post("/upload")
async def upload_file(file: UploadFile):
    # Step 1: Read raw bytes from the uploaded file
    content = await file.read()

    # Step 2: Parse text out of the file (handles .txt / .pdf)
    try:
        text = parse_file(file.filename, content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not text.strip():
        raise HTTPException(status_code=400, detail="No readable text found in file.")

    # Step 3: Chunk the text
    chunks = chunk_text(text)

    # Step 4: Embed the chunks
    embeddings = embed_chunks(chunks)

    # Step 5: Store chunks + embeddings in ChromaDB
    store_chunks(chunks, embeddings, source_filename=file.filename)

    return {
        "filename": file.filename,
        "chunks_created": len(chunks),
        "status": "success"
    }
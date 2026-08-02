from fastapi import FastAPI, UploadFile, HTTPException
from app.ingestion import parse_file, chunk_text, embed_chunks, store_chunks
import requests

from app.retrieval import retrieve_relevant_chunks
from app.llm import build_prompt, call_llm_stream
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse


app = FastAPI(title="RAGify")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")

@app.post("/upload")
async def upload_file(file: UploadFile):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")
    
    # Step 1: Read raw bytes from the uploaded file
    content = await file.read()
    
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    
    # Limit file size to 10MB to prevent server overload
    max_size = 10 * 1024 * 1024  # 10MB
    if len(content) > max_size:
        raise HTTPException(status_code=413, detail="File too large. Max size is 10MB.")
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

@app.post("/ask")
async def ask_question(question: str):
    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    chunks = retrieve_relevant_chunks(question)

    if not chunks:
        return {
            "question": question,
            "answer": "I don't have enough information in the provided documents to answer that.",
            "sources": []
        }

    messages = build_prompt(question, chunks)

    try:
        answer = "".join(call_llm_stream(messages))
    except requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"LLM provider error: {str(e)}")
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="LLM provider timed out. Please try again.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error calling LLM: {str(e)}")

    return {
        "question": question,
        "answer": answer,
        "sources": [c["metadata"] for c in chunks]
    }


@app.post("/ask/stream")
async def ask_question_stream(question: str):
    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    chunks = retrieve_relevant_chunks(question)

    if not chunks:
        def empty_response():
            yield "I don't have enough information in the provided documents to answer that."
        return StreamingResponse(empty_response(), media_type="text/plain")

    messages = build_prompt(question, chunks)

    def generate():
        try:
            for piece in call_llm_stream(messages):
                yield piece
        except Exception as e:
            yield f"\n[Error: {str(e)}]"

    return StreamingResponse(generate(), media_type="text/plain")    
import logging
from dotenv import load_dotenv

load_dotenv()

from app.logging_config import setup_logging
from app.env_check import validate_env

setup_logging()
validate_env()

from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks
from app.ingestion import parse_file, chunk_text, embed_chunks, store_chunks
import httpx
from app.retrieval import retrieve_relevant_chunks
from app.llm import build_prompt, call_llm_stream , contextualize_query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from fastapi.concurrency import run_in_threadpool
from app.memory import conversation_memory
from app.summarizer import trigger_summarization_if_needed



app = FastAPI(title="RAGify")

logger = logging.getLogger(__name__)

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
    logger.info(f"Upload received: {file.filename} ({len(content)} bytes)")
    
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    
    # Limit file size to 10MB to prevent server overload
    max_size = 10 * 1024 * 1024  # 10MB
    if len(content) > max_size:
        logger.warning(f"Upload rejected (too large): {file.filename} ({len(content)} bytes)")
        raise HTTPException(status_code=413, detail="File too large. Max size is 10MB.")
    # Step 2: Parse text out of the file (handles .txt / .pdf)
    try:
        text = parse_file(file.filename, content)
    except ValueError as e:
         logger.error(f"Parse failed for {file.filename}: {e}")
         raise HTTPException(status_code=400, detail=str(e))

    if not text.strip():
        logger.warning(f"No readable text extracted from {file.filename}")
        raise HTTPException(status_code=400, detail="No readable text found in file.")

    # Step 3: Chunk the text
    chunks = chunk_text(text)

    # Step 4: Embed the chunks
    embeddings =  await run_in_threadpool(embed_chunks,chunks)

    # Step 5: Store chunks + embeddings in ChromaDB
    await run_in_threadpool(store_chunks,chunks, embeddings, source_filename=file.filename)
    
    logger.info(f"Upload complete: {file.filename} — {len(chunks)} chunks created")
    
    return {
        "filename": file.filename,
        "chunks_created": len(chunks),
        "status": "success"
    }

@app.post("/ask")
async def ask_question(
    question: str,
    conversation_id: str,
    background_tasks: BackgroundTasks
    ):
    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    logger.info(f"Question received: {question}")

    # Fetch memory context
    context = conversation_memory.get_context(conversation_id)
    
    # Decode raw question into a standalone search query
    search_query = await contextualize_query(
        question=question,
        recent_turns=context["recent_turns"],
        summary=context["summary"]
    )
    logger.info(f"Resolved search query: '{search_query}' (Original: '{question}')")

    try:
        chunks = await run_in_threadpool(retrieve_relevant_chunks,search_query)
    except Exception as e:
        logger.error(f"Retrieval failed for question '{question}': {e}")
        raise HTTPException(status_code=500,detail=f"Error retrieving relevant chunks: {str(e)}")    

    if not chunks:
        logger.info(f"No chunks found for question: {question}")
        answer = "I don't have enough information in the provided documents to answer that."

        conversation_memory.add_turn(
        conversation_id=conversation_id,
        question=question,
        answer=answer,
        )

        return {
            "question": question,
            "answer": answer,
            "sources": []
        }

    messages = build_prompt(
        question, 
        chunks,
        summary=context["summary"],
        recent_turns=context["recent_turns"]
        )

    try:
        pieces = [piece async for piece in call_llm_stream(messages)]
        answer = "".join(pieces)
    except httpx.HTTPStatusError as e:
        logger.error(f"LLM provider error for question '{question}': {e}")
        raise HTTPException(status_code=502, detail=f"LLM provider error: {str(e)}")
    except httpx.TimeoutException:
        logger.error(f"LLM provider timed out for question: {question}")
        raise HTTPException(status_code=504, detail="LLM provider timed out. Please try again.")
    except httpx.ConnectError:
        logger.error(f"Could not reach LLM provider for question: {question}")
        raise HTTPException(status_code=503, detail="LLM provider is unreachable. Please try again.")
    except Exception as e:
        logger.error(f"Unexpected LLM error for question '{question}': {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error calling LLM: {str(e)}")
    
    logger.info(f"Answered question (answer length: {len(answer)} chars, {len(chunks)} sources)")

    conversation_memory.add_turn(
    conversation_id=conversation_id,
    question=question,
    answer=answer,
    )
    
    # Schedule background summarization
    background_tasks.add_task(trigger_summarization_if_needed, conversation_id)
    

    return {
        "question": question,
        "answer": answer,
        "sources": [c["metadata"] for c in chunks]
    }


@app.post("/ask/stream")
async def ask_question_stream(
    question: str,
    conversation_id: str,
    background_tasks: BackgroundTasks
    ):
    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    logger.info(f"Streaming question received: {question}")
    
    # Fetch memory context
    context = conversation_memory.get_context(conversation_id)
    
    # Decode raw question into a standalone search query
    search_query = await contextualize_query(
        question=question,
        recent_turns=context["recent_turns"],
        summary=context["summary"]
    )
    logger.info(f"Resolved search query: '{search_query}' (Original: '{question}')")
    
    try:
        chunks = await run_in_threadpool(retrieve_relevant_chunks,search_query)
    except Exception as e:
        logger.error(f"Retrieval failed for streaming question '{question}': {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving relevant chunks: {str(e)}")    

    if not chunks:
        logger.info(f"No relevant chunks found for streaming question: {question}")
        
        def empty_response():
            answer = "I don't have enough information in the provided documents to answer that."
            conversation_memory.add_turn(
                conversation_id=conversation_id,
                question=question,
                answer=answer,
                )
            yield answer
            
        return StreamingResponse(empty_response(), media_type="text/plain")

    messages = build_prompt(question, 
                            chunks,
                            summary=context["summary"],
                            recent_turns=context["recent_turns"]
                            )

    async def generate():
        answer_parts = []
        try:
            async for piece in call_llm_stream(messages):
                answer_parts.append(piece)
                yield piece

            answer = "".join(answer_parts)
                
            conversation_memory.add_turn(
                conversation_id=conversation_id,
                question=question,
                answer=answer,
            )
            
            # Trigger summarization in background after turn is added
            background_tasks.add_task(trigger_summarization_if_needed, conversation_id)

            logger.info(f"Finished streaming answer for: {question}")    
        except Exception as e:
            logger.error(f"Streaming error for question '{question}': {e}")
            yield f"\n[Error: {str(e)}]"

    return StreamingResponse(generate(), media_type="text/plain")   

@app.get("/memory/{conversation_id}/summary")
async def get_summary(conversation_id: str):
    """
    Returns the current rolling summary and turn metadata for a given conversation.
    """
    context = conversation_memory.get_context(conversation_id)
    return {
        "conversation_id": conversation_id,
        "summary": context["summary"],
        "turn_count": len(context["recent_turns"]),
    } 

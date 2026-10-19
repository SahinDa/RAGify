import json
import os
import httpx
import asyncio
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_URL = os.getenv("NVIDIA_URL")


def build_prompt(
    question: str,
    chunks: list[dict],
    summary: str = "",
    recent_turns: list[dict] = None,
) -> list[dict]:
    """
    Builds the messages array for the LLM:
    1. System instructions.
    2. High-level summary of older turns (if any).
    3. Recent conversation turns as native role turns.
    4. Retrieved grounding chunks + latest user question.
    """
    if recent_turns is None:
        recent_turns = []

    system_prompt = (
        "You are a helpful assistant that answers questions using ONLY the provided context. "
        "Do not use any outside knowledge. "
        "If the answer is not present in the context, respond exactly with: "
        "\"I don't have enough information in the provided documents to answer that.\" "
        "Do not guess or make up information."
    )

    messages = [{"role": "system", "content": system_prompt}]

    # Inject rolling summary of older turns
    if summary and summary.strip():
        messages.append({
            "role": "system",
            "content": f"Summary of earlier conversation context:\n{summary.strip()}"
        })

    # Inject recent back-and-forth turns
    for turn in recent_turns:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})

    # Ground latest question with retrieved context chunks
    context_text = "\n\n".join([f"[Chunk {i+1}]: {c['text']}" for i, c in enumerate(chunks)])
    user_prompt = (
        f"Context:\n{context_text}\n\n"
        f"Question: {question}"
    )
    messages.append({"role": "user", "content": user_prompt})

    return messages

async def _open_stream(client: httpx.AsyncClient, headers: dict, payload: dict):
    """
    Opens the streaming POST request, retrying on transient network errors
    (timeouts, connection failures) with exponential backoff. Does not retry
    on HTTP error responses (4xx/5xx) — those are raised immediately via
    raise_for_status() after the connection succeeds.
    """
    response = client.stream("POST", NVIDIA_URL, headers=headers, json=payload)
    return response

async def call_llm_stream(messages: list[dict], model: str = "meta/llama-3.2-11b-vision-instruct",stream:bool =True):
    """
    Streams the LLM's response piece by piece instead of waiting for the full answer.
    Yields text chunks as they arrive.
    """
    if not NVIDIA_API_KEY :
        raise RuntimeError("NVIDIA API key is missing. "
                           "Please set NVIDIA_API_KEY in your .env file."
                           )
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if stream else "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.2,
        "stream": stream,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        yielded_any = False
        for attempt in range(3):
            try:
                async with client.stream("POST", NVIDIA_URL, headers=headers, json=payload) as response:
                    response.raise_for_status()
            
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                
                        if line.startswith("data: "):
                            data_str = line[len("data: "):]
                    
                    
                            if data_str.strip() == "[DONE]":
                                break
                            
                            chunk = json.loads(data_str)
                            
                            if not chunk.get("choices"):
                                continue # skip empty/metadata only chunks
                    
                            delta = chunk["choices"][0]["delta"].get("content","")
                    
                            if delta :
                                yielded_any = True
                                yield delta
                    
                    return   
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                if yielded_any or attempt == 2:
                    logger.error(f"LLM call failed (attempt {attempt + 1}, partial output sent: {yielded_any}): {e}")
                    raise 
                    
                wait_time = 2 ** attempt
                logger.warning(f"LLM call attempt {attempt + 1} failed ({e}), retrying in {wait_time}s")
                await asyncio.sleep(wait_time)            

async def contextualize_query(
    question: str,
    recent_turns: list[dict],
    summary: str = "",
    model: str = "meta/llama-3.2-11b-vision-instruct",
) -> str:
    """
    Decodes follow-up questions containing pronouns or implicit references
    (e.g., 'its advantages', 'how does it work?') into a standalone search query
    using the last 3 conversation turns and the rolling summary.
    """
    # If there is no previous context, return the question as-is immediately
    if not recent_turns and not summary:
        return question

    # Build context window from summary + up to last 3 turns
    history_lines = []
    if summary and summary.strip():
        history_lines.append(f"Summary of earlier conversation:\n{summary.strip()}")

    for turn in recent_turns[-3:]:
        history_lines.append(f"User: {turn['question']}")
        history_lines.append(f"Assistant: {turn['answer']}")

    history_text = "\n".join(history_lines)

    system_prompt = (
        "You are a query reformulator. Your job is to resolve pronouns and implicit "
        "references in a follow-up user question based on the conversation history. "
        "Output ONLY a direct, standalone search query. "
        "Do NOT answer the question. Do NOT add preamble or quotes."
    )

    user_prompt = (
        f"Conversation History:\n{history_text}\n\n"
        f"Follow-up Question: {question}\n\n"
        f"Standalone Question:"
    )

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 128,
        "temperature": 0.0,
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(NVIDIA_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            rewritten = data["choices"][0]["message"]["content"].strip()
            return rewritten if rewritten else question
    except Exception as e:
        logger.warning(f"Query contextualization failed ({e}), falling back to original query.")
        return question
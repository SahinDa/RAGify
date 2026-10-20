import logging
import os
import httpx
import asyncio
from dotenv import load_dotenv
from app.memory import conversation_memory

load_dotenv()
logger = logging.getLogger(__name__)

ENVIRONMENT = os.getenv("ENVIRONMENT", "dev").lower()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_URL = os.getenv("NVIDIA_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

gemini_client = None
if ENVIRONMENT == "prod":
    from google import genai
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)


async def _generate_summary_prod(system_prompt: str, user_prompt: str) -> str:
    """Generates summary using Gemini 2.0 Flash in production."""
    response = await asyncio.to_thread(
        gemini_client.models.generate_content,
        model="gemini-2.0-flash",
        contents=user_prompt,
        config={
            "system_instruction": system_prompt,
            "max_output_tokens": 512,
            "temperature": 0.1,
        },
    )
    return response.text.strip() if response.text else ""


async def _generate_summary_dev(system_prompt: str, user_prompt: str) -> str:
    """Generates summary using NVIDIA API in development."""
    if not NVIDIA_API_KEY:
        raise RuntimeError("NVIDIA_API_KEY is not set for dev environment.")

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "meta/llama-3.2-11b-vision-instruct",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 512,
        "temperature": 0.1,
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.post(NVIDIA_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()


async def trigger_summarization_if_needed(conversation_id: str):
    """
    Checks if there are older conversation turns that need to be condensed
    into the running summary. Runs asynchronously via FastAPI BackgroundTasks.
    """
    pending_data = conversation_memory.get_pending_summary(conversation_id)
    turns = pending_data["turns"]

    # Nothing to summarize if no pending older turns exist
    if not turns:
        return

    old_summary = pending_data["summary"]
    version = pending_data["summary_version"]
    count = pending_data["turn_count"]

    turns_formatted = "\n".join([f"User: {t['question']}\nAssistant: {t['answer']}" for t in turns])

    system_prompt = (
        "You are an expert conversation summarizer. "
        "Create a dense, structured, and factual summary of the provided conversation turns. "
        "Preserve core entities, technical details, user goals, decisions, and constraints. "
        "Do not include conversational filler. Keep it concise."
    )

    user_prompt = (
        f"Existing Summary:\n{old_summary or '(None)'}\n\n"
        f"New Conversation Turns to Integrate:\n{turns_formatted}\n\n"
        f"Updated Consolidated Summary:"
    )

    try:
        if ENVIRONMENT == "prod":
            new_summary = await _generate_summary_prod(system_prompt, user_prompt)
        else:
            new_summary = await _generate_summary_dev(system_prompt, user_prompt)

        if new_summary:
            conversation_memory.commit_summary(
                conversation_id=conversation_id,
                summary=new_summary,
                summarized_turns=count,
                expected_version=version,
            )
            logger.info(f"Summarization committed for conversation {conversation_id} (version {version + 1})")
    except Exception as e:
        logger.error(f"Background summarization failed for conversation {conversation_id}: {e}")
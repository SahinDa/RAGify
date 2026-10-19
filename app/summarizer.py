import logging
import os
import httpx
from app.memory import conversation_memory
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_URL = os.getenv("NVIDIA_URL")

logger = logging.getLogger(__name__)

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

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(NVIDIA_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            new_summary = data["choices"][0]["message"]["content"].strip()

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
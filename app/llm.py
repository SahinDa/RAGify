import os
import logging
from typing import AsyncGenerator, Dict, List, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
logger = logging.getLogger(__name__)

# --- Environment Detection ---
ENVIRONMENT = os.getenv("ENVIRONMENT", "dev").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL") 
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing. Please set it in your environment configuration.")

# Initialize client
client = genai.Client(api_key=GEMINI_API_KEY)

# --- Environment-Specific Configurations ---
ENV_CONFIGS: Dict[str, dict] = {
    "dev": {
        "model": GEMINI_MODEL or DEFAULT_GEMINI_MODEL,
        "temperature": 0.2,
        "max_output_tokens": 512,
        "log_level": logging.DEBUG,
    },
    "prod": {
        "model": GEMINI_MODEL or DEFAULT_GEMINI_MODEL,
        "temperature": 0.1,
        "max_output_tokens": 1024,
        "log_level": logging.INFO,
    },
}

ACTIVE_CFG = ENV_CONFIGS.get(ENVIRONMENT, ENV_CONFIGS["dev"])
logger.setLevel(ACTIVE_CFG["log_level"])
logger.info(f"Initialized LLM pipeline in '{ENVIRONMENT}' mode (model: {ACTIVE_CFG['model']})")


def build_prompt(
    question: str,
    chunks: list[dict],
    summary: str = "",
    recent_turns: list[dict] = None,
) -> list[dict]:
    """Builds prompt messages with system instructions and document context."""
    if recent_turns is None:
        recent_turns = []

    system_prompt = (
        "You are a helpful assistant that answers questions using ONLY the provided context. "
        "Do not use any outside knowledge. "
        "If the answer is not present in the context, respond exactly with: "
        '"I don\'t have enough information in the provided documents to answer that." '
        "Do not guess or make up information."
    )

    messages = [{"role": "system", "content": system_prompt}]

    if summary and summary.strip():
        messages.append({
            "role": "system",
            "content": f"Summary of earlier conversation context:\n{summary.strip()}"
        })

    for turn in recent_turns:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})

    context_text = "\n\n".join([f"[Chunk {i+1}]: {c['text']}" for i, c in enumerate(chunks)])
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {question}"
    messages.append({"role": "user", "content": user_prompt})

    return messages


def _parse_messages_for_gemini(messages: list[dict]):
    """Converts common messages format to Gemini system instruction and content parts."""
    system_instruction = ""
    contents: list[types.Content] = []

    for msg in messages:
        if msg["role"] == "system":
            system_instruction += (msg["content"] + "\n\n")
        else:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg["content"])]
                )
            )

    return system_instruction.strip() or None, contents


async def _stream_dev(messages: list[dict], model: str) -> AsyncGenerator[str, None]:
    """Dev pipeline: standard streaming with detailed execution logging."""
    system_instruction, contents = _parse_messages_for_gemini(messages)
    
    config = types.GenerateContentConfig(
        temperature=ACTIVE_CFG["temperature"],
        max_output_tokens=ACTIVE_CFG["max_output_tokens"],
        system_instruction=system_instruction,
    )

    logger.debug(f"[DEV] Dispatching stream request to {model}")
    response_stream = await client.aio.models.generate_content_stream(
        model=model,
        contents=contents,
        config=config,
    )
    async for chunk in response_stream:
        if chunk.text:
            yield chunk.text


async def _stream_prod(messages: list[dict], model: str) -> AsyncGenerator[str, None]:
    """Prod pipeline: streaming with error handling and strict boundary logging."""
    system_instruction, contents = _parse_messages_for_gemini(messages)
    
    config = types.GenerateContentConfig(
        temperature=ACTIVE_CFG["temperature"],
        max_output_tokens=ACTIVE_CFG["max_output_tokens"],
        system_instruction=system_instruction,
    )

    try:
        response_stream = await client.aio.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config,
        )
        async for chunk in response_stream:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        logger.error(f"[PROD] Gemini streaming failure: {e}", exc_info=True)
        raise


async def call_llm_stream(
    messages: list[dict],
    model: Optional[str] = None,
    stream: bool = True
) -> AsyncGenerator[str, None]:
    """Routes stream generation through the designated environment pipeline."""
    target_model = model or ACTIVE_CFG["model"]

    if ENVIRONMENT == "prod":
        async for chunk in _stream_prod(messages, model=target_model):
            yield chunk
    else:
        async for chunk in _stream_dev(messages, model=target_model):
            yield chunk


async def contextualize_query(
    question: str,
    recent_turns: list[dict],
    summary: str = "",
    model: Optional[str] = None,
) -> str:
    """Reformulates follow-up queries using the active environment configuration."""
    if not recent_turns and not summary:
        return question

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
    user_prompt = f"Conversation History:\n{history_text}\n\nFollow-up Question: {question}\n\nStandalone Question:"

    target_model = model or ACTIVE_CFG["model"]

    try:
        response = await client.aio.models.generate_content(
            model=target_model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=128,
                temperature=0.0,
            ),
        )
        return response.text.strip() if response.text else question
    except Exception as e:
        logger.warning(f"[{ENVIRONMENT.upper()}] Contextualization fallback triggered: {e}")
        return question
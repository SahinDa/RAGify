import json
import os
import requests
from dotenv import load_dotenv

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


def build_prompt(question: str, chunks: list[dict]) -> list[dict]:
    """
    Builds the messages list for the LLM, injecting retrieved chunks as context.
    """
    context_text = "\n\n".join([f"[Chunk {i+1}]: {c['text']}" for i, c in enumerate(chunks)])

    system_prompt = (
        "You are a helpful assistant that answers questions using ONLY the provided context. "
        "Do not use any outside knowledge. "
        "If the answer is not present in the context, respond exactly with: "
        "\"I don't have enough information in the provided documents to answer that.\" "
        "Do not guess or make up information."
    )

    user_prompt = f"Context:\n{context_text}\n\nQuestion: {question}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def call_llm_stream(messages: list[dict], model: str = "meta/llama-3.1-8b-instruct"):
    """
    Streams the LLM's response piece by piece instead of waiting for the full answer.
    Yields text chunks as they arrive.
    """
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.2,
        "stream": True
    }

    with requests.post(NVIDIA_URL, headers=headers, json=payload, stream=True, timeout=30) as response:
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue

            decoded_line = line.decode("utf-8")

            if decoded_line.startswith("data: "):
                data_str = decoded_line[len("data: "):]

                if data_str.strip() == "[DONE]":
                    break

                chunk = json.loads(data_str)

                if not chunk.get("choices"):
                    continue  # skip empty/metadata-only chunks

                delta = chunk["choices"][0]["delta"].get("content", "")

                if delta:
                    yield delta
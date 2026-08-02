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


def call_llm(messages: list[dict], model: str = "meta/llama-3.1-8b-instruct") -> str:
    """
    Sends messages to NVIDIA NIM and returns the assistant's reply text.
    """
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.2  # low temperature = more focused, less "creative" answers
    }

    response = requests.post(NVIDIA_URL, headers=headers, json=payload)
    response.raise_for_status()  # raises an error if the request failed

    data = response.json()
    return data["choices"][0]["message"]["content"]
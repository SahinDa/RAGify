import os
import json
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

ENVIRONMENT = os.getenv("ENVIRONMENT", "dev").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# ==========================================
# 1. EMBEDDER IMPLEMENTATIONS
# ==========================================
class DevEmbedder:
    """Local SentenceTransformer (384 dimensions)."""
    def __init__(self):
        logger.info("[DEV] Initializing local SentenceTransformer ('all-MiniLM-L6-v2')...")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.model.encode([query], normalize_embeddings=True).tolist()[0]


class GeminiEmbedder:
    """Production Google Gemini API (text-embedding-004, 768 dimensions)."""
    def __init__(self):
        self.model_name = "gemini-embedding-001"
        logger.info("[PROD] Initializing Gemini API Embedder ('{self.model_name}')...")
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set.")
        from google import genai
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        all_embeddings = []
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=batch,
            )
            all_embeddings.extend([e.values for e in response.embeddings])
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=query,
        )
        return response.embeddings[0].values


# ==========================================
# 2. RERANKER IMPLEMENTATIONS
# ==========================================
class DevReranker:
    """Local CrossEncoder for Dev."""
    def __init__(self):
        logger.info("[DEV] Initializing local CrossEncoder ('cross-encoder/ms-marco-MiniLM-L-6-v2')...")
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        if not candidates:
            return []
        pairs = [[query, c["text"]] for c in candidates]
        scores = self.model.predict(pairs)
        for c, score in zip(candidates, scores):
            c["score"] = float(score)
        candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates[:top_k]


class GeminiReranker:
    """Listwise Reranking using Gemini 2.0 Flash (Zero extra RAM)."""
    def __init__(self):
        logger.info("[PROD] Initializing Gemini API Reranker ('gemini-2.0-flash')...")
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set.")
        from google import genai
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        if not candidates:
            return []

        passages_text = "\n\n".join(
            [f"[{i}]: {c['text'][:400]}" for i, c in enumerate(candidates)]
        )
        prompt = (
            f"You are an expert passage reranker. Rank the following passages in order of relevance "
            f"to the user query. Output ONLY a valid JSON list of integer indices from most relevant to least.\n\n"
            f"Query: {query}\n\n"
            f"Passages:\n{passages_text}\n\n"
            f"JSON Array of indices:"
        )

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )
            ranked_indices = json.loads(response.text)
            ranked = []
            seen = set()
            for idx in ranked_indices:
                if isinstance(idx, int) and 0 <= idx < len(candidates) and idx not in seen:
                    ranked.append(candidates[idx])
                    seen.add(idx)

            # Preserve unranked items in original RRF order
            for i, c in enumerate(candidates):
                if i not in seen:
                    ranked.append(c)

            return ranked[:top_k]
        except Exception as e:
            logger.warning(f"Gemini API reranking failed ({e}), using RRF order fallback.")
            candidates.sort(key=lambda c: c.get("score", 0.0), reverse=True)
            return candidates[:top_k]


# ==========================================
# 3. SINGLETON INSTANTIATION
# ==========================================
if ENVIRONMENT == "prod":
    embedder = GeminiEmbedder()
    reranker = GeminiReranker()
else:
    embedder = DevEmbedder()
    reranker = DevReranker()
"""
HU-044 — Embeddings con Google Gemini gemini-embedding-001
"""

import logging
import time
from typing import List

import google.generativeai as genai

from src.config.settings import get_settings

logger = logging.getLogger("agrovision-rag")


class GeminiEmbeddings:
    def __init__(self):
        self.settings = get_settings()
        self._configured = False
        self.model = self.settings.EMBEDDING_MODEL
        logger.info("GeminiEmbeddings inicializado con modelo: %s", self.model)

    def _ensure_configured(self):
        if not self._configured:
            genai.configure(api_key=self.settings.GOOGLE_API_KEY)
            self._configured = True

    def _embed_with_retry(self, text: str, task_type: str, max_retries: int = 3) -> List[float]:
        """Genera embedding con reintentos automáticos en caso de rate limit."""
        for attempt in range(max_retries):
            try:
                result = genai.embed_content(
                    model=self.model,
                    content=text[:2048],
                    task_type=task_type,
                )
                return result["embedding"]
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "quota" in error_str.lower():
                    wait_time = 15 * (attempt + 1)
                    logger.warning(
                        "Rate limit alcanzado. Esperando %ds (intento %d/%d)...",
                        wait_time, attempt + 1, max_retries,
                    )
                    time.sleep(wait_time)
                else:
                    logger.error("Error embedding: %s", e)
                    return [0.0] * self.settings.EMBEDDING_DIMENSIONS
        logger.error("Max reintentos alcanzados para embedding")
        return [0.0] * self.settings.EMBEDDING_DIMENSIONS

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Genera embeddings con rate limiting (máx 80 requests/min)."""
        self._ensure_configured()
        embeddings = []
        total = len(texts)
        logger.info("Generando embeddings para %d chunks...", total)

        for i, text in enumerate(texts):
            embedding = self._embed_with_retry(text, "retrieval_document")
            embeddings.append(embedding)

            # Delay para respetar rate limit (80 req/min = 0.75s entre requests)
            time.sleep(0.75)

            if (i + 1) % 10 == 0:
                logger.info("Embeddings: %d/%d", i + 1, total)

        logger.info("Embeddings completados: %d", len(embeddings))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        self._ensure_configured()
        return self._embed_with_retry(text, "retrieval_query")
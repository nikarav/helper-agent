from typing import Any

import numpy as np
from google import genai
from google.genai import types
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from helper_agent.utilities.logger import get_logger
from helper_agent.utilities.rate_limit import RateLimiter

logger = get_logger("helper_agent")


def _l2_normalize(embeddings: list[list[float]]) -> list[list[float]]:
    """
    L2 normalize embeddings (required for truncated Gemini dimensions).

    :param embeddings: List of embedding vectors
    :return: Normalized embeddings
    """
    arr = np.array(embeddings)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return (arr / norms).tolist()


class GeminiEmbedder:
    """Gemini embedding client with rate limiting and batching."""

    def __init__(
        self,
        model: str = "gemini-embedding-001",
        dimension: int = 512,
        task_type: str = "RETRIEVAL_DOCUMENT",
        rpm: int = 60,
        tpm: int = 100000,
        batch_size: int = 20,
    ) -> None:
        """
        Initialize the Gemini embedder.

        :param model: Embedding model name
        :param dimension: Output embedding dimension (512, 768, 1536, or 3072)
        :param task_type: Task type for embeddings (RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, etc.)
        :param rpm: Requests per minute limit
        :param tpm: Tokens per minute limit
        :param batch_size: Maximum texts per API call
        """
        self.model = model
        self.dimension = dimension
        self.task_type = task_type
        self.batch_size = batch_size

        self._client = genai.Client()
        self._rate_limiter = RateLimiter(rpm=rpm, tpm=tpm)

    def _estimate_tokens(self, texts: list[str]) -> int:
        """
        Rough token estimate: ~4 chars per token + 10% buffer.

        :param texts: List of texts
        :return: Estimated token count
        """
        return int(sum(len(t) for t in texts) * 1.1 / 4)

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
    )
    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """
        Call Gemini API with tenacity retry.

        :param texts: List of texts
        :return: List of embedding vectors
        """
        result = self._client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=self.task_type,
                output_dimensionality=self.dimension,
            ),
        )
        return [e.values for e in result.embeddings]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch with rate limiting.

        :param texts: List of texts
        :return: List of embedding vectors
        """
        if not texts:
            return []

        estimated_tokens = self._estimate_tokens(texts)
        self._rate_limiter.wait_if_needed(estimated_tokens)

        embeddings = self._call_api(texts)

        # L2 normalize for truncated dimensions
        if self.dimension < 3072:
            embeddings = _l2_normalize(embeddings)

        self._rate_limiter.record_request(estimated_tokens)
        return embeddings

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts with automatic batching.

        :param texts: List of texts
        :return: List of embedding vectors
        """
        if not texts:
            return []

        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            all_embeddings.extend(self._embed_batch(batch))
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a query (uses RETRIEVAL_QUERY task type).

        :param query: Query to embed
        :return: Embedding vector
        """
        original = self.task_type
        self.task_type = "RETRIEVAL_QUERY"
        try:
            result = self._embed_batch([query])
            return result[0] if result else []
        finally:
            self.task_type = original

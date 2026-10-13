from typing import List

import numpy as np
from openai import OpenAI

from src.config import settings


class EmbeddingClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
        )
        self.model = settings.embedding_model
        self.dim = settings.embedding_dim

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        norm = np.clip(norm, 1e-12, None)
        return x / norm

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors = []

        # text-embedding-v4 官方单 batch 最多 10 条
        for start in range(0, len(texts), 10):
            batch = texts[start:start + 10]

            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
                dimensions=self.dim,
                encoding_format="float",
            )

            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(item.embedding for item in ordered)

        arr = np.asarray(vectors, dtype=np.float32)
        return self._normalize(arr)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])
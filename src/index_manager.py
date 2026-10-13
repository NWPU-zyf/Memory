import threading
from typing import Dict, List, Tuple

import faiss
import numpy as np

from src.store import MemoryStore


class UserIndex:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.ids: List[str] = []


class IndexManager:
    def __init__(self, store: MemoryStore, dim: int):
        self.store = store
        self.dim = dim
        self.indices: Dict[str, UserIndex] = {}
        self.lock = threading.RLock()

    def _load_user(self, user_id: str):
        records = self.store.get_user_memories(user_id)

        user_index = UserIndex(self.dim)

        if records:
            vectors = np.stack(
                [r["embedding"] for r in records]
            ).astype(np.float32)

            user_index.index.add(vectors)
            user_index.ids = [r["id"] for r in records]

        self.indices[user_id] = user_index

    def ensure_user(self, user_id: str):
        with self.lock:
            if user_id not in self.indices:
                self._load_user(user_id)

    def add(self, user_id: str, memory_id: str, embedding: np.ndarray):
        with self.lock:
            if user_id not in self.indices:
                return

            vector = np.asarray(
                embedding,
                dtype=np.float32,
            ).reshape(1, -1)

            self.indices[user_id].index.add(vector)
            self.indices[user_id].ids.append(memory_id)

    def search(
        self,
        user_id: str,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> List[Tuple[str, float]]:
        self.ensure_user(user_id)

        with self.lock:
            user_index = self.indices[user_id]

            if not user_index.ids:
                return []

            k = min(top_k, len(user_index.ids))

            scores, indices = user_index.index.search(
                query_embedding.astype(np.float32),
                k,
            )

            results = []

            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue

                results.append(
                    (
                        user_index.ids[int(idx)],
                        float(score),
                    )
                )

            return results
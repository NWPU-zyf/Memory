import threading
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import settings


INSTRUCTION = (
    "Given a personal memory question, determine whether the memory "
    "document contains evidence useful for answering the question."
)


class QwenReranker:
    def __init__(self):
        self.device = settings.reranker_device
        self.max_length = settings.reranker_max_length
        self.batch_size = settings.reranker_batch_size
        self.lock = threading.Lock()

        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.reranker_model,
            padding_side="left",
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            settings.reranker_model,
            dtype=torch.bfloat16,
        ).to(self.device).eval()

        self.false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.true_id = self.tokenizer.convert_tokens_to_ids("yes")

        self.prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based "
            "on the Query and the Instruct provided. Note that the "
            'answer can only be "yes" or "no".'
            "<|im_end|>\n"
            "<|im_start|>user\n"
        )

        self.suffix = (
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
        )

        self.prefix_ids = self.tokenizer.encode(
            self.prefix,
            add_special_tokens=False,
        )

        self.suffix_ids = self.tokenizer.encode(
            self.suffix,
            add_special_tokens=False,
        )

    def _score_batch(self, query: str, docs: List[str]) -> List[float]:
        texts = [
            (
                f"<Instruct>: {INSTRUCTION}\n"
                f"<Query>: {query}\n"
                f"<Document>: {doc}"
            )
            for doc in docs
        ]

        body_max = (
            self.max_length
            - len(self.prefix_ids)
            - len(self.suffix_ids)
        )

        encoded = self.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=body_max,
        )

        input_ids = [
            self.prefix_ids + ids + self.suffix_ids
            for ids in encoded["input_ids"]
        ]

        batch = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            return_tensors="pt",
        )

        batch = {
            k: v.to(self.device)
            for k, v in batch.items()
        }

        with torch.inference_mode():
            logits = self.model(**batch).logits[:, -1, :]

        yes_logits = logits[:, self.true_id]
        no_logits = logits[:, self.false_id]

        scores = torch.softmax(
            torch.stack([no_logits, yes_logits], dim=1).float(),
            dim=1,
        )[:, 1]

        return scores.cpu().tolist()

    def rerank(self, query: str, candidates: List[dict]) -> List[dict]:
        scored = []

        # 单 GPU 避免多个 Search 同时挤爆显存
        with self.lock:
            for start in range(0, len(candidates), self.batch_size):
                batch = candidates[start:start + self.batch_size]

                scores = self._score_batch(
                    query,
                    [x["retrieval_text"] for x in batch],
                )

                for item, score in zip(batch, scores):
                    result = item.copy()
                    result["rerank_score"] = float(score)
                    scored.append(result)

        scored.sort(
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        return scored
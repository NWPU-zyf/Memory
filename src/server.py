import hashlib
from typing import Dict

from fastapi import Depends, FastAPI, Header, HTTPException

from src.config import settings
from src.embedding_client import EmbeddingClient
from src.index_manager import IndexManager
from src.memory_llm import MemoryLLM
from src.reranker import QwenReranker
from src.schemas import AddRequest, SearchRequest
from src.store import MemoryStore


app = FastAPI(title="AgentMemory Competition v1")

store = MemoryStore(settings.db_path)
embedder = EmbeddingClient()
memory_llm = MemoryLLM()
index_manager = IndexManager(
    store=store,
    dim=settings.embedding_dim,
)
reranker = QwenReranker()


def verify_auth(
    x_api_key: str = Header(default=""),
):
    if not settings.memory_system_key:
        return

    if x_api_key != settings.memory_system_key:
        raise HTTPException(
            status_code=401,
            detail={"reason": "invalid API key"},
        )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "competition-v1",
    }


@app.post("/add")
def add_memory(
    req: AddRequest,
    _: None = Depends(verify_auth),
):
    if not req.messages:
        raise HTTPException(
            status_code=400,
            detail={"reason": "messages cannot be empty"},
        )

    # 官方重试会保持 request_id 不变，因此必须幂等
    if store.request_exists(req.request_id):
        return {
            "success": True,
            "request_id": req.request_id,
            "user_id": req.user_id,
            "session_id": req.session_id,
        }

    records = []

    try:
        for ordinal, message in enumerate(req.messages):
            structured = memory_llm.extract(
                content=message.content,
                role=message.role,
                timestamp=message.timestamp,
            )

            retrieval_text = memory_llm.build_retrieval_text(
                structured
            )

            if not retrieval_text.strip():
                raise ValueError(
                    f"empty memory after extraction at message {ordinal}"
                )

            embedding = embedder.embed(
                [retrieval_text]
            )[0]

            raw_id = (
                f"{req.request_id}:"
                f"{ordinal}"
            )

            memory_id = (
                "mem_"
                + hashlib.sha256(
                    raw_id.encode("utf-8")
                ).hexdigest()[:24]
            )

            # 多模态赛道 Search content 用 ContentPart[]
            search_content = [
                {
                    "type": "text",
                    "text": retrieval_text,
                }
            ]

            records.append(
                {
                    "id": memory_id,
                    "request_id": req.request_id,
                    "ordinal": ordinal,
                    "user_id": req.user_id,
                    "session_id": req.session_id,
                    "role": message.role,
                    "timestamp": message.timestamp,
                    "content": search_content,
                    "retrieval_text": retrieval_text,
                    "embedding": embedding,
                    "created_at": store.now_iso(),
                }
            )

        # 所有 LLM / embedding 成功后一次事务写入
        store.add_many(records)

        # 若当前用户已有内存 index，增量加入
        for record in records:
            index_manager.add(
                req.user_id,
                record["id"],
                record["embedding"],
            )

        return {
            "success": True,
            "request_id": req.request_id,
            "user_id": req.user_id,
            "session_id": req.session_id,
        }

    except Exception as exc:
        print("[ADD ERROR]", repr(exc))
        raise HTTPException(
            status_code=500,
            detail={"reason": str(exc)},
        )


@app.post("/search")
def search_memory(
    req: SearchRequest,
    _: None = Depends(verify_auth),
):
    if req.top_k <= 0:
        raise HTTPException(
            status_code=400,
            detail={"reason": "top_k must be > 0"},
        )

    try:
        query_text = memory_llm.query_to_text(
            req.query,
            req.options,
        )

        query_embedding = embedder.embed_query(
            query_text
        )

        # 正式评测 top_k=100。
        # 即使 smoke top_k 较小，也先取最多 100 个供 reranker。
        candidate_k = max(req.top_k, 100)

        dense_results = index_manager.search(
            user_id=req.user_id,
            query_embedding=query_embedding,
            top_k=candidate_k,
        )

        if not dense_results:
            return {"data": []}

        memories = store.get_user_memories(req.user_id)
        memory_map: Dict[str, dict] = {
            x["id"]: x
            for x in memories
        }

        candidates = []

        for memory_id, dense_score in dense_results:
            memory = memory_map[memory_id]

            candidates.append(
                {
                    "id": memory_id,
                    "content": memory["content"],
                    "retrieval_text": memory["retrieval_text"],
                    "created_at": memory["created_at"],
                    "dense_score": dense_score,
                }
            )

        reranked = reranker.rerank(
            query=query_text,
            candidates=candidates,
        )

        data = []

        for item in reranked[:req.top_k]:
            data.append(
                {
                    "id": item["id"],
                    "content": item["content"],
                    "score": item["rerank_score"],
                    "created_at": item["created_at"],
                }
            )

        return {"data": data}

    except Exception as exc:
        print("[SEARCH ERROR]", repr(exc))

        raise HTTPException(
            status_code=500,
            detail={"reason": str(exc)},
        )
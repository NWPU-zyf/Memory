import os


class Settings:
    db_path = os.getenv(
        "MEMORY_DB",
        "/mnt/sdc/zhaoyifan/AgentMemory/competition_v1/data/competition.db",
    )

    dashscope_api_key = os.environ["DASHSCOPE_API_KEY"]
    dashscope_base_url = os.environ["DASHSCOPE_BASE_URL"]
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
    embedding_dim = int(os.getenv("EMBEDDING_DIM", "1024"))

    openai_api_key = os.environ["OPENAI_API_KEY"]
    openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    memory_llm_model = os.getenv("MEMORY_LLM_MODEL", "gpt-4o-mini")

    reranker_model = os.getenv(
        "RERANKER_MODEL",
        "/mnt/sdc/zhaoyifan/AgentMemory/models/Qwen3-Reranker-4B",
    )
    reranker_device = os.getenv("RERANKER_DEVICE", "cuda:0")
    reranker_batch_size = int(os.getenv("RERANKER_BATCH_SIZE", "4"))
    reranker_max_length = int(os.getenv("RERANKER_MAX_LENGTH", "8192"))

    memory_system_key = os.getenv("MEMORY_SYSTEM_KEY", "")


settings = Settings()
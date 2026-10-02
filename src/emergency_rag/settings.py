"""进程共享的模型配置；普通参数在此维护，密钥从环境或 .env 读取。"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


@dataclass(slots=True)
class Settings:
    query_cache_root: Path = Path("data/cache")  # 查询处理及查询向量固定使用的持久化目录。
    
    # 填写实际服务的模型名与地址；也可通过同名环境配置覆盖。
    embedding_model: str = "qwen3-embedding-4b"
    embedding_base_url: str | None = None
    embedding_timeout: float = 60
    embedding_max_retries: int = 2
    embedding_api_key: str = field(default="", init=False, repr=False)

    qwen_reranker_model: str = "qwen3-reranker-4b"
    qwen_reranker_base_url: str | None = None  # 缺省时复用 Embedding 网关。
    qwen_reranker_timeout: float = 60
    qwen_reranker_max_retries: int = 2
    qwen_reranker_api_key: str = field(default="", init=False, repr=False)

    chat_model: str = ""
    chat_base_url: str | None = None
    chat_timeout: float = 60
    chat_max_retries: int = 2
    chat_api_key: str = field(default="", init=False, repr=False)

    typesafe_model: str = "jev-1.13.0"
    typesafe_base_url: str = "https://api.typesafe.ai"
    typesafe_timeout: float = 60
    typesafe_max_retries: int = 2
    typesafe_api_key: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        # 只在配置初始化边界加载；已有环境变量优先，组件不再各自查找文件。
        load_dotenv(find_dotenv(usecwd=True), override=False, encoding="utf-8-sig")

        self.embedding_api_key = os.getenv("RAG_EMBEDDING_API_KEY", "")
        self.embedding_model = os.getenv("RAG_EMBEDDING_MODEL", self.embedding_model)
        self.embedding_base_url = os.getenv("RAG_EMBEDDING_BASE_URL", self.embedding_base_url)
        self.embedding_timeout = float(os.getenv("RAG_EMBEDDING_TIMEOUT", str(self.embedding_timeout)))
        self.embedding_max_retries = int(os.getenv("RAG_EMBEDDING_MAX_RETRIES", str(self.embedding_max_retries)))

        self.qwen_reranker_api_key = os.getenv("QWEN_RERANKER_API_KEY", "")
        self.qwen_reranker_model = os.getenv("QWEN_RERANKER_MODEL", self.qwen_reranker_model)
        self.qwen_reranker_base_url = os.getenv("QWEN_RERANKER_BASE_URL", self.qwen_reranker_base_url)
        self.qwen_reranker_timeout = float(os.getenv("QWEN_RERANKER_TIMEOUT", str(self.qwen_reranker_timeout)))
        self.qwen_reranker_max_retries = int(os.getenv("QWEN_RERANKER_MAX_RETRIES", str(self.qwen_reranker_max_retries)))

        self.chat_api_key = os.getenv("RAG_CHAT_API_KEY", "")
        self.chat_model = os.getenv("RAG_CHAT_MODEL", self.chat_model)
        self.chat_base_url = os.getenv("RAG_CHAT_BASE_URL", self.chat_base_url)
        self.chat_timeout = float(os.getenv("RAG_CHAT_TIMEOUT", str(self.chat_timeout)))
        self.chat_max_retries = int(os.getenv("RAG_CHAT_MAX_RETRIES", str(self.chat_max_retries)))

        self.typesafe_api_key = os.getenv("TYPESAFE_API_KEY", "")
        self.typesafe_model = os.getenv("TYPESAFE_MODEL", self.typesafe_model)
        self.typesafe_base_url = os.getenv("TYPESAFE_BASE_URL", self.typesafe_base_url)
        self.typesafe_timeout = float(os.getenv("TYPESAFE_TIMEOUT", str(self.typesafe_timeout)))
        self.typesafe_max_retries = int(os.getenv("TYPESAFE_MAX_RETRIES", str(self.typesafe_max_retries)))


# 导入时读取一次配置，不创建 SDK 连接；模型仍由各模块按需创建并缓存。
settings = Settings()

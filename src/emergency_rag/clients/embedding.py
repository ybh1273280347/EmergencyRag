"""Embedding API 边界：批量、顺序恢复及响应校验，不拥有索引构建。"""

import os
from functools import cache

import numpy as np
from openai import APIError, OpenAI


class EmbeddingError(RuntimeError):
    pass


class EmbeddingClient:
    def __init__(self, client: OpenAI, *, model: str, batch_size: int = 64) -> None:
        if not model or not model.strip():
            raise ValueError("Embedding 模型名未配置")
        if batch_size <= 0:
            raise ValueError("Embedding batch_size 必须为正整数")

        self.model = model
        self.batch_size = batch_size
        self.client = client

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]  # 单条文本复用批量接口

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)  # 空输入返回空矩阵

        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Embedding 输入必须为非空文本")

        batches: list[np.ndarray] = []
        dimension: int | None = None

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]

            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    encoding_format="float",
                )
            except APIError as exc:
                raise EmbeddingError(
                    f"Embedding API 调用失败：{type(exc).__name__}"
                ) from exc

            # 兼容 OpenAI-compatible 服务乱序返回；index 相对当前批次
            if not isinstance(response.data, list):
                raise EmbeddingError("Embedding 响应缺少 data 数组")

            indexes = [item.index for item in response.data]
            if (
                any(type(index) is not int for index in indexes)
                or sorted(indexes) != list(range(len(batch)))
            ):
                raise EmbeddingError("Embedding 响应 index 缺失、重复或越界")

            try:
                vectors = np.asarray(
                    [
                        item.embedding
                        for item in sorted(response.data, key=lambda item: item.index)
                    ],
                    dtype=np.float32,
                )
            except (TypeError, ValueError) as exc:
                raise EmbeddingError("Embedding 响应向量格式或维度无效") from exc

            if (
                vectors.ndim != 2
                or vectors.shape[1] == 0
                or not np.isfinite(vectors).all()
            ):
                raise EmbeddingError("Embedding 响应必须为有限数值的二维向量")

            if dimension is not None and vectors.shape[1] != dimension:
                raise EmbeddingError("Embedding 响应在不同批次间维度不一致")

            dimension = vectors.shape[1]
            batches.append(vectors)

        return np.ascontiguousarray(
            np.concatenate(batches),
            dtype=np.float32,
        )


@cache
def get_embedding_client() -> EmbeddingClient:
    """首次使用时创建进程共享的模型实例，离线与在线使用同一配置。"""
    api_key = os.environ.get("RAG_EMBEDDING_API_KEY")
    model = os.environ.get("RAG_EMBEDDING_MODEL")

    if not api_key or not model:
        raise ValueError("请设置 RAG_EMBEDDING_API_KEY 和 RAG_EMBEDDING_MODEL")

    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get("RAG_EMBEDDING_BASE_URL"),
        timeout=float(os.environ.get("RAG_EMBEDDING_TIMEOUT", "60")),
        max_retries=int(os.environ.get("RAG_EMBEDDING_MAX_RETRIES", "2")),
    )
    return EmbeddingClient(client, model=model)
"""Embedding API 边界：批量、顺序恢复及响应校验，不拥有索引构建。"""

import re
from functools import cache
from pathlib import Path

import numpy as np
from openai import APIError, OpenAI

from emergency_rag.cache import JsonFileCache
from emergency_rag.settings import settings


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
        self._query_caches: dict[Path, JsonFileCache] = {}  # 按缓存路径复用实例

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]  # 单条文本复用批量接口

    def embed_query(
        self,
        text: str,
        *,
        cache_directory: Path | None = None,
        refresh: bool = False,
    ) -> np.ndarray:
        """仅缓存在线查询向量；离线文档仍使用 embed_batch，不混用缓存。"""
        if cache_directory is None:
            cache_directory = settings.query_cache_root / "query_embeddings"

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Embedding 输入必须为非空文本")

        # 模型名转义为安全的目录名，避免非法字符
        model_directory = re.sub(r"[^A-Za-z0-9._-]+", "-", self.model).strip("-.")
        if not model_directory:
            raise EmbeddingError("Embedding 模型名不能生成有效缓存目录")

        path = (Path(cache_directory) / model_directory / "embeddings.json").resolve()
        if path not in self._query_caches:
            self._query_caches[path] = JsonFileCache(path)
        cache = self._query_caches[path]

        # 缓存已存在时必须与当前模型一致
        if cache.data and cache.data.get("model") != self.model:
            raise EmbeddingError("查询向量缓存的模型与当前模型不一致，请选择新的缓存目录")

        queries = cache.data.get("queries", {})
        if not isinstance(queries, dict):
            raise EmbeddingError("查询向量缓存缺少有效 queries 对象")

        hit = text in queries and not refresh
        try:
            vector = (
                np.asarray(queries[text], dtype=np.float32)
                if hit
                else self.embed(text)
            )
        except (TypeError, ValueError) as exc:
            raise EmbeddingError("查询向量缓存格式无效") from exc

        # 查询向量必须为一维、非零、有限
        if (
            vector.ndim != 1
            or not vector.size
            or not np.isfinite(vector).all()
            or not np.any(vector)
        ):
            raise EmbeddingError("查询向量必须为非零且有限的一维向量")

        # 已存在的缓存维度必须与新向量一致
        if cache.data and cache.data.get("dimension") != vector.size:
            raise EmbeddingError("查询向量维度与缓存不一致，请选择新的缓存目录")

        if not hit:
            cache.save({
                "model": self.model,
                "dimension": int(vector.size),
                "queries": {**queries, text: vector.tolist()},
            })

        # FAISS 会原地归一化查询向量，返回独立数组以免影响后续命中
        return vector.copy()

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
    api_key = settings.embedding_api_key
    model = settings.embedding_model

    if not api_key or not model:
        raise ValueError("请配置 RAG_EMBEDDING_API_KEY 和 settings.embedding_model")

    client = OpenAI(
        api_key=api_key,
        base_url=settings.embedding_base_url,
        timeout=settings.embedding_timeout,
        max_retries=settings.embedding_max_retries,
    )
    return EmbeddingClient(client, model=model)
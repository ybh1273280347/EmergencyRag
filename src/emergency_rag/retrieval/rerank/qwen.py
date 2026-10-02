"""Qwen 重排网关边界：完整评分、批内身份映射，不拥有候选预算。"""

import math
from functools import cache
from typing import Any

from openai import APIError, OpenAI

from emergency_rag.retrieval.models import Candidate
from emergency_rag.settings import settings

from .base import Reranker


class QwenRerankerError(RuntimeError):
    pass


class QwenReranker(Reranker):
    """通过 /rerank 获取 Unit 分数；指令作为查询上下文送入网关。"""

    name = "qwen"

    def __init__(
        self,
        client: OpenAI,
        *,
        model: str,
        instruction: str = "",
        batch_size: int = 64,
    ) -> None:
        if not model or not model.strip():
            raise ValueError("Qwen 模型名未配置")
        if batch_size <= 0:
            raise ValueError("Qwen batch_size 必须为正整数")

        self.client = client
        self.model = model
        self.instruction = instruction
        # 请求拆分大小，不是候选数量上限
        self.batch_size = batch_size

    def rerank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        if not candidates:
            return []

        # 重排依赖 unit_id 作为身份键，输入必须唯一
        if len({item.unit_id for item in candidates}) != len(candidates):
            raise ValueError("重排输入包含重复 unit_id")

        # 当前网关忽略独立 instruction 字段，显式将指令送入 query。
        # 这是查询上下文增强，不声称网关支持模型原生的自定义 prompt。
        provider_query = query
        if self.instruction:
            provider_query = f"<Instruct>: {self.instruction}\n<Query>: {query}"

        ranked: list[Candidate] = []

        # 按 batch_size 分批调用，批间独立
        for start in range(0, len(candidates), self.batch_size):
            batch = candidates[start : start + self.batch_size]

            try:
                # OpenAI SDK 没有标准 rerank 资源，使用其自定义请求入口；
                # 仍由 SDK 统一处理连接、超时和重试。
                response = self.client.post(
                    "/rerank",
                    cast_to=dict[str, Any],
                    body={
                        "model": self.model,
                        "query": provider_query,
                        "documents": [item.text for item in batch],
                        "top_n": len(batch),
                    },
                )
            except APIError as exc:
                raise QwenRerankerError(
                    f"Qwen API 调用失败：{type(exc).__name__}"
                ) from exc
            except ValueError as exc:
                raise QwenRerankerError("Qwen 响应不是有效 JSON") from exc

            # 校验 index 完整覆盖 [0, len(batch))
            results = response.get("results") if isinstance(response, dict) else None
            if not isinstance(results, list) or any(not isinstance(item, dict) for item in results):
                raise QwenRerankerError("Qwen 响应缺少有效 results 数组")

            indexes = [item.get("index") for item in results]
            if (
                any(type(index) is not int for index in indexes)
                or sorted(indexes) != list(range(len(batch)))
            ):
                raise QwenRerankerError("Qwen 响应 index 缺失、重复或越界")

            # 逐条解析分数并恢复候选对象
            for item in results:
                try:
                    value = item.get("relevance_score")
                    if isinstance(value, bool):
                        raise ValueError("布尔值不是相关性分数")
                    score = float(value)
                except (TypeError, ValueError) as exc:
                    raise QwenRerankerError(
                        "Qwen relevance_score 不是有效数值"
                    ) from exc
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise QwenRerankerError(
                        "Qwen relevance_score 必须为 [0,1] 内有限数值"
                    )

                # 服务只返回批内 index；必须先恢复 unit_id，不能按响应顺序 zip
                candidate = batch[item["index"]].model_copy(deep=True)
                candidate.final_score = score
                candidate.metadata["rerank"] = {"model": self.model}
                ranked.append(candidate)

        # 分数降序；同分按 unit_id 稳定排序
        return sorted(ranked, key=lambda item: (-item.final_score, item.unit_id))


@cache
def get_rerank_model(
    *, instruction: str = "", batch_size: int = 64,
) -> QwenReranker:
    """按指令与批大小缓存重排器；相同参数复用，不同实验互不覆盖。"""
    api_key = settings.qwen_reranker_api_key or settings.embedding_api_key
    model = settings.qwen_reranker_model
    base_url = settings.qwen_reranker_base_url or settings.embedding_base_url
    if not api_key or not model:
        raise ValueError("请配置 QWEN_RERANKER_API_KEY（或 RAG_EMBEDDING_API_KEY）和 settings.qwen_reranker_model")
    if not base_url:
        raise ValueError("请配置 QWEN_RERANKER_BASE_URL（或 RAG_EMBEDDING_BASE_URL）")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=settings.qwen_reranker_timeout,
        max_retries=settings.qwen_reranker_max_retries,
    )
    return QwenReranker(
        client, model=model, instruction=instruction, batch_size=batch_size,
    )

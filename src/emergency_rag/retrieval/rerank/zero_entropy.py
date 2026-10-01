"""ZeroEntropy 边界：完整评分、批内身份映射、排序，不拥有候选预算。"""

import math
import os
from functools import cache
from xml.sax.saxutils import escape

from zeroentropy import APIError, ZeroEntropy

from emergency_rag.retrieval.models import Candidate

from .base import Reranker


class ZeroEntropyRerankerError(RuntimeError):
    pass


class ZeroEntropyReranker(Reranker):
    name = "zeroentropy"

    def __init__(
        self,
        client: ZeroEntropy,
        *,
        model: str,
        instruction: str,
        batch_size: int = 64,
    ) -> None:
        if not model or not model.strip():
            raise ValueError("ZeroEntropy 模型名未配置")
        if batch_size <= 0:
            raise ValueError("ZeroEntropy batch_size 必须为正整数")

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

        # 拼接 query 与指令；XML 转义避免标签注入
        provider_query = query
        if self.instruction:
            provider_query = (
                f"<query>{escape(query)}</query>"
                f"<instruction>{escape(self.instruction)}</instruction>"
            )

        ranked: list[Candidate] = []

        # 按 batch_size 分批调用，批间独立
        for start in range(0, len(candidates), self.batch_size):
            batch = candidates[start : start + self.batch_size]

            try:
                response = self.client.models.rerank(
                    model=self.model,
                    query=provider_query,
                    documents=[item.text for item in batch],
                    top_n=len(batch),
                )
            except APIError as exc:
                raise ZeroEntropyRerankerError(
                    f"ZeroEntropy API 调用失败：{type(exc).__name__}"
                ) from exc

            # 校验 index 完整覆盖 [0, len(batch))
            if not isinstance(response.results, list):
                raise ZeroEntropyRerankerError("ZeroEntropy 响应缺少 results 数组")

            indexes = [item.index for item in response.results]
            if (
                any(type(index) is not int for index in indexes)
                or sorted(indexes) != list(range(len(batch)))
            ):
                raise ZeroEntropyRerankerError("ZeroEntropy 响应 index 缺失、重复或越界")

            # 逐条解析分数并恢复候选对象
            for item in response.results:
                try:
                    score = float(item.relevance_score)
                except (TypeError, ValueError) as exc:
                    raise ZeroEntropyRerankerError(
                        "ZeroEntropy relevance_score 不是有效数值"
                    ) from exc
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ZeroEntropyRerankerError(
                        "ZeroEntropy relevance_score 必须为 [0,1] 内有限数值"
                    )

                # 服务只返回批内 index；必须先恢复 unit_id，不能按响应顺序 zip
                candidate = batch[item.index].model_copy(deep=True)
                candidate.final_score = score
                candidate.metadata["rerank"] = {"model": self.model}
                ranked.append(candidate)

        # 分数降序；同分按 unit_id 稳定排序
        return sorted(ranked, key=lambda item: (-item.final_score, item.unit_id))


@cache
def get_rerank_model(
    *, instruction: str = "", batch_size: int = 64,
) -> ZeroEntropyReranker:
    """按指令与批大小缓存重排器；相同参数复用，不同实验互不覆盖。"""
    api_key = os.environ.get("ZEROENTROPY_API_KEY")
    model = os.environ.get("ZEROENTROPY_MODEL")
    if not api_key or not model:
        raise ValueError("请设置 ZEROENTROPY_API_KEY 和 ZEROENTROPY_MODEL")

    client = ZeroEntropy(
        api_key=api_key,
        base_url=os.environ.get("ZEROENTROPY_URL"),
        timeout=float(os.environ.get("ZEROENTROPY_TIMEOUT", "60")),
        max_retries=int(os.environ.get("ZEROENTROPY_MAX_RETRIES", "2")),
    )
    return ZeroEntropyReranker(
        client, model=model, instruction=instruction, batch_size=batch_size,
    )

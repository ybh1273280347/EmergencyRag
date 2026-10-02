"""本地 BCE 重排：滑窗评分并保留全部 Unit，不执行阈值过滤或 Top-K。"""

import math
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

from emergency_rag.retrieval.models import Candidate

from .base import Reranker

if TYPE_CHECKING:
    from BCEmbedding import RerankerModel


class BCERerankerError(RuntimeError):
    pass


class BCEReranker(Reranker):
    """将 BCE SDK 的文档排序结果映射回完整 Candidate 集合。"""

    name = "bce"

    def __init__(self, model: "RerankerModel", *, batch_size: int = 32) -> None:
        if batch_size <= 0:
            raise ValueError("BCE batch_size 必须为正整数")

        self.model = model
        self.batch_size = batch_size

    def rerank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        if not candidates:
            return []

        # compute_score 会截断长文本；rerank 才执行滑窗和最大分数聚合
        response = self.model.rerank(
            query=query,
            passages=[item.text for item in candidates],
            batch_size=self.batch_size,
        )

        indexes = response.get("rerank_ids") if isinstance(response, dict) else None
        scores = response.get("rerank_scores") if isinstance(response, dict) else None

        # 响应必须完整覆盖输入候选的索引集合与分数数量
        if (
            not isinstance(indexes, list)
            or not isinstance(scores, list)
            or len(scores) != len(candidates)
            or any(type(index) is not int for index in indexes)
            or sorted(indexes) != list(range(len(candidates)))
        ):
            raise BCERerankerError("BCE 响应必须完整覆盖输入候选的索引与分数")

        ranked = []
        for index, value in zip(indexes, scores, strict=True):
            try:
                if isinstance(value, bool):
                    raise ValueError("布尔值不是相关性分数")
                score = float(value)
            except (TypeError, ValueError) as exc:
                raise BCERerankerError("BCE 分数不是有效数值") from exc

            if not math.isfinite(score) or not 0 <= score <= 1:
                raise BCERerankerError("BCE 分数必须为 [0,1] 内有限数值")

            # 按 SDK 返回的原始文档索引恢复身份，不能按排序后的位置 zip 候选
            item = candidates[index].model_copy(deep=True)
            item.final_score = score
            item.metadata["rerank"] = {"model": "bce-reranker-base_v1"}
            ranked.append(item)

        # 分数降序、unit_id 升序，稳定排序
        return sorted(ranked, key=lambda item: (-item.final_score, item.unit_id))


@cache
def get_rerank_model(
    *,
    model_path: str = "models/bce-reranker-base_v1",
    device: str | None = None,
    use_fp16: bool = True,
    batch_size: int = 32,
) -> BCEReranker:
    """首次使用时加载本地权重并缓存；未选择 BCE 时不导入 Torch 或占用显存。"""
    import torch
    from BCEmbedding import RerankerModel

    directory = Path(model_path).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"BCE 本地模型目录不存在：{directory}")

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # CPU 使用 FP32；启用 FP16 仅在 CUDA 推理时生效
    model = RerankerModel(
        model_name_or_path=str(directory),
        device=selected_device,
        use_fp16=use_fp16 and selected_device != "cpu",
        local_files_only=True,
    )
    return BCEReranker(model, batch_size=batch_size)
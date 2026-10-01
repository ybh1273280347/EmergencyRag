from dataclasses import dataclass, field
from itertools import zip_longest

from emergency_rag.retrieval.models import Candidate

from .base import Fusion
from .merge import merge_candidate


@dataclass(frozen=True, slots=True)
class UnionFusionConfig:
    # 每路先消耗预算，再合并去重；不补齐重叠候选。
    per_source_top_k: dict[str, int] = field(default_factory=lambda: {"bm25": 30, "dense": 30})


class UnionFusion(Fusion):
    """先消耗各来源预算，再去重；不存在融合后总上限或重叠补齐。"""

    name = "union"

    def __init__(self, config: UnionFusionConfig | None = None) -> None:
        self.config = config or UnionFusionConfig()
        if any(type(limit) is not int or limit < 0 for limit in self.config.per_source_top_k.values()):
            raise ValueError("Union 每路预算必须为非负整数")

    def fuse(self, result_sets: list[list[Candidate]]) -> list[Candidate]:
        selected_sets: list[list[Candidate]] = []
        for result_set in result_sets:
            if not result_set:
                continue
            source_names = {source for candidate in result_set for source in candidate.sources}
            if len(source_names) != 1:
                raise ValueError("Union 输入的每个结果集必须来自一个稳定来源")
            source = next(iter(source_names))
            if source not in self.config.per_source_top_k:
                raise ValueError(f"Union 缺少来源预算：{source}")
            selected_sets.append(result_set[: self.config.per_source_top_k[source]])

        merged: dict[str, Candidate] = {}
        # 交替遍历只决定稳定输出顺序，不比较不同算法的原始得分。
        for round_candidates in zip_longest(*selected_sets):
            for candidate in round_candidates:
                if candidate is not None:
                    merge_candidate(merged, candidate)
        candidates = list(merged.values())
        for rank, candidate in enumerate(candidates, start=1):
            candidate.metadata["fusion"] = {"strategy": self.name, "rank": rank, "score": None}
        return candidates

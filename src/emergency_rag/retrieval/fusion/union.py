from itertools import zip_longest

from emergency_rag.retrieval.models import Candidate

from .base import Fusion
from .utils.merge import merge_candidate


class UnionFusion(Fusion):
    """保持各路输入顺序，交替合并全部候选；召回数量由检索器决定。"""

    name = "union"

    def fuse(self, result_sets: list[list[Candidate]]) -> list[Candidate]:
        merged: dict[str, Candidate] = {}
        # 交替遍历只决定稳定输出顺序，不比较不同算法的原始得分。
        for round_candidates in zip_longest(*result_sets):
            for candidate in round_candidates:
                if candidate is not None:
                    merge_candidate(merged, candidate)
        candidates = list(merged.values())
        for rank, candidate in enumerate(candidates, start=1):
            candidate.metadata["fusion"] = {"strategy": self.name, "rank": rank, "score": None}
        return candidates

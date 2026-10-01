from emergency_rag.retrieval.models import Candidate

from .base import Fusion
from .utils.merge import merge_candidate


class RRFFusion(Fusion):
    """RRF 只拥有融合后总预算，独立于各路召回及最终返回数量。"""

    name = "rrf"

    def __init__(self, *, k: int = 60, top_k: int = 60) -> None:
        if  k < 0:
            raise ValueError("RRF k 必须为非负整数")
        if top_k <= 0:
            raise ValueError("RRF top_k 必须为正整数")
        self.k = k
        self.top_k = top_k

    def fuse(self, result_sets: list[list[Candidate]]) -> list[Candidate]:
        merged: dict[str, Candidate] = {}
        scores: dict[str, float] = {}
        for result_set in result_sets:
            seen: set[str] = set()
            # rank 是当前召回列表中的 1-based 位置，不能使用最终排名。
            for rank, candidate in enumerate(result_set, start=1):
                merge_candidate(merged, candidate)
                if candidate.unit_id in seen:
                    continue
                seen.add(candidate.unit_id)
                scores[candidate.unit_id] = (
                    scores.get(candidate.unit_id, 0.0)
                    + 1.0 / (self.k + rank)
                )
        ordered = sorted(merged.values(), key=lambda item: (-scores[item.unit_id], item.unit_id))
        for rank, candidate in enumerate(ordered, start=1):
            candidate.metadata["fusion"] = {
                "strategy": self.name,
                "rank": rank,
                "score": scores[candidate.unit_id],
            }
        return ordered[: self.top_k]

from dataclasses import dataclass
from emergency_rag.retrieval.models import Candidate

from .base import Fusion
from .merge import merge_candidate


@dataclass(frozen=True, slots=True)
class RRFFusionConfig:
    rank_constant: int = 60
    max_candidates: int = 60  # 融合排序后的总上限


class RRFFusion(Fusion):
    """RRF 只拥有融合后总预算，独立于各路召回及最终返回数量。"""

    name = "rrf"

    def __init__(self, config: RRFFusionConfig | None = None) -> None:
        self.config = config or RRFFusionConfig()
        if type(self.config.rank_constant) is not int or self.config.rank_constant < 0:
            raise ValueError("RRF rank_constant 必须为非负整数")
        if type(self.config.max_candidates) is not int or self.config.max_candidates <= 0:
            raise ValueError("RRF max_candidates 必须为正整数")

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
                    + 1.0 / (self.config.rank_constant + rank)
                )
        ordered = sorted(merged.values(), key=lambda item: (-scores[item.unit_id], item.unit_id))
        for rank, candidate in enumerate(ordered, start=1):
            candidate.metadata["fusion"] = {
                "strategy": self.name,
                "rank": rank,
                "score": scores[candidate.unit_id],
            }
        return ordered[: self.config.max_candidates]

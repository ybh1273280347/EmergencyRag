from abc import ABC

from emergency_rag.retrieval.models import Candidate


class CandidateExpander(ABC):
    """候选扩展接口，默认原样返回。"""

    name = "default"

    def expand(self, candidates: list[Candidate]) -> list[Candidate]:
        return candidates

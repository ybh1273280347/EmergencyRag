from abc import ABC

from emergency_rag.retrieval.models import Candidate, QueryContext


class CandidateExpander(ABC):
    """候选扩展接口，默认原样返回。"""

    def expand(self, candidates: list[Candidate]) -> list[Candidate]:
        return candidates

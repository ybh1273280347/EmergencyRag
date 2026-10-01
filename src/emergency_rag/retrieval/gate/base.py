from abc import ABC

from emergency_rag.retrieval.models import Candidate


class CandidateGate(ABC):
    """候选过滤接口，默认放行全部候选。"""

    def filter(self, candidates: list[Candidate]) -> list[Candidate]:
        return candidates

from abc import ABC, abstractmethod

from emergency_rag.retrieval.models import Candidate


class Fusion(ABC):
    """融合接口；候选预算属于具体策略。"""

    name: str

    @abstractmethod
    def fuse(self, result_sets: list[list[Candidate]]) -> list[Candidate]:
        raise NotImplementedError

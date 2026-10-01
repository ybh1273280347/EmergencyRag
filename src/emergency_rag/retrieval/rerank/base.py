from abc import ABC, abstractmethod

from emergency_rag.retrieval.models import Candidate


class Reranker(ABC):
    """完整评分和排序接口。"""

    name: str

    @abstractmethod
    def rerank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        raise NotImplementedError

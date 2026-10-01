from abc import ABC, abstractmethod

from emergency_rag.retrieval.models import Candidate


class Retriever(ABC):
    """独立召回接口；默认预算属于具体组件。"""

    name: str

    @abstractmethod
    def retrieve(self, query: str, top_k: int | None = None) -> list[Candidate]:
        raise NotImplementedError

from abc import ABC

from emergency_rag.retrieval.models import Candidate


class Reranker(ABC):
    """重排接口，默认原样返回候选，便于独立验证检索链路。"""

    name = "passthrough"

    def rerank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        return candidates

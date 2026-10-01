from abc import ABC

from emergency_rag.retrieval.models import Candidate


class EvidenceProjector(ABC):
    """证据投影接口，默认保留当前检索单元。"""

    def project(self, candidates: list[Candidate]) -> list[Candidate]:
        return candidates

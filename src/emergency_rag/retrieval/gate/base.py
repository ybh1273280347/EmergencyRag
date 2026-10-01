from abc import ABC

from emergency_rag.retrieval.models import RuleEvidence


class EvidenceGate(ABC):
    """完整规则证据的过滤接口，默认放行全部规则。"""

    def filter(self, evidence: list[RuleEvidence]) -> list[RuleEvidence]:
        return evidence

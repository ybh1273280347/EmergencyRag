"""在线阶段编排；组件由调用方注入，默认阶段由其 ABC 提供。"""

import logging
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from time import perf_counter

from emergency_rag.data.models import IndexedDataset

from .expansion.base import CandidateExpander
from .fusion.base import Fusion
from .gate.base import EvidenceGate
from .models import Candidate, RetrievalResult, RuleEvidence
from .query.base import QueryProcessor
from .rerank.base import Reranker
from .retrievers.base import Retriever

logger = logging.getLogger(__name__)


def aggregate_rule_evidence(
    candidates: list[Candidate],
    dataset: IndexedDataset,
) -> list[RuleEvidence]:
    """按规则聚合全部命中单元，恢复完整原文并取最高重排分数。

    无打分时保留首次命中的规则顺序，不把融合或召回分数当成相关性分数。
    聚合只使用本次命中，不补查兄弟单元或重新评分完整规则。
    """
    grouped: dict[str, RuleEvidence] = {}

    for candidate in candidates:
        # 首次命中该规则时，从 Dataset 恢复完整原文
        if candidate.rule_id not in grouped:
            rule = dataset.rules[candidate.rule_id]
            grouped[candidate.rule_id] = RuleEvidence(
                rule_id=rule.rule_id,
                text=rule.text,
                metadata=deepcopy(rule.metadata),
            )

        evidence = grouped[candidate.rule_id]
        evidence.matched_units.append(candidate)

        # 合并来源，去重并保持出现顺序
        for source in candidate.sources:
            if source not in evidence.sources:
                evidence.sources.append(source)

        # 取命中单元中的最高重排分数作为规则分数
        if candidate.final_score is not None:
            if evidence.final_score is None or candidate.final_score > evidence.final_score:
                evidence.final_score = candidate.final_score

    evidence = list(grouped.values())

    # 仅有分数时才排序；否则保留首次命中的规则顺序
    if any(item.final_score is not None for item in evidence):
        evidence.sort(
            key=lambda item: (
                -(item.final_score if item.final_score is not None else -1.0),
                item.rule_id,
            ),
        )

    return evidence


@dataclass(slots=True)
class RetrievalPipeline:
    dataset: IndexedDataset
    retrievers: Sequence[Retriever]
    fusion: Fusion

    # 未显式传入时使用 ABC 默认行为，每个 Pipeline 持有独立的阶段实例
    query_processor: QueryProcessor = field(default_factory=QueryProcessor)
    reranker: Reranker = field(default_factory=Reranker)
    expander: CandidateExpander = field(default_factory=CandidateExpander)
    gate: EvidenceGate = field(default_factory=EvidenceGate)

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 必须为非空文本")
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k 必须为正整数")

        started = perf_counter()

        # 查询处理阶段：可能改写、扩展成多个子查询
        query_ctx = self.query_processor.process(query)

        # 多子查询 × 多召回路，逐路调用检索器
        result_sets = []
        retrieval_counts: dict[str, int] = {}
        for subquery in query_ctx.queries:
            for retriever in self.retrievers:
                result_set = retriever.retrieve(subquery, self.dataset)
                result_sets.append(result_set)
                retrieval_counts[retriever.name] = (
                    retrieval_counts.get(retriever.name, 0) + len(result_set)
                )

        # 融合 → 扩展 → 重排，仍以候选单元为单位
        candidates = self.fusion.fuse(result_sets)
        fusion_count = len(candidates)
        candidates = self.expander.expand(candidates)
        candidates = self.reranker.rerank(query_ctx.original_query, candidates)

        # 聚合到规则级别后过门控，再做最终截断
        evidence = aggregate_rule_evidence(candidates, self.dataset)
        rule_count = len(evidence)
        evidence = self.gate.filter(evidence)

        # Gate 与最终 Top-K 都以完整规则为单位，子单元不会重复占据名额
        evidence = [
            item.model_copy(update={"final_rank": rank})
            for rank, item in enumerate(evidence[:top_k], start=1)
        ]

        latency_ms = (perf_counter() - started) * 1000
        logger.info(
            "检索完成 fusion=%s rules=%d latency_ms=%.2f",
            self.fusion.name,
            len(evidence),
            latency_ms,
        )

        return RetrievalResult(
            query=query_ctx.original_query,
            evidence=evidence,
            metadata={
                "latency_ms": latency_ms,
                "retrieval_counts": retrieval_counts,
                "candidate_counts": {
                    "fusion": fusion_count,
                    "rules": rule_count,
                    "final": len(evidence),
                },
                "active_retrievers": [retriever.name for retriever in self.retrievers],
                "fusion_strategy": self.fusion.name,
                "reranker": self.reranker.name,
            },
        )
"""在线阶段编排；组件由调用方注入，默认阶段由其 ABC 提供。"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from time import perf_counter

from emergency_rag.data.models import IndexedDataset

from .expansion.base import CandidateExpander
from .fusion.base import Fusion
from .gate.base import CandidateGate
from .models import RetrievalResult
from .projector.base import EvidenceProjector
from .query.base import QueryProcessor
from .rerank.base import Reranker
from .retrievers.base import Retriever

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RetrievalPipeline:
    dataset: IndexedDataset
    retrievers: Sequence[Retriever]
    fusion: Fusion
    # 未显式传入时使用 ABC 默认行为，每个 Pipeline 持有独立的阶段实例。
    query_processor: QueryProcessor = field(default_factory=QueryProcessor)
    reranker: Reranker = field(default_factory=Reranker)
    expander: CandidateExpander = field(default_factory=CandidateExpander)
    gate: CandidateGate = field(default_factory=CandidateGate)
    projector: EvidenceProjector = field(default_factory=EvidenceProjector)

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 必须为非空文本")
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k 必须为正整数")
        started = perf_counter()
        query_ctx = self.query_processor.process(query)

        result_sets = []
        retrieval_counts: dict[str, int] = {}
        for subquery in query_ctx.queries:
            for retriever in self.retrievers:
                result_set = retriever.retrieve(subquery, self.dataset)
                result_sets.append(result_set)
                retrieval_counts[retriever.name] = retrieval_counts.get(retriever.name, 0) + len(result_set)

        candidates = self.fusion.fuse(result_sets)
        fusion_count = len(candidates)
        candidates = self.expander.expand(query_ctx, candidates)
        candidates = self.reranker.rerank(query_ctx.original_query, candidates)
        candidates = self.gate.filter(candidates)
        # 最终 Top-K 只截断通过 Gate 的有序候选，不改变任何组件的预算。
        candidates = self.projector.project(candidates[:top_k])
        candidates = [
            candidate.model_copy(update={"final_rank": rank})
            for rank, candidate in enumerate(candidates, start=1)
        ]
        latency_ms = (perf_counter() - started) * 1000
        logger.info("检索完成 fusion=%s candidates=%d latency_ms=%.2f", self.fusion.name, len(candidates), latency_ms)
        return RetrievalResult(
            query=query,
            candidates=candidates,
            metadata={
                "latency_ms": latency_ms,
                "retrieval_counts": retrieval_counts,
                "candidate_counts": {"fusion": fusion_count, "final": len(candidates)},
                "active_retrievers": [retriever.name for retriever in self.retrievers],
                "fusion_strategy": self.fusion.name,
                "reranker": self.reranker.name,
            },
        )

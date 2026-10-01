"""在线阶段编排；召回与融合预算均由具体组件拥有。"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

from .expansion.base import CandidateExpander
from .fusion.base import Fusion
from .gate.base import CandidateGate
from .models import RetrievalResult
from .projector.base import EvidenceProjector
from .query.base import QueryProcessor
from .rerank.base import Reranker
from .retrievers.base import Retriever

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RetrievalPipelineConfig:
    final_top_k: int = 10


class RetrievalPipeline:
    def __init__(
        self,
        *,
        retrievers: Sequence[Retriever],
        fusion: Fusion,
        reranker: Reranker,
        query_processor: QueryProcessor | None = None,
        expander: CandidateExpander | None = None,
        gate: CandidateGate | None = None,
        projector: EvidenceProjector | None = None,
        config: RetrievalPipelineConfig | None = None,
    ) -> None:
        # None 表示使用该阶段基类的默认实现，不跳过生命周期阶段。
        self.query_processor = QueryProcessor() if query_processor is None else query_processor
        self.retrievers = retrievers
        self.fusion = fusion
        self.expander = CandidateExpander() if expander is None else expander
        self.reranker = reranker
        self.gate = CandidateGate() if gate is None else gate
        self.projector = EvidenceProjector() if projector is None else projector
        self.config = config or RetrievalPipelineConfig()

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 必须为非空文本")
        limit = self.config.final_top_k if top_k is None else top_k
        if type(limit) is not int or limit <= 0:
            raise ValueError("top_k 必须为正整数")
        started = perf_counter()
        query_ctx = self.query_processor.process(query)

        result_sets = []
        retrieval_counts: dict[str, int] = {}
        for subquery in query_ctx.queries:
            for retriever in self.retrievers:
                result_set = retriever.retrieve(subquery)
                result_sets.append(result_set)
                retrieval_counts[retriever.name] = retrieval_counts.get(retriever.name, 0) + len(result_set)

        candidates = self.fusion.fuse(result_sets)
        fusion_count = len(candidates)
        candidates = self.expander.expand(query_ctx, candidates)
        candidates = self.reranker.rerank(query_ctx.original_query, candidates)
        candidates = self.gate.filter(query_ctx, candidates)
        candidates = self.projector.project(candidates)

        # 最终预算在投影之后消耗；重排器对输入的完整候选集评分。
        candidates = [
            candidate.model_copy(update={"final_rank": rank})
            for rank, candidate in enumerate(candidates[:limit], start=1)
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

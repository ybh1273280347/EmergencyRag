
import pytest

from emergency_rag.retrieval.query.base import QueryProcessor as QueryProcessorBase
from emergency_rag.retrieval.retrievers.base import Retriever as RetrieverBase
from emergency_rag.retrieval.fusion.base import Fusion as FusionBase
from emergency_rag.retrieval.expansion.base import CandidateExpander
from emergency_rag.retrieval.rerank.base import Reranker as RerankerBase
from emergency_rag.retrieval.gate.base import CandidateGate
from emergency_rag.retrieval.projector.base import EvidenceProjector
from emergency_rag.retrieval.models import QueryContext
from emergency_rag.retrieval.pipeline import RetrievalPipeline, RetrievalPipelineConfig
from emergency_rag.chunking.base import Chunker
from emergency_rag.retrieval.tokenizer.base import TextTokenizer


def make_pipeline(candidate, events):
    class QueryProcessor(QueryProcessorBase):
        def process(self, query):
            events.append(("query", query))
            return QueryContext(original_query=query, queries=[query, "子查询"])

    class Retriever(RetrieverBase):
        def __init__(self, name, top_k):
            self.name = name
            self.top_k = top_k

        def retrieve(self, query, top_k=None):
            events.append((self.name, query, self.top_k if top_k is None else top_k))
            return [candidate(self.name, self.name, query=query)]

    class Fusion(FusionBase):
        name = "test-fusion"

        def fuse(self, result_sets):
            events.append(("fusion", len(result_sets)))
            return [candidate("a"), candidate("b"), candidate("c")]

    class Expander(CandidateExpander):
        def expand(self, query_ctx, candidates):
            events.append(("expansion", len(candidates)))
            return candidates

    class Reranker(RerankerBase):
        name = "test-reranker"

        def rerank(self, query, candidates):
            events.append(("rerank", query, len(candidates)))
            return [item.model_copy(update={"final_score": score}) for item, score in zip(candidates, [0.9, 0.8, 0.7])]

    class Gate(CandidateGate):
        def filter(self, query_ctx, candidates):
            events.append(("gate", len(candidates)))
            return candidates

    class Projector(EvidenceProjector):
        def project(self, candidates):
            events.append(("projector", len(candidates)))
            return candidates[1:]

    return RetrievalPipeline(
        query_processor=QueryProcessor(),
        retrievers=[Retriever("bm25", 7), Retriever("dense", 11)],
        fusion=Fusion(), expander=Expander(), reranker=Reranker(),
        gate=Gate(), projector=Projector(), config=RetrievalPipelineConfig(final_top_k=2),
    )


def test_lifecycle_and_final_limit_after_projection(candidate):
    events = []
    pipeline = make_pipeline(candidate, events)
    result = pipeline.retrieve("原查询", top_k=1)
    assert events == [
        ("query", "原查询"),
        ("bm25", "原查询", 7), ("dense", "原查询", 11),
        ("bm25", "子查询", 7), ("dense", "子查询", 11),
        ("fusion", 4), ("expansion", 3), ("rerank", "原查询", 3),
        ("gate", 3), ("projector", 3),
    ]
    assert [item.unit_id for item in result.candidates] == ["b"]
    assert result.candidates[0].final_rank == 1
    assert result.metadata["retrieval_counts"] == {"bm25": 2, "dense": 2}
    assert result.metadata["candidate_counts"] == {"fusion": 3, "final": 1}
    assert result.metadata["latency_ms"] >= 0
    assert result.metadata["reranker"] == "test-reranker"
    assert len(pipeline.retrieve("原查询").candidates) == 2
    assert len(pipeline.retrieve("原查询", top_k=100).candidates) == 2


@pytest.mark.parametrize("query,top_k", [(" ", 1), (None, 1), ("问题", 0), ("问题", -1), ("问题", True), ("问题", 1.5)])
def test_invalid_public_input(candidate, query, top_k):
    with pytest.raises(ValueError):
        make_pipeline(candidate, []).retrieve(query, top_k)


@pytest.mark.parametrize("interface", [Chunker, RetrieverBase, FusionBase, RerankerBase, TextTokenizer])
def test_algorithm_interfaces_require_implementation(interface):
    with pytest.raises(TypeError):
        interface()


def test_none_uses_default_stage_implementations(candidate):
    class Retriever(RetrieverBase):
        name = "bm25"

        def retrieve(self, query, top_k=None):
            return [candidate("a", query=query)]

    class Fusion(FusionBase):
        name = "test"

        def fuse(self, result_sets):
            return result_sets[0]

    class Reranker(RerankerBase):
        name = "test"

        def rerank(self, query, candidates):
            return [item.model_copy(update={"final_score": 0.8}) for item in candidates]

    pipeline = RetrievalPipeline(
        retrievers=[Retriever()], fusion=Fusion(), reranker=Reranker(),
        query_processor=None, expander=None, gate=None, projector=None,
    )
    assert type(pipeline.query_processor) is QueryProcessorBase
    assert type(pipeline.expander) is CandidateExpander
    assert type(pipeline.gate) is CandidateGate
    assert type(pipeline.projector) is EvidenceProjector
    result = pipeline.retrieve("保留原查询")
    assert result.candidates[0].final_score == 0.8
    assert result.candidates[0].final_rank == 1
    assert result.candidates[0].metadata["retrieval"][0]["query"] == "保留原查询"
    ctx = pipeline.query_processor.process("问题")
    assert ctx.queries == ["问题"]
    assert pipeline.expander.expand(ctx, result.candidates) is result.candidates
    assert pipeline.gate.filter(ctx, result.candidates) is result.candidates
    assert pipeline.projector.project(result.candidates) is result.candidates


def test_custom_falsy_component_is_not_replaced_by_default(candidate):
    class FalsyGate(CandidateGate):
        def __bool__(self):
            return False

        def filter(self, query_ctx, candidates):
            return []

    pipeline = make_pipeline(candidate, [])
    custom = FalsyGate()
    configured = RetrievalPipeline(
        retrievers=pipeline.retrievers, fusion=pipeline.fusion, reranker=pipeline.reranker,
        gate=custom,
    )
    assert configured.gate is custom
    assert configured.retrieve("问题").candidates == []


def test_caller_can_supply_unlisted_component_types(candidate):
    # 自定义来源及算法无需全局配置、注册或修改 Pipeline 的分支。
    class ArchiveRetriever(RetrieverBase):
        name = "archive"

        def retrieve(self, query, top_k=None):
            return [candidate("a", self.name, query=query)]

    class FirstFusion(FusionBase):
        name = "first"

        def fuse(self, result_sets):
            return result_sets[0]

    class LocalReranker(RerankerBase):
        name = "local"

        def rerank(self, query, candidates):
            return [item.model_copy(update={"final_score": 0.75}) for item in candidates]

    pipeline = RetrievalPipeline(
        retrievers=(ArchiveRetriever(),), fusion=FirstFusion(), reranker=LocalReranker(),
    )
    result = pipeline.retrieve("查档案")
    assert result.candidates[0].sources == ["archive"]
    assert result.candidates[0].final_score == 0.75
    assert result.metadata["active_retrievers"] == ["archive"]
    assert result.metadata["fusion_strategy"] == "first"
    assert result.metadata["reranker"] == "local"


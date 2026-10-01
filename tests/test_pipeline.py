
import pytest

from emergency_rag.retrieval.query.base import QueryProcessor as QueryProcessorBase
from emergency_rag.retrieval.retrievers.base import Retriever as RetrieverBase
from emergency_rag.retrieval.fusion.base import Fusion as FusionBase
from emergency_rag.retrieval.expansion.base import CandidateExpander
from emergency_rag.retrieval.rerank.base import Reranker as RerankerBase
from emergency_rag.retrieval.gate.base import CandidateGate
from emergency_rag.retrieval.projector.base import EvidenceProjector
from emergency_rag.retrieval.models import QueryContext
from emergency_rag.retrieval.pipeline import RetrievalPipeline
from emergency_rag.chunking.base import Chunker
from emergency_rag.retrieval.tokenizer.base import TextTokenizer


def make_pipeline(candidate, events, dataset):
    expected_dataset = dataset
    class QueryProcessor(QueryProcessorBase):
        def process(self, query):
            events.append(("query", query))
            return QueryContext(original_query=query, rewritten_query="改写查询", sub_queries=["子查询"])

    class Retriever(RetrieverBase):
        def __init__(self, name, top_k):
            self.name = name
            self.top_k = top_k

        def retrieve(self, query, dataset):
            assert dataset is expected_dataset
            events.append((self.name, query, self.top_k))
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
        def filter(self, candidates):
            events.append(("gate", len(candidates)))
            return candidates[1:]

    class Projector(EvidenceProjector):
        def project(self, candidates):
            events.append(("projector", len(candidates)))
            return candidates

    return RetrievalPipeline(
        dataset=dataset,
        query_processor=QueryProcessor(),
        retrievers=[Retriever("bm25", 7), Retriever("dense", 11)],
        fusion=Fusion(), expander=Expander(), reranker=Reranker(),
        gate=Gate(), projector=Projector(),
    )


def test_lifecycle_and_final_limit_after_gate(candidate, dataset):
    events = []
    pipeline = make_pipeline(candidate, events, dataset)
    result = pipeline.retrieve("原查询", top_k=1)
    assert events == [
        ("query", "原查询"),
        ("bm25", "改写查询", 7), ("dense", "改写查询", 11),
        ("bm25", "子查询", 7), ("dense", "子查询", 11),
        ("fusion", 4), ("expansion", 3), ("rerank", "原查询", 3),
        ("gate", 3), ("projector", 1),
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
def test_invalid_public_input(candidate, query, top_k, dataset):
    with pytest.raises(ValueError):
        make_pipeline(candidate, [], dataset).retrieve(query, top_k)


@pytest.mark.parametrize("interface", [Chunker, RetrieverBase, FusionBase, TextTokenizer])
def test_algorithm_interfaces_require_implementation(interface):
    with pytest.raises(TypeError):
        interface()


def test_default_stage_implementations(candidate, dataset):
    class Retriever(RetrieverBase):
        name = "bm25"

        def retrieve(self, query, actual_dataset):
            assert actual_dataset is dataset
            return [candidate("a", query=query)]

    class Fusion(FusionBase):
        name = "test"

        def fuse(self, result_sets):
            return result_sets[0]

    pipeline = RetrievalPipeline(dataset=dataset, retrievers=[Retriever()], fusion=Fusion())
    assert type(pipeline.query_processor) is QueryProcessorBase
    assert type(pipeline.reranker) is RerankerBase
    assert type(pipeline.expander) is CandidateExpander
    assert type(pipeline.gate) is CandidateGate
    assert type(pipeline.projector) is EvidenceProjector
    result = pipeline.retrieve("保留原查询")
    assert result.candidates[0].final_score is None
    assert result.candidates[0].final_rank == 1
    assert result.candidates[0].metadata["retrieval"][0]["query"] == "保留原查询"
    ctx = pipeline.query_processor.process("问题")
    assert ctx.queries == ["问题"]
    assert pipeline.expander.expand(ctx, result.candidates) is result.candidates
    assert pipeline.gate.filter(result.candidates) is result.candidates
    assert pipeline.projector.project(result.candidates) is result.candidates


def test_default_reranker_returns_input_unchanged(candidate):
    candidates = [candidate("b"), candidate("a")]
    candidates[0].final_score = 0.7
    assert RerankerBase().rerank("问题", candidates) is candidates
    assert [item.unit_id for item in candidates] == ["b", "a"]
    assert [item.final_score for item in candidates] == [0.7, None]


def test_query_context_combines_rewrite_and_subqueries_without_mutation():
    context = QueryContext(original_query="原查询", rewritten_query="改写查询", sub_queries=["子查询一", "子查询二"])
    assert context.queries == ["改写查询", "子查询一", "子查询二"]
    assert context.original_query == "原查询"
    context.queries.append("临时值")
    assert context.sub_queries == ["子查询一", "子查询二"]
    fallback = QueryContext(original_query="原查询")
    assert fallback.queries == ["原查询"]
    assert fallback.sub_queries == []
    context.sub_queries.append("新增子查询")
    assert fallback.sub_queries == []


def test_custom_falsy_component_is_not_replaced_by_default(candidate, dataset):
    class FalsyGate(CandidateGate):
        def __bool__(self):
            return False

        def filter(self, candidates):
            return []

    pipeline = make_pipeline(candidate, [], dataset)
    custom = FalsyGate()
    configured = RetrievalPipeline(
        dataset=pipeline.dataset,
        retrievers=pipeline.retrievers, fusion=pipeline.fusion, reranker=pipeline.reranker,
        gate=custom,
    )
    assert configured.gate is custom
    assert configured.retrieve("问题").candidates == []


def test_caller_can_supply_unlisted_component_types(candidate, dataset):
    # 自定义来源及算法无需全局配置、注册或修改 Pipeline 的分支。
    class ArchiveRetriever(RetrieverBase):
        name = "archive"

        def retrieve(self, query, actual_dataset):
            assert actual_dataset is dataset
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
        dataset=dataset,
        retrievers=(ArchiveRetriever(),), fusion=FirstFusion(), reranker=LocalReranker(),
    )
    result = pipeline.retrieve("查档案")
    assert result.candidates[0].sources == ["archive"]
    assert result.candidates[0].final_score == 0.75
    assert result.metadata["active_retrievers"] == ["archive"]
    assert result.metadata["fusion_strategy"] == "first"
    assert result.metadata["reranker"] == "local"


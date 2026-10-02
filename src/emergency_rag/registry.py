"""按阶段显式登记 YAML 可用组件；工厂只负责组件所需模型的注入。"""

from collections.abc import Callable
from typing import Any

from emergency_rag.clients import embedding
from emergency_rag.retrieval.expansion.base import CandidateExpander
from emergency_rag.retrieval.fusion.rrf import RRFFusion
from emergency_rag.retrieval.fusion.union import UnionFusion
from emergency_rag.retrieval.gate.base import EvidenceGate
from emergency_rag.retrieval.query.base import QueryProcessor
from emergency_rag.retrieval.rerank import bce, qwen
from emergency_rag.retrieval.rerank.base import Reranker
from emergency_rag.retrieval.retrievers.bm25 import BM25Retriever
from emergency_rag.retrieval.retrievers.dense import DenseRetriever
from emergency_rag.retrieval.tokenizer.jieba import JiebaTokenizer
from emergency_rag.data.units.rule import RuleUnitBuilder


def _dense_retriever(**params: Any) -> DenseRetriever:
    return DenseRetriever(
        embedding_client=embedding.get_embedding_client(),
        **params,
    )


def _qwen_reranker(**params: Any) -> qwen.QwenReranker:
    return qwen.get_rerank_model(**params)


def _bce_reranker(**params: Any) -> bce.BCEReranker:
    return bce.get_rerank_model(**params)


COMPONENT_REGISTRY: dict[str, dict[str, Callable[..., Any]]] = {
    "unit_builder": {
        RuleUnitBuilder.name: RuleUnitBuilder,
    },
    "tokenizer": {
        JiebaTokenizer.name: JiebaTokenizer,
    },
    "retrievers": {
        BM25Retriever.name: BM25Retriever,
        DenseRetriever.name: _dense_retriever,
    },
    "fusion": {
        RRFFusion.name: RRFFusion,
        UnionFusion.name: UnionFusion,
    },
    "query_processor": {
        QueryProcessor.name: QueryProcessor,
    },
    "expander": {
        CandidateExpander.name: CandidateExpander,
    },
    "reranker": {
        Reranker.name: Reranker,
        qwen.QwenReranker.name: _qwen_reranker,
        bce.BCEReranker.name: _bce_reranker,
    },
    "gate": {
        EvidenceGate.name: EvidenceGate,
    },
}

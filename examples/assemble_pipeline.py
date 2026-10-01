"""调用方装配示例：只使用 BM25，依赖已经准备好的资源。"""

from zeroentropy import ZeroEntropy

from emergency_rag.data.models import SearchUnit
from emergency_rag.retrieval.fusion.rrf import RRFFusion, RRFFusionConfig
from emergency_rag.retrieval.pipeline import RetrievalPipeline, RetrievalPipelineConfig
from emergency_rag.retrieval.rerank.zero_entropy import ZeroEntropyReranker, ZeroEntropyRerankerConfig
from emergency_rag.retrieval.retrievers.bm25 import BM25Retriever, BM25RetrieverConfig
from emergency_rag.retrieval.retrievers.bm25_backend import Tokenizer, bm25s


def build_bm25_rule_search_pipeline(
    *,
    index: bm25s.BM25,
    tokenizer: Tokenizer,
    unit_ids: list[str],
    units: dict[str, SearchUnit],
    rerank_client: ZeroEntropy,
    rerank_model: str,
) -> RetrievalPipeline:
    """应用自行决定组件组合；资源及 SDK 的创建和生命周期由应用拥有。"""
    return RetrievalPipeline(
        retrievers=(
            BM25Retriever(
                index=index,
                tokenizer=tokenizer,
                unit_ids=unit_ids,
                units=units,
                config=BM25RetrieverConfig(top_k=30),
            ),
        ),
        fusion=RRFFusion(RRFFusionConfig(max_candidates=30)),
        reranker=ZeroEntropyReranker(
            client=rerank_client,
            config=ZeroEntropyRerankerConfig(model=rerank_model, batch_size=64),
        ),
        config=RetrievalPipelineConfig(final_top_k=10),
    )

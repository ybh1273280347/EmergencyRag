"""一次完整实验：选择组件，准备 Dataset，检索并将最终候选用于作答。"""

from pathlib import Path

from emergency_rag.chunking.rule import RuleChunker
from emergency_rag.clients.chat import ChatClient
from emergency_rag.clients.embedding import EmbeddingClient
from emergency_rag.data.pipeline import DatasetPipeline
from emergency_rag.retrieval.fusion.rrf import RRFFusion
from emergency_rag.retrieval.models import RetrievalResult
from emergency_rag.retrieval.pipeline import RetrievalPipeline
from emergency_rag.retrieval.rerank.base import Reranker
from emergency_rag.retrieval.retrievers.bm25 import BM25Retriever
from emergency_rag.retrieval.retrievers.dense import DenseRetriever
from emergency_rag.retrieval.tokenizer.jieba import JiebaTokenizer


def run_experiment(
    *,
    embedding: EmbeddingClient,
    chat: ChatClient,
    chat_model: str,
    question: str,
    instructions: str,
    top_k: int = 10,
    source: Path = Path("datasets/初赛规则集rules1.json"),
    index_root: Path = Path("data/indexes"),
    dataset_name: str = "preliminary",
) -> tuple[RetrievalResult, str]:
    # 实验配置集中在这里；Dataset 准备自动完成构建或缓存复用。
    dataset = DatasetPipeline(
        embedding=embedding,
        index_root=index_root,
        chunker=RuleChunker(),
        tokenizer=JiebaTokenizer(),
    ).prepare(source, dataset_name=dataset_name)
    pipeline = RetrievalPipeline(
        dataset=dataset,
        retrievers=(BM25Retriever(top_k=30), DenseRetriever(embedding=embedding, top_k=30)),
        fusion=RRFFusion(k=60, top_k=60),
        reranker=Reranker(),
    )

    result = pipeline.retrieve(question, top_k=top_k)
    # 作答用例只传最终候选；题型、提示词和答案格式由实验指定。
    context = "\n\n".join(
        f"[规则 {candidate.rule_id}]\n{candidate.text}"
        for candidate in result.candidates
    )
    answer = chat.complete(
        model=chat_model,
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": f"参考规则：\n{context}\n\n问题：\n{question}"},
        ],
    )
    return result, answer

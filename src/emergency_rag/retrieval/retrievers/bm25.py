from emergency_rag.data.indexed_dataset import IndexedDataset
from emergency_rag.retrieval.models import Candidate

from .base import Retriever


class BM25Retriever(Retriever):
    """查询 Dataset 持有的离线 BM25 索引，组件只拥有召回预算。"""

    name = "bm25"

    def __init__(self, *, top_k: int = 30) -> None:
        self.top_k = top_k

    def retrieve(self, query: str, dataset: IndexedDataset) -> list[Candidate]:
        index, tokenizer, unit_ids = dataset.load_bm25()

        # 查询只分词，不更新离线词表或重建索引
        query_tokens = tokenizer.tokenize(
            [query],
            update_vocab=False,
            allow_empty=False,
            show_progress=False,
        )
        if not query_tokens[0] or not unit_ids:
            return []

        # 召回预算受索引规模约束，避免请求超过候选总数
        rows, scores = index.retrieve(
            query_tokens,
            k=min(self.top_k, len(unit_ids)),
            show_progress=False,
        )

        # 统一按分数降序、unit_id 升序稳定排序
        ranked = sorted(
            zip(rows[0], scores[0]),
            key=lambda item: (-float(item[1]), unit_ids[int(item[0])]),
        )

        candidates = []
        for rank, (row, score) in enumerate(ranked, start=1):
            unit = dataset.units[unit_ids[int(row)]]
            candidates.append(
                Candidate(
                    unit_id=unit.unit_id,
                    rule_id=unit.rule_id,
                    text=unit.text,
                    sources=[self.name],
                    metadata={
                        **unit.metadata,
                        "retrieval": [{
                            "source": self.name,
                            "query": query,
                            "rank": rank,
                            "score": float(score),
                        }],
                    },
                ),
            )

        return candidates
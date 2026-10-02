import faiss
import numpy as np

from emergency_rag.clients.embedding import EmbeddingClient
from emergency_rag.data.models import IndexedDataset
from emergency_rag.retrieval.models import Candidate

from .base import Retriever


class DenseRetriever(Retriever):
    """查询离线 FAISS 索引；在线只计算查询向量。"""

    name = "dense"

    def __init__(self, embedding_client: EmbeddingClient, *, top_k: int = 30) -> None:
        self.embedding = embedding_client
        self.top_k = top_k

    def retrieve(self, query: str, dataset: IndexedDataset) -> list[Candidate]:
        index, unit_ids = dataset.load_dense()
        if not unit_ids:
            return []

        # 文档向量离线时已归一化；在线只需处理查询向量
        vector = np.ascontiguousarray(
            self.embedding.embed_query(query)[None, :],
            dtype=np.float32,
        )
        faiss.normalize_L2(vector)

        # 取 top_k 与索引规模的最小值，避免越界请求
        scores, rows = index.search(vector, min(self.top_k, len(unit_ids)))

        # 分数降序；同分按 unit_id 稳定排序
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
                        # 记录本次召回来源、查询、名次与分数
                        "retrieval": [
                            {
                                "source": self.name,
                                "query": query,
                                "rank": rank,
                                "score": float(score),
                            },
                        ],
                    },
                ),
            )
        return candidates

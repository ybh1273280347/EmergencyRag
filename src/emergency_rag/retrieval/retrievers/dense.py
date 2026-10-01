from dataclasses import dataclass

import faiss

from emergency_rag.clients.embedding import EmbeddingClient
from emergency_rag.data.models import SearchUnit
from emergency_rag.retrieval.models import Candidate

from .base import Retriever
from .vectors import normalize_vectors


@dataclass(frozen=True, slots=True)
class DenseRetrieverConfig:
    top_k: int = 30


class DenseRetriever(Retriever):
    """查询调用方注入的向量索引和 Embedding 客户端。"""

    name = "dense"

    def __init__(
        self,
        index: faiss.IndexFlatIP,
        unit_ids: list[str],
        units: dict[str, SearchUnit],
        embedding: EmbeddingClient,
        config: DenseRetrieverConfig | None = None,
    ) -> None:
        self.index = index
        self.unit_ids = unit_ids
        self.units = units
        self.embedding = embedding
        self.config = config or DenseRetrieverConfig()
        if not isinstance(index, faiss.IndexFlatIP):
            raise ValueError("Dense 索引必须为 FAISS IndexFlatIP")
        if index.ntotal != len(unit_ids) or len(set(unit_ids)) != len(unit_ids):
            raise ValueError("FAISS 行数与映射不匹配或存在重复 ID")
        if any(unit_id not in units for unit_id in unit_ids):
            raise ValueError("FAISS 映射包含知识库中不存在的 SearchUnit")

    def retrieve(self, query: str, top_k: int | None = None) -> list[Candidate]:
        limit = self.config.top_k if top_k is None else top_k
        if type(limit) is not int or limit <= 0:
            raise ValueError("top_k 必须为正整数")
        if not self.unit_ids:
            return []
        vector = normalize_vectors(self.embedding.embed(query)[None, :], dimension=self.index.d)
        scores, rows = self.index.search(vector, min(limit, len(self.unit_ids)))
        ranked = sorted(
            zip(rows[0], scores[0]),
            key=lambda item: (-float(item[1]), self.unit_ids[int(item[0])]),
        )
        candidates = []
        for rank, (row, score) in enumerate(ranked, start=1):
            unit = self.units[self.unit_ids[int(row)]]
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

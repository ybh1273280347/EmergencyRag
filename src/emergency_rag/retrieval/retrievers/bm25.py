from dataclasses import dataclass

from emergency_rag.data.models import SearchUnit
from emergency_rag.retrieval.models import Candidate

from .base import Retriever
from .bm25_backend import Tokenizer, bm25s


@dataclass(frozen=True, slots=True)
class BM25RetrieverConfig:
    top_k: int = 30
    k1: float = 1.5
    b: float = 0.75


class BM25Retriever(Retriever):
    """只查询已准备的索引、词表和知识单元；不负责资源加载。"""

    name = "bm25"

    def __init__(
        self,
        index: bm25s.BM25,
        tokenizer: Tokenizer,
        unit_ids: list[str],
        units: dict[str, SearchUnit],
        config: BM25RetrieverConfig | None = None,
    ) -> None:
        self.index = index
        self.tokenizer = tokenizer
        self.unit_ids = unit_ids
        self.units = units
        self.config = config or BM25RetrieverConfig()
        if index.scores["num_docs"] != len(unit_ids) or len(set(unit_ids)) != len(unit_ids):
            raise ValueError("BM25 文档数与映射不匹配或存在重复 ID")
        if any(unit_id not in units for unit_id in unit_ids):
            raise ValueError("BM25 映射包含知识库中不存在的 SearchUnit")

    def retrieve(self, query: str, top_k: int | None = None) -> list[Candidate]:
        limit = self.config.top_k if top_k is None else top_k
        if type(limit) is not int or limit <= 0:
            raise ValueError("top_k 必须为正整数")
        # 查询不能扩充训练词表，否则新 token 会与持久化矩阵脱节。
        tokens = self.tokenizer.tokenize(
            [query], update_vocab=False, allow_empty=False, show_progress=False,
        )
        if not tokens[0] or not self.unit_ids:
            return []
        rows, scores = self.index.retrieve(tokens, k=min(limit, len(self.unit_ids)), show_progress=False)
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

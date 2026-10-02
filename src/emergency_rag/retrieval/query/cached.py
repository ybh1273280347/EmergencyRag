"""缓存可插拔 QueryProcessor 的改写与拆分结果，不改变检索生命周期。"""

from pathlib import Path

from emergency_rag.cache import JsonFileCache
from emergency_rag.retrieval.models import QueryContext

from .base import QueryProcessor


class CachedQueryProcessor(QueryProcessor):
    def __init__(self, processor: QueryProcessor, *, directory: Path, refresh: bool = False) -> None:
        self.processor = processor
        self.name = processor.name
        self.refresh = refresh
        self.cache = JsonFileCache(Path(directory) / "queries.json")

    def process(self, query: str, *, refresh: bool = False) -> QueryContext:
        # 以原始查询作键；不同模型、提示词或拆分算法由调用方选择不同版本目录。
        hit = query in self.cache.data and not (refresh or self.refresh)
        if hit:
            context = QueryContext.model_validate(self.cache.data[query])
        else:
            context = self.processor.process(query)

        if context.original_query != query or any(not text.strip() for text in context.queries):
            raise ValueError("QueryProcessor 必须保留原查询并返回非空检索文本")
        if not hit:
            self.cache.save({**self.cache.data, query: context.model_dump(mode="json")})

        # 缓存存储模型快照，不共享可变 sub_queries 列表。
        return context.model_copy(deep=True)

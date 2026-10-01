from abc import ABC

from emergency_rag.retrieval.models import QueryContext


class QueryProcessor(ABC):
    """查询预处理接口，默认保留原查询。"""

    name = "default"

    def process(self, query: str) -> QueryContext:
        return QueryContext(original_query=query, rewritten_query=query)

"""阶段间共享候选；只有重排器可写 final_score，最终排名由 Pipeline 写入。"""

from typing import Any

from pydantic import BaseModel, Field


class QueryContext(BaseModel):
    original_query: str
    rewritten_query: str | None = None
    sub_queries: list[str] = Field(default_factory=list)

    @property
    def queries(self) -> list[str]:
        """先检索改写后的主查询，再检索子查询；未改写时使用原查询。"""
        return [self.rewritten_query if self.rewritten_query is not None else self.original_query, *self.sub_queries]


class Candidate(BaseModel):
    unit_id: str
    rule_id: str
    text: str
    final_rank: int | None = Field(default=None, ge=1)
    final_score: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    sources: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    query: str
    candidates: list[Candidate]
    metadata: dict[str, Any] = Field(default_factory=dict)

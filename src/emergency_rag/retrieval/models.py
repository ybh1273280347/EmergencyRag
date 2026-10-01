"""阶段间共享候选；只有重排器可写 final_score，最终排名由 Pipeline 写入。"""

from typing import Any

from pydantic import BaseModel, Field


class QueryContext(BaseModel):
    original_query: str
    queries: list[str]


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

"""Unit 候选与完整规则证据；重排器评分，Pipeline 聚合并写入最终排名。"""

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
    final_score: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    sources: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RuleEvidence(BaseModel):
    """最终规则证据，原文来自 Dataset，分数为命中单元重排分数的最大值。"""

    rule_id: str
    text: str
    final_score: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    final_rank: int | None = Field(default=None, ge=1)
    sources: list[str] = Field(default_factory=list)
    matched_units: list[Candidate] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    query: str
    evidence: list[RuleEvidence]
    metadata: dict[str, Any] = Field(default_factory=dict)

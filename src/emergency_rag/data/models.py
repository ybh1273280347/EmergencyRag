"""规则事实及索引单元；text 始终保存原始规则文本。"""

from typing import Any

from pydantic import BaseModel, Field, field_validator


class Rule(BaseModel):
    rule_id: str
    text: str

    @field_validator("rule_id", "text")
    @classmethod
    def require_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("规则 ID 和文本不能为空")
        return value


class SearchUnit(BaseModel):
    unit_id: str
    rule_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Rule(BaseModel):
    rule_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("rule_id", "text")
    @classmethod
    def require_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("规则 ID 和文本不能为空")
        return value


class SearchUnit(BaseModel):
    unit_id: str
    rule_id: str
    text: str          # 单元原文，用于重排和命中记录
    index_text: str    # 离线索引文本，可包含主题、领域等增强内容
    metadata: dict[str, Any] = Field(default_factory=dict)


class UnitBuilder(ABC):
    """规则到检索单元的构建接口，负责单元划分与索引文本增强。"""

    # 策略名用于离线产物目录；不同划分或增强实验应提供不同名称。
    name: str

    @abstractmethod
    def build(self, rules: list[Rule]) -> list[SearchUnit]:
        raise NotImplementedError



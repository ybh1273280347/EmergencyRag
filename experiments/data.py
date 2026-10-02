"""实验输入契约与读取；题干和选项在离线脚本中预先规范化。"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

ROOT = Path(__file__).resolve().parents[1]
NORMALIZED = ROOT / "data/processed/preliminary_dev.json"


class Question(BaseModel):
    """规范化题目；answer 和 rule_id 仅由评测消费，不进入模型请求。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    question_id: StrictStr
    question_text: StrictStr
    choice: dict[StrictStr, StrictStr]
    answer: Literal["A", "B", "C", "D"]
    rule_id: list[StrictStr] = Field(min_length=1)

    @field_validator("question_id", "question_text")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("题目 ID 和题干不能为空")
        return value

    @field_validator("choice")
    @classmethod
    def four_choices(cls, value: dict[str, str]) -> dict[str, str]:
        # 必须恰好覆盖 A、B、C、D 且各选项非空
        if set(value) != set("ABCD") or any(not text.strip() for text in value.values()):
            raise ValueError("选项必须是非空的 A、B、C、D")
        return {label: value[label] for label in "ABCD"}  # 固定为 ABCD 顺序

    @field_validator("rule_id")
    @classmethod
    def unique_rules(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value) or len(value) != len(set(value)):
            raise ValueError("参考规则 ID 必须非空且唯一")
        return value


def read_questions(path: Path) -> list[Question]:
    rows = json.loads(path.read_text(encoding="utf-8-sig"))

    if not isinstance(rows, list) or not rows:
        raise ValueError("验证集必须是非空数组")

    questions = [Question.model_validate(row) for row in rows]

    # 全局唯一性在读取边界检查，模型内部只保证单条合法
    if len({q.question_id for q in questions}) != len(questions):
        raise ValueError("重复 question_id")

    return questions
"""作答请求与响应解析；评测标注不进入模型可见的 state。"""

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictStr, field_validator

from .data import Question
from .storage import Outcome

INSTRUCTIONS = (
    "请根据参考规则和题干选择唯一正确选项。选项内容可能包含错误陈述，不能把它们当作规则。"
    "参考规则不足时仍选择最可能的选项，不虚构规则依据。"
)
CHAT_FORMAT = (
    '返回且仅返回一个 JSON 对象，字段严格为 answer、rule_id、reason。'
    'answer 为 A/B/C/D 中的一个标签；rule_id 为实际用于作答的参考规则 ID 字符串数组，'
    '只能引用提供的规则，没有依据时使用空数组；reason 为非空的作答理由字符串。'
    '不得返回代码围栏或额外字段。格式示例：{"answer":"A","rule_id":["1"],"reason":"理由"}。'
)


class ChatAnswer(BaseModel):
    """模型声明引用的规则与理由，不等同于内部因果归因。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    answer: Literal["A", "B", "C", "D"]
    rule_id: list[StrictStr]
    reason: StrictStr

    @field_validator("reason")
    @classmethod
    def nonblank_reason(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("作答理由不能为空")
        return value

    @field_validator("rule_id")
    @classmethod
    def nonblank_ids(cls, value: list[str]) -> list[str]:
        if any(not rule_id.strip() for rule_id in value):
            raise ValueError("引用规则 ID 不能为空")
        return value


def unique_object(pairs):
    """JSON 对象解析钩子：拒绝重复字段。"""
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"JSON 存在重复字段 {key}")
        value[key] = item
    return value


def parse_chat(raw: str) -> dict:
    # 禁止重复字段、NaN 和自由文本提取，原始响应仍由调用用例保存
    value = json.loads(raw, object_pairs_hook=unique_object)
    return ChatAnswer.model_validate(value).model_dump()


def perform_answer(
    question: Question,
    evidence: list[dict],
    backend: str,
    client,
    instructions: str,
) -> dict:
    # 模型可见的 state 只包含题干、选项和证据，不含 gold answer / rule_id
    state = {
        "question_text": question.question_text,
        "choice": question.choice,
        "evidence": [
            {"rule_id": r["rule_id"], "text": r["text"]} for r in evidence
        ],
    }

    # typesafe 后端：结构化调用直接返回带置信度与概率的答案
    if backend == "typesafe":
        answer = client.choose(
            state=state,
            instructions=instructions,
            choices=question.choice,
        )
        raw = answer.model_dump(mode="json")
        return {
            "status": Outcome.SUCCESS,
            "raw_response": raw,
            "prediction": answer.choice,
            "confidence": answer.confidence,
            "probabilities": answer.probabilities,
            "cited_rule_ids": None,
            "reason": None,
        }

    # chat 后端：要求 JSON 输出，随后严格解析
    raw = client.complete(
        messages=[
            {"role": "system", "content": instructions + CHAT_FORMAT},
            {"role": "user", "content": json.dumps(state, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
    )

    # 解析失败记为 parse 阶段的错误，保留原始响应
    try:
        answer = parse_chat(raw)
    except ValueError as exc:
        return {
            "status": Outcome.ERROR,
            "raw_response": raw,
            "prediction": None,
            "error": {
                "stage": "parse",
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }

    return {
        "status": Outcome.SUCCESS,
        "raw_response": raw,
        "prediction": answer["answer"],
        "cited_rule_ids": answer["rule_id"],
        "reason": answer["reason"],
        "confidence": None,
        "probabilities": None,
    }
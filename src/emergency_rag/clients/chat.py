"""Chat API 调用及 JSON object 输出边界，供后续语义组件使用。"""

import json
from dataclasses import dataclass

from openai import APIError, OpenAI
from openai.types.chat import ChatCompletionMessageParam

@dataclass(frozen=True, slots=True)
class ChatLLMConfig:
    model: str
    temperature: float = 0


class ChatLLMError(RuntimeError):
    pass


class ChatLLM:
    def __init__(self, config: ChatLLMConfig, client: OpenAI) -> None:
        if not config.model or not config.model.strip():
            raise ValueError("Chat 模型名未配置")
        self.model = config.model
        self.config = config
        self.client = client

    def invoke(self, messages: list[ChatCompletionMessageParam]) -> str:
        if not messages:
            raise ValueError("Chat messages 不能为空")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "请仅输出一个合法的 JSON object。"},
                    *messages,
                ],
                temperature=self.config.temperature,
                response_format={"type": "json_object"},
            )
        except APIError as exc:
            raise ChatLLMError(f"Chat API 调用失败：{type(exc).__name__}") from exc
        if not response.choices or response.choices[0].finish_reason != "stop":
            raise ChatLLMError("Chat 响应缺失、被截断或未正常结束")
        content = response.choices[0].message.content
        try:
            parsed = json.loads(content) if content else None
        except json.JSONDecodeError as exc:
            raise ChatLLMError("Chat 响应不是合法 JSON") from exc
        if not isinstance(parsed, dict):
            raise ChatLLMError("Chat 响应必须为 JSON object")
        return content

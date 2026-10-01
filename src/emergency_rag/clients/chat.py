"""通用 Chat Completions 文本客户端，作答规则由调用方提供。"""

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.completion_create_params import ResponseFormat


class ChatClient:
    """只负责一次 Chat Completions 请求与文本响应边界校验。

    业务提示词、结构化响应解析和请求节流属于调用用例。
    超时与重试由注入的 SDK 客户端配置。
    """

    def __init__(self, client: OpenAI) -> None:
        self._client = client

    def complete(
        self,
        *,
        model: str,
        messages: list[ChatCompletionMessageParam],
        max_tokens: int | None = None,
        response_format: ResponseFormat | None = None,
    ) -> str:
        request: dict[str, object] = {"model": model, "messages": messages}
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        if response_format is not None:
            request["response_format"] = response_format

        response = self._client.chat.completions.create(**request)
        if not response.choices:
            raise ValueError("chat completion response is empty")
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise ValueError("chat completion response is empty")
        return content

"""通用 Chat Completions 文本客户端，作答规则由调用方提供。"""

from functools import cache

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.completion_create_params import ResponseFormat

from emergency_rag.settings import settings


class ChatClient:
    """只负责一次 Chat Completions 请求与文本响应边界校验。

    业务提示词、结构化响应解析和请求节流属于调用用例。
    超时与重试由注入的 SDK 客户端配置。
    """

    def __init__(self, client: OpenAI, *, model: str) -> None:
        self._client = client
        self.model = model

    def complete(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        max_tokens: int | None = None,
        response_format: ResponseFormat | None = None,
    ) -> str:
        # 仅按需附带可选参数，避免向 SDK 传 None
        request: dict[str, object] = {"model": self.model, "messages": messages}
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


@cache
def get_chat_client() -> ChatClient:
    """共享作答 SDK；检索加载不调用它，提示词与返回格式仍属于用例。"""
    api_key = settings.chat_api_key
    model = settings.chat_model
    if not api_key or not model:
        raise ValueError("请配置 RAG_CHAT_API_KEY 和 settings.chat_model")

    return ChatClient(OpenAI(
        api_key=api_key,
        base_url=settings.chat_base_url,
        timeout=settings.chat_timeout,
        max_retries=settings.chat_max_retries,
    ), model=model)

"""TypeSafe 选择题请求边界；题干、证据和指令由实验用例提供。"""

from functools import cache
import math

from emergency_rag.settings import settings
from typesafe_sdk import Choice, ChoiceAnswer, JSONContent, RetryPolicy, TypeSafeClient


class ChoiceQAClient:
    """发送一次单选请求，返回 SDK 答案；不拼业务提示词或计算准确率。"""

    def __init__(self, client: TypeSafeClient, *, model: str) -> None:
        self.client = client
        self.model = model

    def choose(
        self,
        *,
        state: JSONContent,
        instructions: JSONContent,
        choices: dict[str, str],
    ) -> ChoiceAnswer:
        """执行单项选择请求并校验返回结果的完整性与概率范围。"""
        question = Choice(instructions=instructions, criteria=choices)
        response = self.client.system_one(
            model=self.model,
            state=state,
            questions={"answer": question},
        )

        answer = response.answers.get("answer")
        if not isinstance(answer, ChoiceAnswer):
            raise ValueError("TypeSafe 响应缺少 Choice 答案 answer")

        if answer.choice not in choices or set(answer.probabilities) != set(choices):
            raise ValueError("TypeSafe 响应选项与请求不一致")

        values_to_check = [answer.confidence, *answer.probabilities.values()]
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values_to_check):
                raise ValueError("TypeSafe 置信度与概率必须为 [0, 1] 内的有限数值")

        return answer

    @staticmethod
    def _validate_answer(answer: object, choices: dict[str, str]) -> None:
        """校验答案实例身份、选项一致性及概率分布有效性。"""



@cache
def get_choice_qa_client() -> ChoiceQAClient:
    """按需创建并缓存选择题客户端，连接配置统一来自 settings。"""
    if not settings.typesafe_api_key or not settings.typesafe_model:
        raise ValueError("请配置 TYPESAFE_API_KEY 和 settings.typesafe_model")

    client = TypeSafeClient(
        api_key=settings.typesafe_api_key,
        model=settings.typesafe_model,
        base_url=settings.typesafe_base_url,
        timeout=settings.typesafe_timeout,
        retry=RetryPolicy(max_retries=settings.typesafe_max_retries),
    )
    return ChoiceQAClient(client, model=settings.typesafe_model)
"""对已加载的流水线运行一道题，提示词和答案解析由实验负责。"""

from emergency_rag.clients.chat import ChatClient
from emergency_rag.retrieval.models import RetrievalResult
from emergency_rag.retrieval.pipeline import RetrievalPipeline


def run_experiment(
    *,
    pipeline: RetrievalPipeline,
    chat: ChatClient,
    question: str,
    instructions: str,
    top_k: int = 10,
) -> tuple[RetrievalResult, str]:
    # 一次加载后复用于整份验证集，每道题不再重复加载索引或创建模型客户端。
    result = pipeline.retrieve(question, top_k=top_k)
    # 作答用例只传最终完整规则证据；题型、提示词和答案格式由实验指定。
    context = "\n\n".join(
        f"[规则 {evidence.rule_id}]\n{evidence.text}"
        for evidence in result.evidence
    )
    answer = chat.complete(
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": f"参考规则：\n{context}\n\n问题：\n{question}"},
        ],
    )
    return result, answer

"""对已加载的流水线运行一道题，提示词和答案解析由实验负责。"""

import json
from pathlib import Path

from emergency_rag.clients.chat import ChatClient, get_chat_client
from emergency_rag.load_pipeline import load_pipeline
from emergency_rag.retrieval.models import RetrievalResult
from emergency_rag.retrieval.pipeline import RetrievalPipeline


def run_experiment(
    *,
    pipeline: RetrievalPipeline,
    chat: ChatClient,
    question: str,
    instructions: str,
    top_k: int = 3,
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


if __name__ == "__main__":
    # 连接配置来自 settings 和 .env；示例只声明实验问题与作答指令。
    pipeline = load_pipeline(Path(__file__).resolve().parents[1] / "config/baseline.yaml")
    chat = get_chat_client()
    result, answer = run_experiment(
        pipeline=pipeline,
        chat=chat,
        question="危险化学品事故应急结束需要满足哪些条件？",
        instructions=(
            "请仅根据参考规则回答问题，并注明规则编号。"
            "分别说明现场应急处置工作结束的条件和程序，以及应急处置最终结束的程序。"
            "参考规则不足时请明确说明，不要补充未经证据支持的要求。"
        ),
        top_k=3,
    )

    print(json.dumps({
        "question": result.query,
        "evidence": [
            {
                "rule_id": rule.rule_id,
                "text": rule.text,
                "final_rank": rule.final_rank,
                "final_score": rule.final_score,
            }
            for rule in result.evidence
        ],
        "retrieval_metadata": result.metadata,
        "answer": answer,
    }, ensure_ascii=False, indent=2))

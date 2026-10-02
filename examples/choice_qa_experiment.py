"""单选题实验：检索完整规则证据，再通过 TypeSafe / Jev 选择答案。"""

import json
from pathlib import Path

from typesafe_sdk import ChoiceAnswer

from emergency_rag.clients.choice_qa_client import ChoiceQAClient, get_choice_qa_client
from emergency_rag.load_pipeline import load_pipeline
from emergency_rag.retrieval.models import RetrievalResult
from emergency_rag.retrieval.pipeline import RetrievalPipeline


def run_choice_experiment(
    *,
    pipeline: RetrievalPipeline,
    client: ChoiceQAClient,
    question: str,
    choices: dict[str, str],
    instructions: str,
    top_k: int = 3,
) -> tuple[RetrievalResult, ChoiceAnswer]:
    # 验证集循环外创建 Pipeline 和客户端，每道题复用已加载的索引及连接。
    result = pipeline.retrieve(question, top_k=top_k)
    answer = client.choose(
        state={
            "question": question,
            # 只传最终完整规则证据，不传索引增强文本或命中片段。
            "evidence": [
                {"rule_id": rule.rule_id, "text": rule.text}
                for rule in result.evidence
            ],
        },
        instructions=instructions,
        choices=choices,
    )
    return result, answer


if __name__ == "__main__":
    # 作答需 TypeSafe 密钥。
    pipeline = load_pipeline(Path(__file__).resolve().parents[1] / "config/baseline.yaml")
    client = get_choice_qa_client()
    result, answer = run_choice_experiment(
        pipeline=pipeline,
        client=client,
        question="事故现场应急处置工作结束需要谁确认和批准？",
        choices={
            "A": "现场应急救援指挥部",
            "B": "任意救援队员",
            "C": "现场围观人员",
            "D": "事故企业任意员工",
        },
        instructions="仅根据参考规则选择唯一正确选项。",
    )
    print(json.dumps({
        "question": result.query,
        "retrieved_rule_ids": [rule.rule_id for rule in result.evidence],
        "retrieved_rule_texts": [rule.text for rule in result.evidence],
        "answer": answer.model_dump(mode="json"),
        "expected_choice": "A",
        "correct": answer.choice == "A",
    }, ensure_ascii=False, indent=2))

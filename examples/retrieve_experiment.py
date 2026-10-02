"""对已加载的流水线运行一道题，提示词和答案解析由实验负责。"""

import json
from pathlib import Path

from emergency_rag.load_pipeline import load_pipeline


if __name__ == "__main__":
    # 连接配置来自 settings 和 .env；示例只声明实验问题与作答指令。
    pipeline = load_pipeline(Path(__file__).resolve().parents[1] / "config/preliminary/example.yaml")

    # 冷启动
    result = pipeline.retrieve(
        query="危险化学品事故应急结束需要满足哪些条件？",
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
    }, ensure_ascii=False, indent=2))

    # 预热启动
    result = pipeline.retrieve(
        query="危险化学品事故应急结束需要满足哪些条件？",
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
    }, ensure_ascii=False, indent=2))
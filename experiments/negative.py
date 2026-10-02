"""汇总最新尝试中的答错与调用失败，重点呈现作答窗口内的标注规则覆盖。"""

from collections import Counter
from enum import StrEnum
from pathlib import Path

from .storage import Outcome


class FailureType(StrEnum):
    WRONG_ANSWER = "wrong_answer"
    ERROR = "error"


def build_negative(manifest: dict, records: list[dict]) -> dict:
    """负样本只包含已经结束的错误尝试；待完成题目单独由 summary 统计。"""
    samples = []

    for record in records:
        # 答对或仍在等待的题目不进入负样本
        if record["correct"] or record["status"] == Outcome.PENDING:
            continue

        answer = record["answer"] or {}
        retrieval = record["retrieval"]

        # 排名仅覆盖实际保存的最终检索窗口；未出现不等于整个规则库不可检索
        ranked_ids = (
            [item["rule_id"] for item in retrieval["result"]["evidence"]]
            if retrieval
            else []
        )
        ranks = {rule_id: rank for rank, rule_id in enumerate(ranked_ids, 1)}

        # 作答证据优先取 answer 记录（模型真正看到的），否则退回检索窗口截断
        if answer:
            evidence = answer["evidence"]
        elif retrieval:
            evidence = [
                {"rule_id": item["rule_id"], "text": item["text"]}
                for item in retrieval["result"]["evidence"][:manifest["answer_top_k"]]
            ]
        else:
            evidence = []

        samples.append({
            "question_id": record["question_id"],
            "attempt": record["attempt"],
            "failure_type": (
                FailureType.ERROR
                if record["status"] == Outcome.ERROR
                else FailureType.WRONG_ANSWER
            ),
            "question_text": record["question_text"],
            "choice": record["choice"],
            "gold_answer": record["gold_answer"],
            "prediction": record["prediction"],
            "required_rule_ids": record["gold_rule_ids"],
            "coverage": record["coverage"],
            "answer_context": record["answer_context"],
            "retrieved_rule_ids": ranked_ids,
            "required_rule_ranks": {
                rule_id: ranks.get(rule_id)
                for rule_id in record["gold_rule_ids"]
            },
            "coverage_by_top_k": record["coverage_by_top_k"],
            "answer_evidence": evidence,
            "reason": answer.get("reason"),
            "confidence": answer.get("confidence"),
            "cited_rule_ids": answer.get("cited_rule_ids"),
            "error": record["error"],
        })

    # 三个维度的分布统计：覆盖情况、失败类型、命中数
    coverage = Counter(sample["coverage"] for sample in samples)
    types = Counter(sample["failure_type"] for sample in samples)
    hits = Counter(
        (
            sample["answer_context"]["matched_rule_count"],
            sample["answer_context"]["required_rule_count"],
        )
        for sample in samples
    )

    # 缺失规则出现的位置：作答窗口内已找全 / 完整检索中有但窗口外 / 检索结果中没有
    locations = Counter()
    for sample in samples:
        missing = sample["answer_context"]["missing_rule_ids"]
        if sample["coverage"] == "retrieval_error":
            locations["retrieval_error"] += 1
        elif not missing:
            locations["all_in_answer_context"] += 1
        elif all(
            sample["required_rule_ranks"][rule_id] is not None
            for rule_id in missing
        ):
            locations["outside_answer_context"] += 1
        else:
            locations["absent_from_saved_retrieval"] += 1

    return {
        "schema_version": 1,
        "dataset_name": manifest["dataset_name"],
        "pipeline": Path(manifest["pipeline_source"]).name,
        "backend": manifest["backend"],
        "answer_top_k": manifest["answer_top_k"],
        "retrieval_top_k": max(
            manifest["metric_top_k"] + [manifest["answer_top_k"]]
        ),
        "counts": {
            "sample_count": len(samples),
            "wrong_answer": types[FailureType.WRONG_ANSWER],
            "error": types[FailureType.ERROR],
        },
        "coverage_counts": {
            name: coverage[name]
            for name in ("all", "partial", "none", "retrieval_error")
        },
        "hit_count_distribution": [
            {
                "matched_rule_count": matched,
                "required_rule_count": required,
                "question_count": count,
            }
            for (matched, required), count in sorted(hits.items())
        ],
        "missing_rule_locations": {
            name: locations[name]
            for name in (
                "all_in_answer_context",
                "outside_answer_context",
                "absent_from_saved_retrieval",
                "retrieval_error",
            )
        },
        "samples": samples,
    }


def render_negative(negative: dict) -> str:
    """中文报告先索引，再逐题列出命中数、缺失规则、排名与实际作答证据。"""
    counts = negative["counts"]
    k = negative["answer_top_k"]

    # 报告头部：实验元信息与统计口径说明
    lines = [
        "# 失败样本分析（Negative Samples）",
        "",
        f"- 数据集（Dataset）：`{negative['dataset_name']}`",
        f"- 检索配置（Pipeline）：`{negative['pipeline']}`",
        f"- 作答后端（Backend）：`{negative['backend']}`",
        f"- 作答证据窗口（Answer Top-K）：{k}",
        f"- 负样本：{counts['sample_count']} 题，其中答错 {counts['wrong_answer']} 题、检索或作答失败 {counts['error']} 题。",
        "",
        "只统计每题最新已结束尝试；最新尝试尚未完成的题目不列为负样本。",
        "命中数量按验证集标注规则 ID 计算；证据全部命中但答错，不能单凭此报告确定具体原因。",
        f"缺失规则全部已进入完整检索、但位于作答窗口外：{negative['missing_rule_locations']['outside_answer_context']} 题。",
        f"至少一条标注规则未进入已保存检索窗口：{negative['missing_rule_locations']['absent_from_saved_retrieval']} 题。",
        "",
        "## 样本索引（Case Index）",
        "",
        f"| 题号 | 类型 | 标准 → 预测 | Top-{k} 命中 / 所需 | 缺失规则 |",
        "| --- | --- | --- | ---: | --- |",
    ]

    labels = {
        FailureType.WRONG_ANSWER: "答错（Wrong Answer）",
        FailureType.ERROR: "调用或检索失败（Error）",
    }

    # 汇总表：一行一题
    for sample in negative["samples"]:
        context = sample["answer_context"]
        missing = ", ".join(context["missing_rule_ids"]) or "无"
        prediction = sample["prediction"] or "无有效答案"
        lines.append(
            f"| {sample['question_id']} | {labels[sample['failure_type']]} "
            f"| {sample['gold_answer']} → {prediction} "
            f"| {context['matched_rule_count']} / {context['required_rule_count']} "
            f"| {missing} |"
        )

    # 无负样本时直接返回
    if not counts["sample_count"]:
        lines += ["", "当前没有答错或调用失败的题目。", ""]
        return "\n".join(lines)

    # 逐题展开详情
    for sample in negative["samples"]:
        context = sample["answer_context"]

        lines += [
            "",
            f"## 题目 {sample['question_id']}（Question {sample['question_id']}）",
            "",
            f"类型：{labels[sample['failure_type']]}；尝试编号（Attempt）：{sample['attempt']}。",
            "",
            "### 题干与选项（Question and Choices）",
            "",
            sample["question_text"],
            "",
        ]

        for label, text in sample["choice"].items():
            lines.append(f"- **{label}**：{text}")

        lines += [
            "",
            f"标准答案（Gold Answer）：**{sample['gold_answer']}**；预测答案（Prediction）：**{sample['prediction'] or '无有效答案'}**。",
            "",
            f"### 作答窗口规则覆盖（Answer Context Coverage，Top-{k}）",
            "",
            f"- 命中 / 所需标注规则：**{context['matched_rule_count']} / {context['required_rule_count']}**。",
            f"- 实际窗口内规则数：{len(context['rule_ids'])}。",
            f"- 所需标注规则：{', '.join(sample['required_rule_ids'])}。",
            f"- 已命中规则：{', '.join(context['matched_rule_ids']) or '无'}。",
            f"- 缺失规则：{', '.join(context['missing_rule_ids']) or '无'}。",
        ]

        if sample["coverage"] == "retrieval_error":
            lines.append("- 检索失败，未获得可供作答的规则。")

        # 标注规则在完整检索结果中的具体名次
        lines += [
            "",
            "标注规则在完整检索结果中的排名（Gold Rule Ranks）：",
            "",
        ]
        for rule_id, rank in sample["required_rule_ranks"].items():
            detail = (
                f"第 {rank} 名"
                if rank is not None
                else f"未进入已保存的 Top-{negative['retrieval_top_k']}"
            )
            lines.append(f"- 规则 `{rule_id}`：{detail}。")

        # 各检索窗口 K 下的命中情况
        lines += [
            "",
            "| 检索窗口 K | 命中 / 所需 | 缺失标注规则 |",
            "| ---: | ---: | --- |",
        ]
        for cutoff, cov in sample["coverage_by_top_k"].items():
            missing = ", ".join(cov["missing_rule_ids"]) or "无"
            lines.append(
                f"| {cutoff} | {cov['matched_rule_count']} / {cov['required_rule_count']} | {missing} |"
            )

        # 模型理由（如存在）
        if sample["reason"]:
            lines += ["", "### 模型理由（Model Reason）", "", sample["reason"]]

        # 错误详情（如存在）
        if sample["error"]:
            error = sample["error"]
            lines += [
                "",
                "### 失败信息（Error Details）",
                "",
                f"- 阶段（Stage）：`{error.get('stage', 'unknown')}`。",
                f"- 原因（Message）：{error.get('message', '未提供')}。",
            ]

        # 作答窗口内的规则原文，折叠展示
        if sample["answer_evidence"]:
            lines += [
                "",
                "<details>",
                "<summary>查看作答窗口规则原文（Answer Evidence）</summary>",
                "",
            ]
            for rank, evidence in enumerate(sample["answer_evidence"], 1):
                matched = (
                    "命中标注"
                    if evidence["rule_id"] in sample["required_rule_ids"]
                    else "未标注"
                )
                lines += [
                    f"**第 {rank} 条：规则 {evidence['rule_id']}（{matched}）**",
                    "",
                    evidence["text"],
                    "",
                ]
            lines += ["</details>"]

    lines += [
        "",
        "原始响应与所有历史尝试保留在 logs/；机器可读明细见 [negative.json](negative.json)。",
        "",
    ]
    return "\n".join(lines)
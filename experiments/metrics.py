"""从最新尝试计算指标，历史成功答案不会替代新尝试的失败或待处理结果。"""

import json
import math
from collections import Counter
from pathlib import Path

from .data import Question
from .negative import build_negative, render_negative
from .storage import (
    Outcome,
    atomic_bytes,
    atomic_json,
    read_history,
    retrieval_for,
    validate_context,
)


def ratio(numerator: float, denominator: int) -> dict:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
    }


def rule_metrics(gold: list[str], ranked: list[str], k: int) -> dict:
    """按标注规则 ID 计算覆盖与排序质量；逐题计分后取宏平均。"""
    gold_set = set(gold)

    # 按名次去重并截断到 K
    ranked = list(dict.fromkeys(ranked))[:k]
    relevant = [int(rule_id in gold_set) for rule_id in ranked]
    hits = sum(relevant)

    # NDCG：实际 DCG 除以理想 DCG（gold 数不足 K 时只算到实际 gold 数）
    dcg = sum(value / math.log2(rank + 1) for rank, value in enumerate(relevant, 1))
    ideal = sum(
        1 / math.log2(rank + 1)
        for rank in range(1, min(k, len(gold_set)) + 1)
    )

    return {
        "recall": hits / len(gold_set),
        "all_gold_coverage": int(hits == len(gold_set)),
        "ndcg": dcg / ideal,
    }


def rule_coverage(gold: list[str], ranked: list[str], k: int) -> dict:
    """按标注 ID 列出一个窗口中的实际命中与缺失；保持标注和排序顺序。"""
    selected = list(dict.fromkeys(ranked))[:k]
    selected_ids = set(selected)
    matched = [rule_id for rule_id in gold if rule_id in selected_ids]
    missing = [rule_id for rule_id in gold if rule_id not in selected_ids]
    return {
        "top_k": k,
        "rule_ids": selected,
        "matched_rule_count": len(matched),
        "required_rule_count": len(gold),
        "matched_rule_ids": matched,
        "missing_rule_ids": missing,
    }


def build_summary(
    manifest: dict,
    history: dict,
) -> tuple[dict, dict]:
    """生成总体摘要与负样本报告；未完成题目不当作答错或失败。"""
    questions = [Question.model_validate(row) for row in manifest["questions"]]
    validate_context(manifest["questions"], history, manifest["answer_top_k"])

    # 逐题取最新尝试，并把状态归为 success / error / pending 三类
    records = []
    for question in questions:
        attempts = history.get(question.question_id, {})
        latest = max(attempts) if attempts else None
        attempt = attempts.get(latest)
        retrieval = retrieval_for(attempts, latest) if attempt else None
        answer = attempt["answer"] if attempt else None
        retrieval_error = (
            attempt["retrieval"] if attempt and attempt["retrieval"] else None
        )

        if answer is not None:
            status = answer["status"]
        elif retrieval_error and retrieval_error["status"] == Outcome.ERROR:
            status = Outcome.ERROR
        else:
            status = Outcome.PENDING

        evidence = retrieval["result"]["evidence"] if retrieval else []

        # answer 记录保存模型真正看到的证据，不能用最新其他尝试的证据推算
        supplied = (
            answer["evidence"]
            if answer
            else evidence[:manifest["answer_top_k"]]
        )
        ranked_ids = [r["rule_id"] for r in evidence]
        context = rule_coverage(question.rule_id, [r["rule_id"] for r in supplied], manifest["answer_top_k"])
        hits = context["matched_rule_ids"]
        coverage = (
            "all" if len(hits) == len(question.rule_id)
            else "partial" if hits
            else "none"
        )
        if retrieval is None:
            coverage = "retrieval_error" if status == Outcome.ERROR else "pending"

        prediction = answer.get("prediction") if answer else None
        records.append({
            "question_id": question.question_id,
            "attempt": latest,
            "status": status,
            "question_text": question.question_text,
            "choice": question.choice,
            "gold_answer": question.answer,
            "prediction": prediction,
            "correct": status == Outcome.SUCCESS and prediction == question.answer,
            "gold_rule_ids": question.rule_id,
            "coverage": coverage,
            "answer_context": context,
            "coverage_by_top_k": {
                str(k): rule_coverage(question.rule_id, ranked_ids, k)
                for k in sorted(set(manifest["metric_top_k"]) | {manifest["answer_top_k"]})
            },
            "retrieval": retrieval,
            "answer": answer,
            "error": (answer or {}).get("error")
                     or (retrieval_error or {}).get("error"),
        })

    # 各类计数与子集划分
    counts = Counter(r["status"] for r in records)
    processed = [r for r in records if r["status"] != Outcome.PENDING]
    valid = [r for r in records if r["status"] == Outcome.SUCCESS]
    complete = not counts[Outcome.PENDING]
    correct = sum(r["correct"] for r in valid)
    retrieval_processed = [
        r for r in records if r["retrieval"] or r["coverage"] == "retrieval_error"
    ]

    # 各 K 下的检索指标，逐题计算后再聚合
    retrieval_metrics = {}
    for k in sorted(set(manifest["metric_top_k"]) | {manifest["answer_top_k"]}):
        per_question = [
            rule_metrics(
                r["gold_rule_ids"],
                (
                    [e["rule_id"] for e in r["retrieval"]["result"]["evidence"]]
                    if r["retrieval"]
                    else []
                ),
                k,
            )
            for r in retrieval_processed
        ]
        retrieval_metrics[str(k)] = {
            name: ratio(sum(item[name] for item in per_question), len(per_question))
            for name in (
                "recall",
                "all_gold_coverage",
                "ndcg",
            )
        }

    # 引用只用于发现不在供给证据中的 ID，不能据此判定理由是否忠实。
    cited_samples = [
        r for r in valid if r["answer"].get("cited_rule_ids") is not None
    ]
    unsupported_citations = {
        r["question_id"]: sorted(
            set(r["answer"]["cited_rule_ids"])
            - {e["rule_id"] for e in r["answer"]["evidence"]}
        )
        for r in cited_samples
    }
    unsupported_citations = {
        question_id: ids for question_id, ids in unsupported_citations.items() if ids
    }

    negative = build_negative(manifest, records)
    coverage_counts = Counter(r["coverage"] for r in processed)
    hit_counts = Counter(
        (r["answer_context"]["matched_rule_count"], r["answer_context"]["required_rule_count"])
        for r in processed
    )

    summary = {
        "schema_version": 1,
        "dataset_name": manifest["dataset_name"],
        "pipeline": Path(manifest["pipeline_source"]).name,
        "backend": manifest["backend"],
        "answer_top_k": manifest["answer_top_k"],
        "retrieval_top_k": max(manifest["metric_top_k"] + [manifest["answer_top_k"]]),
        "complete": complete,
        "metric_scope": "final" if complete else "interim",
        "counts": {
            "planned": len(records),
            "processed": len(processed),
            "successful": len(valid),
            "correct": correct,
            "wrong_answer": len(valid) - correct,
            "failed": counts[Outcome.ERROR],
            "pending": counts["pending"],
            "attempts": sum(len(a) for a in history.values()),
        },
        "accuracy": {
            # 端到端准确率只在完整实验时发布；阶段准确率始终可用
            "end_to_end": ratio(correct, len(records)) if complete else None,
            "interim_processed": ratio(correct, len(processed)),
        },
        "retrieval": retrieval_metrics,
        "diagnostics": {
            "citation_sample_count": len(cited_samples),
            "unsupported_citations": unsupported_citations,
        },
        "answer_context": {
            "top_k": manifest["answer_top_k"],
            "sample_count": len(processed),
            "coverage_counts": {
                name: coverage_counts[name] for name in ("all", "partial", "none", "retrieval_error")
            },
            "hit_count_distribution": [
                {"matched_rule_count": matched, "required_rule_count": required, "question_count": count}
                for (matched, required), count in sorted(hit_counts.items())
            ],
        },
        "negative": {
            **negative["counts"],
            "coverage_counts": negative["coverage_counts"],
            "hit_count_distribution": negative["hit_count_distribution"],
            "missing_rule_locations": negative["missing_rule_locations"],
            "question_ids": [sample["question_id"] for sample in negative["samples"]],
            "json_file": "negative.json",
            "markdown_file": "negative.md",
        },
        "errors": dict(Counter(
            (r["error"] or {}).get("stage", "unknown")
            for r in records
            if r["status"] == Outcome.ERROR
        )),
    }
    return summary, negative


def summarize(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))

    summary, negative = build_summary(manifest, read_history(directory))
    # 四份报告均从日志与检查点重建；原始响应只保留在具体日志中。
    atomic_json(directory / "summary.json", summary)
    atomic_json(directory / "negative.json", negative)
    atomic_bytes(directory / "summary.md", render_report(manifest, summary).encode("utf-8"))
    atomic_bytes(directory / "negative.md", render_negative(negative).encode("utf-8"))
    return summary


def render_report(manifest: dict, summary: dict) -> str:
    """Markdown 使用中文解释与英文术语；机器字段和分母细节留在 JSON。"""
    counts = summary["counts"]
    answer_k = str(manifest["answer_top_k"])
    status = "完整（Complete）" if summary["complete"] else "未完成，以下为阶段指标（Interim）"
    lines = [
        "# 选择题实验评估（MCQA Evaluation）",
        "",
        f"- 数据集（Dataset）：`{manifest['dataset_name']}`",
        f"- 检索配置（Pipeline）：`{Path(manifest['pipeline_source']).name}`",
        f"- 作答后端（Backend）：`{manifest['backend']}`",
        f"- 状态（Status）：{status}",
        f"- 作答证据数量（Context Top-K）：{answer_k}",
        "",
        "## 样本进度（Sample Progress）",
        "",
        "| 项目 | 数量 |",
        "| --- | ---: |",
        f"| 计划题数（Planned） | {counts['planned']} |",
        f"| 已处理题数（Processed） | {counts['processed']} |",
        f"| 有效作答题数（Successful） | {counts['successful']} |",
        f"| 正确题数（Correct） | {counts['correct']} |",
        f"| 答错题数（Wrong Answer） | {counts['wrong_answer']} |",
        f"| 检索或作答失败题数（Error） | {counts['failed']} |",
        f"| 待完成题数（Pending） | {counts['pending']} |",
        f"| 累计尝试次数（Attempts） | {counts['attempts']} |",
        "",
        "## 核心指标（Core Metrics）",
        "",
        "| 指标 | 结果 | 统计口径 |",
        "| --- | ---: | --- |",
    ]
    accuracy = summary["accuracy"]["end_to_end" if summary["complete"] else "interim_processed"]
    accuracy_name = "答案准确率（Accuracy）" if summary["complete"] else "阶段答案准确率（Interim Accuracy）"
    value = f"{accuracy['value']:.2%}" if accuracy["value"] is not None else "暂无数据"
    lines.append(f"| {accuracy_name} | {value} | {int(accuracy['numerator'])} / {accuracy['denominator']} 题答对 |")
    retrieval = summary["retrieval"][answer_k]
    names = {
        "recall": "规则召回率（Rule Recall）",
        "all_gold_coverage": "完整规则覆盖率（All-Gold Coverage）",
        "ndcg": "归一化折损累计增益（nDCG）",
    }
    for name, title in names.items():
        metric = retrieval[name]
        value = metric["value"]
        rendered = "暂无数据" if value is None else f"{value:.4f}" if name == "ndcg" else f"{value:.2%}"
        detail = (
            f"{int(metric['numerator'])} / {metric['denominator']} 题找全标注规则"
            if name == "all_gold_coverage" else f"{metric['denominator']} 题，逐题宏平均（Macro Average）"
        )
        lines.append(f"| {title}@{answer_k} | {rendered} | {detail} |")
    lines += [
        "",
        "准确率按 A–D 选项精确匹配（Exact Match）；完整实验的作答失败计错。",
        "规则指标在最终规则排序上按标注 ID 计算，检索失败计零；未处理题目不计入阶段分母。",
    ]
    failures = summary["negative"]
    labels = {"all": "标注规则全部命中（All）", "partial": "标注规则部分命中（Partial）", "none": "未命中标注规则（None）", "retrieval_error": "检索失败（Retrieval Error）"}
    lines += [
        "",
        "## 失败样本与规则覆盖（Negative Samples and Rule Coverage）",
        "",
        f"共 {failures['sample_count']} 个负样本：答错 {failures['wrong_answer']} 题，检索或作答失败 {failures['error']} 题。",
        "",
        "| 作答 Top-K 证据覆盖 | 全部已处理题目 | 负样本 |",
        "| --- | ---: | ---: |",
    ]
    for name, title in labels.items():
        lines.append(f"| {title} | {summary['answer_context']['coverage_counts'][name]} | {failures['coverage_counts'][name]} |")
    lines += [
        "",
        "### 负样本命中数量（Negative Hit Counts）",
        "",
        "| 命中标注规则数 / 所需标注规则数 | 负样本题数 |",
        "| --- | ---: |",
    ]
    for group in failures["hit_count_distribution"]:
        lines.append(f"| {group['matched_rule_count']} / {group['required_rule_count']} | {group['question_count']} |")
    if not failures["sample_count"]:
        lines.append("| — | 0 |")
    lines += ["", "逐题查看题干、选项、预测、缺失规则与各 K 的命中情况：[失败样本报告（Negative Samples）](negative.md)。"]
    locations = failures["missing_rule_locations"]
    lines += [
        "",
        f"- 作答窗口缺失的标注规则全部已进入完整检索，只是排在 Top-{answer_k} 之后：{locations['outside_answer_context']} 题。",
        f"- 至少一条标注规则未进入已保存的 Top-{summary['retrieval_top_k']}：{locations['absent_from_saved_retrieval']} 题。",
        "",
        "上述统计描述检索与作答窗口的关系，不代表增加 K 后这些题必然答对。",
    ]
    lines += [
        "",
        "## 其他检索窗口（Other Top-K）",
        "",
        "| K | 规则召回率（Recall） | 完整规则覆盖率（All-Gold Coverage） | 排序质量（nDCG） |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for k, metrics in summary["retrieval"].items():
        if k == answer_k:
            continue
        cells = []
        for name in names:
            value = metrics[name]["value"]
            cells.append("暂无数据" if value is None else f"{value:.4f}" if name == "ndcg" else f"{value:.2%}")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    if summary["errors"]:
        labels = {"retrieval": "检索", "api": "作答接口", "parse": "答案解析", "execution": "实验执行", "unknown": "未知"}
        lines += ["", "## 错误诊断（Error Diagnostics）", ""]
        for stage, count in summary["errors"].items():
            lines.append(f"- {labels.get(stage, stage)}（{stage}）：{count} 题。")
    unsupported = summary["diagnostics"]["unsupported_citations"]
    if unsupported:
        lines += ["", f"引用异常（Unsupported Citations）：{len(unsupported)} 题引用了未供给的规则 ID，明细见 summary.json。"]
    lines += [
        "",
        "## 结果文件与解释（Artifacts and Interpretation）",
        "",
        "- [结构化摘要（JSON Summary）](summary.json)：机器可读的指标、分母、命中分布与失败统计。",
        "- [负样本数据（Negative JSON）](negative.json)：机器可读的答错与调用失败明细。",
        "- [负样本报告（Negative Markdown）](negative.md)：人类可读的规则缺失与作答分析。",
        "- [完整日志（JSONL Logs）](logs/)：每题每次尝试一条记录，包含启动信息、检索结果和原始回答。",
        "- [恢复检查点（Checkpoint）](checkpoint.json)：仅包含尚未完成的尝试。",
        "",
        "标注规则可能未穷尽其他有效证据；完整覆盖只表示找全标注，不证明模型实际使用了这些规则。",
        "引用 ID 检查不能替代理由忠实性（Faithfulness）评估。",
        "",
    ]
    return "\n".join(lines)

"""一次性拆分与修正初赛验证集，生成供实验直接读取的规范化文件。"""

import argparse
import json
import re
from pathlib import Path

from experiments.data import NORMALIZED, ROOT, Question
from experiments.storage import atomic_json

RAW_QUESTIONS = ROOT / "data/raw/初赛验证集dev.json"
RAW_RULES = ROOT / "data/raw/初赛规则集rules1.json"


def normalize(source: Path = RAW_QUESTIONS, rules: Path = RAW_RULES,
              output: Path = NORMALIZED) -> dict:
    """全量校验后落盘；已知标签修正同时写入报告，不改原始语料。"""
    rows = json.loads(source.read_text(encoding="utf-8-sig"))
    rule_ids = {r["rule_id"] for r in json.loads(rules.read_text(encoding="utf-8-sig"))}
    if not isinstance(rows, list) or not rows:
        raise ValueError("验证集必须是非空数组")

    questions = []
    corrections = []
    seen = set()
    for row in rows:
        qid = row.get("question_id")
        try:
            text = row["question_text"].strip()
            changes = []
            # 仅修正已经审阅过的格式错误；已修好的输入无需再次改动。
            if qid == "27" and "\n仅向参与者发送预警信息" in text:
                text = text.replace("\n仅向参与者发送预警信息", "\nD. 仅向参与者发送预警信息", 1)
                changes.append("补齐末项 D 标签")
            if qid == "286" and "A. A. " in text:
                text = text.replace("A. A. ", "A. ", 1)
                changes.append("去除重复 A 标签")

            if "选择：" in text:
                stem, options = text.split("选择：", 1)
            else:
                marker = re.search(r"A[.．]", text)
                if marker is None:
                    raise ValueError("找不到选项区")
                stem, options = text[:marker.start()], text[marker.start():]
                changes.append("从 A 标签定位缺少分隔符的选项区")

            # 顿号会出现在“河流A、”等正文中，不作为选项标签分隔符。
            markers = list(re.finditer(r"([A-D])[.．]", options))
            if [m.group(1) for m in markers] != list("ABCD"):
                raise ValueError("选项标签必须按 A、B、C、D 各出现一次")
            choice = {
                marker.group(1): options[marker.end():
                    markers[i + 1].start() if i + 1 < len(markers) else len(options)
                ].strip()
                for i, marker in enumerate(markers)
            }
            question = Question(
                question_id=qid, question_text=stem.removeprefix("问题：").strip(),
                choice=choice, answer=row["answer"], rule_id=row["rule_id"],
            )
            if qid in seen:
                raise ValueError("重复 question_id")
            if set(question.rule_id) - rule_ids:
                raise ValueError("标注引用了不存在的规则")
            seen.add(qid)
            questions.append(question.model_dump())
            if changes:
                corrections.append({"question_id": qid, "changes": changes,
                                    "original_question_text": row["question_text"]})
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"题目 {qid!r} 无法规范化：{exc}") from exc

    report = {"source": str(source.resolve()), "rules": str(rules.resolve()),
              "count": len(questions), "corrections": corrections}
    atomic_json(output, questions)
    atomic_json(output.with_name(output.stem + ".normalization.json"), report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="一次性拆分题干与选项并修正已知标签错误")
    parser.add_argument("--source", type=Path, default=RAW_QUESTIONS)
    parser.add_argument("--rules", type=Path, default=RAW_RULES)
    parser.add_argument("--output", type=Path, default=NORMALIZED)
    args = parser.parse_args()
    report = normalize(args.source, args.rules, args.output)
    print(json.dumps({"count": report["count"], "output": str(args.output),
                      "corrected_questions": len(report["corrections"])}, ensure_ascii=False))

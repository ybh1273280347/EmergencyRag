"""初赛原始 JSON 的解析与事务入库，一次性工程逻辑留在 scripts。"""

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from time import perf_counter

from emergency_rag.chunking.rule import RuleChunker
from emergency_rag.data.models import Rule


def read_rules(source: Path) -> list[Rule]:
    data = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list) or not data:
        raise ValueError("规则文件必须为非空 JSON 数组")
    rules = []
    ids: set[str] = set()
    for index, row in enumerate(data, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"第 {index} 条规则必须为对象")
        rule = Rule(rule_id=row.get("rule_id"), text=row.get("rule_text"))
        if rule.rule_id in ids:
            raise ValueError(f"重复 rule_id：{rule.rule_id}")
        ids.add(rule.rule_id)
        rules.append(rule)
    return rules


def prepare_rules(source: Path, database: Path, *, overwrite: bool = False) -> dict[str, int | float]:
    started = perf_counter()
    if database.exists() and not overwrite:
        raise FileExistsError("处理后数据库已存在；重建请传入 overwrite=True")
    rules = read_rules(source)
    units = RuleChunker().chunk(rules)
    database.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database)) as db:
        db.execute("PRAGMA foreign_keys = ON")
        # 不使用 executescript：它可能提前提交事务，破坏覆盖失败时的回滚。
        with db:
            db.execute("BEGIN")
            db.execute(
                "CREATE TABLE IF NOT EXISTS rules ("
                "rule_id TEXT PRIMARY KEY, rule_text TEXT NOT NULL)",
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS search_units ("
                "unit_id TEXT PRIMARY KEY, rule_id TEXT NOT NULL, text TEXT NOT NULL, "
                "metadata_json TEXT NOT NULL, FOREIGN KEY(rule_id) REFERENCES rules(rule_id))",
            )
            db.execute("DELETE FROM search_units")
            db.execute("DELETE FROM rules")
            db.executemany(
                "INSERT INTO rules (rule_id, rule_text) VALUES (?, ?)",
                [(rule.rule_id, rule.text) for rule in rules],
            )
            db.executemany(
                "INSERT INTO search_units (unit_id, rule_id, text, metadata_json) VALUES (?, ?, ?, ?)",
                [
                    (unit.unit_id, unit.rule_id, unit.text, json.dumps(unit.metadata, ensure_ascii=False))
                    for unit in units
                ],
            )
    return {"rules": len(rules), "search_units": len(units), "elapsed_seconds": perf_counter() - started}


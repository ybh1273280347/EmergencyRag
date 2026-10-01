"""只读取处理后知识库；原始数据解析和写入由离线脚本拥有。"""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from .models import Rule, SearchUnit


class RuleRepository:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()

    def load_rules(self) -> list[Rule]:
        with closing(sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)) as db:
            rows = db.execute("SELECT rule_id, rule_text FROM rules ORDER BY rule_id")
            return [Rule(rule_id=row[0], text=row[1]) for row in rows]

    def load_search_units(self) -> list[SearchUnit]:
        with closing(sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)) as db:
            rows = db.execute(
                "SELECT unit_id, rule_id, text, metadata_json "
                "FROM search_units ORDER BY unit_id",
            )
            return [
                SearchUnit(
                    unit_id=row[0],
                    rule_id=row[1],
                    text=row[2],
                    metadata=json.loads(row[3] or "{}"),
                )
                for row in rows
            ]

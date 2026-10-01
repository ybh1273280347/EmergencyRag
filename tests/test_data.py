import json
import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

from emergency_rag.chunking.rule import RuleChunker
from emergency_rag.data.models import Rule, SearchUnit
from emergency_rag.data.repository import RuleRepository
from emergency_rag.retrieval.models import Candidate
from scripts.prepare_rules import prepare_rules, read_rules

ROOT = Path(__file__).resolve().parents[1]


def test_real_rules_are_preserved(database):
    source = ROOT / "datasets/初赛规则集rules1.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    stats = prepare_rules(source, database)
    assert stats["rules"] == stats["search_units"] == 800
    repository = RuleRepository(database)
    rules = {rule.rule_id: rule for rule in repository.load_rules()}
    units = {unit.rule_id: unit for unit in repository.load_search_units()}
    for row in raw:
        assert rules[row["rule_id"]].text == row["rule_text"]
        assert units[row["rule_id"]].text == row["rule_text"]
        assert units[row["rule_id"]].unit_id == f"rule:{row['rule_id']}"


@pytest.mark.parametrize("rows", [
    {}, [], [1], [{"rule_id": "1", "rule_text": "  "}],
    [{"rule_id": "", "rule_text": "规则"}],
    [{"rule_id": 1, "rule_text": "规则"}],
    [{"rule_id": "1", "rule_text": "甲"}, {"rule_id": "1", "rule_text": "乙"}],
])
def test_invalid_input_rejected(tmp_path, rows):
    source = tmp_path / "rules.json"
    source.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError):
        read_rules(source)


def test_chunker_one_to_one_and_independent_defaults():
    rules = [Rule(rule_id="01", text="  保留文本 \n"), Rule(rule_id="2", text="规则乙")]
    units = RuleChunker().chunk(rules)
    assert [(unit.unit_id, unit.rule_id, unit.text) for unit in units] == [
        ("rule:01", "01", rules[0].text), ("rule:2", "2", rules[1].text),
    ]
    units[0].metadata["x"] = 1
    assert not units[1].metadata
    a = Candidate(unit_id="a", rule_id="1", text="甲")
    b = Candidate(unit_id="b", rule_id="2", text="乙")
    a.sources.append("bm25")
    a.metadata["x"] = 1
    assert b.sources == [] and b.metadata == {}
    assert a.final_score is None and a.final_rank is None


@pytest.mark.parametrize("fields", [{"final_rank": 0}, {"final_score": 1.1}, {"final_score": float("nan")}])
def test_candidate_final_contract(fields):
    with pytest.raises(ValidationError):
        Candidate(unit_id="a", rule_id="1", text="甲", **fields)


def test_overwrite_and_transaction_rollback(database, tmp_path, monkeypatch):
    source = tmp_path / "rules.json"
    source.write_text('[{"rule_id":"1","rule_text":"原文"}]', encoding="utf-8")
    prepare_rules(source, database)
    with pytest.raises(FileExistsError):
        prepare_rules(source, database)
    source.write_text('[{"rule_id":"2","rule_text":"新文"}]', encoding="utf-8")

    # 在删除旧记录后的写入阶段制造真实外键错误，证明整批回滚。
    with monkeypatch.context() as patch:
        patch.setattr(RuleChunker, "chunk", lambda self, records: [
            SearchUnit(unit_id="bad", rule_id="missing", text="错误"),
        ])
        with pytest.raises(sqlite3.IntegrityError):
            prepare_rules(source, database, overwrite=True)
    assert RuleRepository(database).load_rules() == [Rule(rule_id="1", text="原文")]
    assert RuleRepository(database).load_search_units()[0].unit_id == "rule:1"
    prepare_rules(source, database, overwrite=True)
    assert RuleRepository(database).load_rules() == [Rule(rule_id="2", text="新文")]


def test_missing_database_read_does_not_create_file(tmp_path):
    database = tmp_path / "missing.db"
    with pytest.raises(sqlite3.OperationalError):
        RuleRepository(database).load_search_units()
    assert not database.exists()

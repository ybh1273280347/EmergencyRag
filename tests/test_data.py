import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from emergency_rag.unit_building.rule import RuleUnitBuilder
from emergency_rag.unit_building.base import UnitBuilder
from emergency_rag.data.models import Rule, SearchUnit, dataset_index_directory
from emergency_rag.data.pipeline import DatasetPipeline
from emergency_rag.retrieval.models import Candidate, RuleEvidence
from emergency_rag.data.pipeline import read_rules

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def stub_index_build(monkeypatch):
    # 本文件检查读取与单元契约；真实双路构建由 test_indexes 覆盖。
    def build(units, index_root, *, rules, tokenizer, embedding, overwrite):
        directory = dataset_index_directory(
            index_root, units[0].metadata["dataset"], units[0].metadata["unit_builder"]["strategy"], tokenizer.name,
        )
        directory.mkdir(parents=True)
        (directory / "rules.json").write_text(json.dumps([rule.model_dump() for rule in rules]), encoding="utf-8")
        (directory / "units.json").write_text(json.dumps([unit.model_dump() for unit in units]), encoding="utf-8")
        return directory

    monkeypatch.setattr("emergency_rag.data.pipeline.build_dataset_indexes", build)


def test_real_rules_are_preserved():
    source = ROOT / "datasets/初赛规则集rules1.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    rules = read_rules(source)
    units = RuleUnitBuilder().build(rules)
    assert len(rules) == len(units) == 800
    # 默认构建直接保留原规则 ID 和文本，不通过额外数据库中转。
    for row, rule, unit in zip(raw, rules, units, strict=True):
        assert rule.rule_id == unit.rule_id == row["rule_id"]
        assert rule.text == unit.text == row["rule_text"]
        assert unit.index_text == rule.text
        assert unit.unit_id == f"rule:{row['rule_id']}"


@pytest.mark.parametrize("rows", [
    {}, [], [1], [{"rule_id": "1", "rule_text": "  "}],
    [{"rule_id": "", "rule_text": "规则"}],
    [{"rule_id": 1, "rule_text": "规则"}],
    [{"rule_id": "1", "rule_text": "甲"}, {"rule_id": "1", "rule_text": "乙"}],
    [{"rule_id": "1", "rule_text": "甲", "metadata": []}],
])
def test_invalid_input_rejected(tmp_path, rows):
    source = tmp_path / "rules.json"
    source.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError):
        read_rules(source)


def test_unit_builder_one_to_one_and_independent_defaults():
    rules = [Rule(rule_id="01", text="  保留文本 \n"), Rule(rule_id="2", text="规则乙")]
    units = RuleUnitBuilder().build(rules)
    assert [(unit.unit_id, unit.rule_id, unit.text) for unit in units] == [
        ("rule:01", "01", rules[0].text), ("rule:2", "2", rules[1].text),
    ]
    units[0].metadata["x"] = 1
    assert not units[1].metadata
    rules[0].metadata["domain"] = "海洋"
    assert not rules[1].metadata
    a = Candidate(unit_id="a", rule_id="1", text="甲")
    b = Candidate(unit_id="b", rule_id="2", text="乙")
    a.sources.append("bm25")
    a.metadata["x"] = 1
    assert b.sources == [] and b.metadata == {}
    assert a.final_score is None
    assert "final_rank" not in Candidate.model_fields


@pytest.mark.parametrize("fields", [{"final_score": 1.1}, {"final_score": float("nan")}])
def test_candidate_final_contract(fields):
    with pytest.raises(ValidationError):
        Candidate(unit_id="a", rule_id="1", text="甲", **fields)


@pytest.mark.parametrize("fields", [{"final_rank": 0}, {"final_score": 1.1}, {"final_score": float("nan")}])
def test_rule_evidence_final_contract(fields):
    with pytest.raises(ValidationError):
        RuleEvidence(rule_id="1", text="完整规则", **fields)


@pytest.mark.parametrize("dataset_name", ["preliminary", "semifinal"])
def test_dataset_pipeline_records_provenance(dataset_name, tmp_path, stub_index_build):
    source = ROOT / "datasets/初赛规则集rules1.json"
    dataset = DatasetPipeline(embedding=Mock(), index_root=tmp_path).prepare(source, dataset_name=dataset_name)
    units = list(dataset.units.values())
    assert len(units) == 800
    assert len(dataset.rules) == 800
    assert all(unit.metadata == {"dataset": dataset_name, "unit_builder": {"strategy": "rule"}} for unit in units)
    units[0].metadata["unit_builder"]["strategy"] = "changed"
    assert units[1].metadata["unit_builder"]["strategy"] == "rule"


def test_dataset_pipeline_preserves_builder_metadata_and_output(tmp_path, stub_index_build):
    output = SearchUnit(
        unit_id="1:1", rule_id="1", text="子规则文本", index_text="海洋 子规则文本",
        metadata={"offset": 8, "topic": "单元主题"},
    )
    observed = []

    class CustomBuilder(UnitBuilder):
        name = "custom"

        def build(self, rules):
            observed.extend(rules)
            return [output]

    source = tmp_path / "rules.json"
    source.write_text(json.dumps([{
        "rule_id": "1", "rule_text": "完整原文",
        "metadata": {"domain": {"name": "海洋"}, "topic": "规则主题"},
    }]), encoding="utf-8")
    dataset = DatasetPipeline(embedding=Mock(), index_root=tmp_path, unit_builder=CustomBuilder()).prepare(
        source, dataset_name="preliminary",
    )
    units = list(dataset.units.values())
    assert observed[0].text == dataset.rules["1"].text == "完整原文"
    assert units[0].text == "子规则文本"
    assert units[0].index_text == "海洋 子规则文本"
    assert units[0].metadata == {
        "domain": {"name": "海洋"}, "topic": "单元主题", "offset": 8,
        "dataset": "preliminary", "unit_builder": {"strategy": "custom"},
    }
    assert output.metadata == {"offset": 8, "topic": "单元主题"}
    units[0].metadata["domain"]["name"] = "修改"
    assert observed[0].metadata["domain"]["name"] == dataset.rules["1"].metadata["domain"]["name"] == "海洋"

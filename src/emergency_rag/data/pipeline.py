"""数据集准备流水线，缓存目录命中时跳过全部离线计算。"""

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

from emergency_rag.unit_building.base import UnitBuilder
from emergency_rag.unit_building.rule import RuleUnitBuilder
from emergency_rag.clients.embedding import EmbeddingClient
from emergency_rag.retrieval.tokenizer.base import TextTokenizer
from emergency_rag.retrieval.tokenizer.jieba import JiebaTokenizer

from .models import IndexedDataset, Rule, dataset_index_directory
from .indexing import build_dataset_indexes


def read_rules(source: Path) -> list[Rule]:
    """读取原始规则并保留原文，数据格式和重复 ID 在读取边界检查。"""
    data = json.loads(source.read_text(encoding="utf-8-sig"))

    if not isinstance(data, list) or not data:
        raise ValueError("规则文件必须为非空 JSON 数组")

    rules = []
    ids: set[str] = set()

    for index, row in enumerate(data, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"第 {index} 条规则必须为对象")

        rule = Rule(
            rule_id=row.get("rule_id"),
            text=row.get("rule_text"),
            metadata=row.get("metadata", {}),
        )

        if rule.rule_id in ids:
            raise ValueError(f"重复 rule_id：{rule.rule_id}")

        ids.add(rule.rule_id)
        rules.append(rule)

    return rules


@dataclass(slots=True)
class DatasetPipeline:
    """读取规则、构建检索单元和双路索引，返回可直接检索的数据集。"""

    embedding: EmbeddingClient
    index_root: Path = field(default_factory=lambda: Path("data/indexes"))
    unit_builder: UnitBuilder = field(default_factory=RuleUnitBuilder)
    tokenizer: TextTokenizer = field(default_factory=JiebaTokenizer)

    def prepare(
        self,
        source: Path,
        *,
        dataset_name: str,
        overwrite: bool = False,
    ) -> IndexedDataset:
        directory = dataset_index_directory(
            self.index_root,
            dataset_name,
            self.unit_builder.name,
            self.tokenizer.name,
        )

        # 已有产物是本次实验的输入，命中时不读源文件、不构建单元、不请求文档向量。
        # 读取失败直接暴露给调用方，不静默重建或覆盖已有产物。
        if directory.exists() and not overwrite:
            return IndexedDataset(directory, tokenizer=self.tokenizer)

        rules = read_rules(source)
        rules_by_id = {rule.rule_id: rule for rule in rules}

        units = []
        for unit in self.unit_builder.build(rules):
            # 关联正确性在离线准备边界保证，在线直接用 rule_id 查完整原文
            if unit.rule_id not in rules_by_id:
                raise ValueError(f"SearchUnit 指向不存在的规则：{unit.rule_id}")

            # 合并规则元数据与单元元数据，并补充数据集与构建策略标识
            metadata = deepcopy({
                **rules_by_id[unit.rule_id].metadata,
                **unit.metadata,
                "dataset": dataset_name,
                "unit_builder": {"strategy": self.unit_builder.name},
            })
            units.append(unit.model_copy(update={"metadata": metadata}))

        directory = build_dataset_indexes(
            units,
            self.index_root,
            rules=rules,
            tokenizer=self.tokenizer,
            embedding=self.embedding,
            overwrite=overwrite,
        )

        return IndexedDataset(directory, tokenizer=self.tokenizer)
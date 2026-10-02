"""数据集准备流水线，缓存目录命中时跳过全部离线计算。"""

import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

from emergency_rag.data.units import UnitBuilder
from emergency_rag.data.units import RuleUnitBuilder
from emergency_rag.clients.embedding import EmbeddingClient
from emergency_rag.retrieval.tokenizer.base import TextTokenizer
from emergency_rag.retrieval.tokenizer.jieba import JiebaTokenizer

from .indexed_dataset import IndexedDataset, dataset_index_directory
from .units.base import Rule
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

    embedding: EmbeddingClient | Callable[[], EmbeddingClient]
    embedding_model: str | None = None  # 使用延迟工厂时声明模型名，缓存查找无需创建客户端。
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
        model = self.embedding_model if callable(self.embedding) else self.embedding.model
        if not model:
            raise ValueError("使用 Embedding 工厂时必须声明 embedding_model")
        directory = dataset_index_directory(
            self.index_root,
            dataset_name,
            self.unit_builder.name,
            self.tokenizer.name,
            model,
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

        # 客户端工厂仅在缓存未命中或显式重建时调用；读取索引不需要模型凭据。
        embedding = self.embedding() if callable(self.embedding) else self.embedding
        # 工厂声明与实际构建模型必须一致，避免产物写到另一个缓存目录。
        if embedding.model != model:
            raise ValueError("Embedding 工厂的实际模型与 embedding_model 声明不一致")
        directory = build_dataset_indexes(
            units,
            self.index_root,
            rules=rules,
            tokenizer=self.tokenizer,
            embedding=embedding,
            overwrite=overwrite,
        )

        return IndexedDataset(directory, tokenizer=self.tokenizer)

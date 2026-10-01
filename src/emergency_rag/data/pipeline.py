"""数据集准备流水线，缓存目录命中时跳过全部离线计算。"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from emergency_rag.chunking.base import Chunker
from emergency_rag.chunking.rule import RuleChunker
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
        rule = Rule(rule_id=row.get("rule_id"), text=row.get("rule_text"))
        if rule.rule_id in ids:
            raise ValueError(f"重复 rule_id：{rule.rule_id}")
        ids.add(rule.rule_id)
        rules.append(rule)
    return rules


@dataclass(slots=True)
class DatasetPipeline:
    """读取规则、分块、构建双路索引并返回可直接检索的数据集。"""

    embedding: EmbeddingClient
    index_root: Path = field(default_factory=lambda: Path("data/indexes"))
    chunker: Chunker = field(default_factory=RuleChunker)
    tokenizer: TextTokenizer = field(default_factory=JiebaTokenizer)

    def prepare(self, source: Path, *, dataset_name: str, overwrite: bool = False) -> IndexedDataset:
        directory = dataset_index_directory(self.index_root, dataset_name, self.chunker.name, self.tokenizer.name)
        # 已有产物是本次实验的输入，命中时不读规则、不分块、不请求文档向量。
        # 读取失败直接暴露给调用方，不静默重建或覆盖已有产物。
        if directory.exists() and not overwrite:
            return IndexedDataset(directory, tokenizer=self.tokenizer)
        units = self.chunker.chunk(read_rules(source))
        # 保留 Chunker 的位置信息等 metadata，并为离线产物标记来源。
        units = [
            unit.model_copy(update={"metadata": {
                **unit.metadata,
                "dataset": dataset_name,
                "chunker": {"strategy": self.chunker.name},
            }})
            for unit in units
        ]
        directory = build_dataset_indexes(
            units, self.index_root, tokenizer=self.tokenizer, embedding=self.embedding,
            overwrite=overwrite,
        )
        return IndexedDataset(directory, tokenizer=self.tokenizer)

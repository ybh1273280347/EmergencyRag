"""规则、检索单元和已索引数据集。"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import warnings

# 忽略包含该特定文本的警告
warnings.filterwarnings("ignore", message=".*resource module not available on Windows.*")

import bm25s
import faiss
import numpy as np
from bm25s.tokenization import Tokenizer
from pydantic import BaseModel, Field, field_validator

from emergency_rag.retrieval.tokenizer.base import TextTokenizer


class Rule(BaseModel):
    rule_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("rule_id", "text")
    @classmethod
    def require_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("规则 ID 和文本不能为空")
        return value


class SearchUnit(BaseModel):
    unit_id: str
    rule_id: str
    text: str          # 单元原文，用于重排和命中记录
    index_text: str    # 离线索引文本，可包含主题、领域等增强内容
    metadata: dict[str, Any] = Field(default_factory=dict)


def dataset_index_directory(
    index_root: Path,
    dataset_name: str,
    strategy: str,
    tokenizer_name: str,
    embedding_model: str,
) -> Path:
    """准备与构建共用的目录命名规则，在读取原始文件前即可确定位置。"""
    # 模型 ID 可以包含命名空间斜杠等字符，目录中转为单个安全名称。
    model_name = re.sub(r"[^A-Za-z0-9._-]+", "-", embedding_model).strip("-.")
    if not model_name:
        raise ValueError("Embedding 模型名不能生成有效索引目录")
    return (
        Path(index_root) / f"{dataset_name}-{strategy}-{tokenizer_name}-{model_name}"
    ).resolve()


@dataclass(slots=True)
class IndexedDataset:
    directory: Path
    # 从准备流水线携带同一分词器，检索组件不再接受独立分词配置
    tokenizer: TextTokenizer
    rules: dict[str, Rule] = field(init=False)
    units: dict[str, SearchUnit] = field(init=False)
    _bm25: tuple[bm25s.BM25, Tokenizer, list[str]] | None = field(
        default=None, init=False, repr=False
    )
    _dense: tuple[faiss.IndexFlatIP, list[str]] | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.directory = Path(self.directory).resolve()

        # 加载规则与检索单元快照
        records = json.loads(
            (self.directory / "rules.json").read_text(encoding="utf-8")
        )
        self.rules = {record["rule_id"]: Rule(**record) for record in records}

        records = json.loads(
            (self.directory / "units.json").read_text(encoding="utf-8")
        )
        self.units = {record["unit_id"]: SearchUnit(**record) for record in records}

        # Dataset 拥有索引资源，加载时一次性读取已有产物；查询只复用内存对象
        if (self.directory / "bm25").is_dir():
            self.load_bm25()
        if (self.directory / "dense").is_dir():
            self.load_dense()

    def load_bm25(self) -> tuple[bm25s.BM25, Tokenizer, list[str]]:
        """加载离线索引与词表，查询分词使用 Dataset 携带的分词器。"""
        if self._bm25 is not None:
            return self._bm25

        directory = self.directory / "bm25"
        index = bm25s.BM25.load(str(directory), load_corpus=False)

        vocabulary = Tokenizer(
            splitter=self.tokenizer.tokenize,
            stopwords=[],
            stemmer=None,
        )
        vocabulary.load_vocab(str(directory))

        unit_ids = json.loads(
            (directory / "unit_ids.json").read_text(encoding="utf-8")
        )
        self._bm25 = index, vocabulary, unit_ids
        return self._bm25

    def load_dense(self) -> tuple[faiss.IndexFlatIP, list[str]]:
        """加载 Dense 产物，不读取构建说明或调用文档 Embedding。"""
        if self._dense is not None:
            return self._dense

        directory = self.directory / "dense"

        # 序列化字节避开 FAISS 原生文件 API 的 Windows Unicode 路径限制
        index = faiss.deserialize_index(
            np.frombuffer(
                (directory / "faiss.index").read_bytes(),
                dtype="uint8",
            ),
        )
        unit_ids = json.loads(
            (directory / "faiss_mapping.json").read_text(encoding="utf-8")
        )
        self._dense = index, unit_ids
        return self._dense

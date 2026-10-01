"""可复用的离线索引构建能力，供数据集流水线和脚本入口共同使用。"""

import json
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import bm25s
import faiss
import numpy as np
from bm25s.tokenization import Tokenizer

from emergency_rag.clients.embedding import EmbeddingClient
from emergency_rag.retrieval.tokenizer.base import TextTokenizer

from .models import SearchUnit, dataset_index_directory


def normalize_vectors(vectors: np.ndarray, dimension: int | None = None) -> np.ndarray:
    """离线构建边界负责拒绝无效向量，输出 FAISS 可用的单位 float32 矩阵。"""
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or not array.shape[1]:
        raise ValueError("向量必须为二维矩阵且维度非零")
    if dimension is not None and array.shape[1] != dimension:
        raise ValueError(f"向量维度不匹配：期望 {dimension}，实际 {array.shape[1]}")
    if not np.isfinite(array).all():
        raise ValueError("向量包含非有限数值")
    norms = np.linalg.norm(array.astype(np.float64), axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("零向量不能用于余弦检索")
    return np.ascontiguousarray(array / norms, dtype=np.float32)


def _validate_units(units: list[SearchUnit]) -> None:
    """数据正确性在离线构建边界保证，不交给在线检索器反复检查。"""
    if not units:
        raise ValueError("没有可构建的 SearchUnit")
    if len({unit.unit_id for unit in units}) != len(units):
        raise ValueError("SearchUnit 包含重复 ID")
    if any(not unit.text.strip() for unit in units):
        raise ValueError("SearchUnit 文本不能为空")


def build_bm25_index(
    units: list[SearchUnit],
    tokenizer: TextTokenizer,
) -> tuple[bm25s.BM25, Tokenizer, list[str]]:
    """返回离线 BM25 索引、带词表分词器及行映射。"""
    _validate_units(units)
    bm25_tokenizer = Tokenizer(splitter=tokenizer.tokenize, stopwords=[], stemmer=None)
    tokens = bm25_tokenizer.tokenize([unit.text for unit in units], show_progress=False, allow_empty=False)
    if not any(tokens):
        raise ValueError("规则文本未产生有效索引 token")
    index = bm25s.BM25()
    index.index(tokens, show_progress=False)
    return index, bm25_tokenizer, [unit.unit_id for unit in units]


def build_dense_index(
    units: list[SearchUnit],
    embedding: EmbeddingClient,
) -> tuple[faiss.IndexFlatIP, list[str]]:
    """按传入数据顺序构建归一化内积索引，可独立于 BM25 使用。"""
    _validate_units(units)
    vectors = normalize_vectors(embedding.embed_batch([unit.text for unit in units]))
    if vectors.shape[0] != len(units):
        raise ValueError("Embedding 文档数与 SearchUnit 数量不一致")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index, [unit.unit_id for unit in units]


@contextmanager
def _replacement_directory(directory: Path, overwrite: bool) -> Generator[Path, None, None]:
    """完成全部写入才替换目录；写入或移动失败时保留旧产物。"""
    target = directory.resolve()
    if target.exists() and not overwrite:
        raise FileExistsError("索引目录已存在；重建请传入 overwrite=True")
    target.parent.mkdir(parents=True, exist_ok=True)
    # 临时目录与目标同盘，仅由 TemporaryDirectory 清理本次创建的文件。
    with TemporaryDirectory(prefix=".indexes-", dir=target.parent) as temporary:
        workspace = Path(temporary)
        built = workspace / "built"
        built.mkdir()
        yield built
        previous = workspace / "previous"
        if target.exists():
            target.rename(previous)
        try:
            built.rename(target)
        except OSError:
            if previous.exists():
                previous.rename(target)
            raise


def save_bm25_index(
    index: bm25s.BM25,
    tokenizer: Tokenizer,
    unit_ids: list[str],
    directory: Path,
    *,
    overwrite: bool = False,
) -> None:
    """保存已有 BM25 资源；不读取数据库、不创建模型客户端。"""
    with _replacement_directory(directory, overwrite) as built:
        index.save(str(built))
        tokenizer.save_vocab(str(built))
        (built / "unit_ids.json").write_text(json.dumps(unit_ids), encoding="utf-8")


def save_dense_index(
    index: faiss.IndexFlatIP,
    unit_ids: list[str],
    directory: Path,
    *,
    embedding_model: str,
    overwrite: bool = False,
) -> None:
    """保存已有 FAISS 资源；构建说明仅供人查看，检索器不依赖它。"""
    with _replacement_directory(directory, overwrite) as built:
        # 使用序列化字节，避免 FAISS 原生文件 API 的 Windows Unicode 路径问题。
        (built / "faiss.index").write_bytes(faiss.serialize_index(index).tobytes())
        (built / "faiss_mapping.json").write_text(json.dumps(unit_ids), encoding="utf-8")
        metadata = {
            "index_version": 1,
            "build_time": datetime.now(timezone.utc).isoformat(),
            "document_count": index.ntotal,
            "embedding_model": embedding_model,
            "embedding_dimension": index.d,
        }
        (built / "index_meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def build_dataset_indexes(
    units: list[SearchUnit],
    index_root: Path,
    *,
    tokenizer: TextTokenizer,
    embedding: EmbeddingClient,
    overwrite: bool = False,
) -> Path:
    """按数据集、分块和分词策略确定目录，固定构建 BM25 和 Dense。

    units.json 与索引由同一批知识单元生成，整体构建成功后才替换旧目录。
    在线 IndexedDataset 只读取这些产物，不依赖构建说明。
    """
    _validate_units(units)
    dataset_name = units[0].metadata.get("dataset")
    chunker = units[0].metadata.get("chunker")
    strategy = chunker.get("strategy") if isinstance(chunker, dict) else None
    directory = dataset_index_directory(index_root, dataset_name, strategy, tokenizer.name)
    if any(unit.metadata.get("dataset") != dataset_name or unit.metadata.get("chunker") != chunker for unit in units):
        raise ValueError("同次构建的 SearchUnits 必须来自同一数据集和分块策略")
    metadata: dict[str, object] = {
        "index_version": 1,
        "build_time": datetime.now(timezone.utc).isoformat(),
        "document_count": len(units),
        "dataset": dataset_name,
        "chunker": chunker,
        "tokenizer": {"strategy": tokenizer.name},
        "embedding_model": embedding.model,
    }
    with _replacement_directory(directory, overwrite) as built:
        sparse = build_bm25_index(units, tokenizer)
        save_bm25_index(*sparse, built / "bm25")
        dense = build_dense_index(units, embedding)
        save_dense_index(*dense, built / "dense", embedding_model=embedding.model)
        metadata["embedding_dimension"] = dense[0].d
        (built / "units.json").write_text(
            json.dumps([unit.model_dump(mode="json") for unit in units], ensure_ascii=False),
            encoding="utf-8",
        )
        (built / "index_meta.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return directory.resolve()

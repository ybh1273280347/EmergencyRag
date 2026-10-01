"""离线索引构建和保存工具；数据、分词器和模型客户端均由调用方提供。"""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import faiss

from emergency_rag.clients.embedding import EmbeddingClient
from emergency_rag.data.models import SearchUnit
from emergency_rag.retrieval.retrievers.bm25 import BM25RetrieverConfig
from emergency_rag.retrieval.retrievers.bm25_backend import Tokenizer, bm25s
from emergency_rag.retrieval.retrievers.vectors import normalize_vectors
from emergency_rag.retrieval.tokenizer.base import TextTokenizer


def build_bm25_index(
    units: list[SearchUnit],
    tokenizer: TextTokenizer,
    config: BM25RetrieverConfig | None = None,
) -> tuple[bm25s.BM25, Tokenizer, list[str]]:
    """返回可直接注入 BM25Retriever 的索引、带词表分词器及行映射。"""
    config = config or BM25RetrieverConfig()
    if not units:
        raise ValueError("没有可构建的 SearchUnit")
    bm25_tokenizer = Tokenizer(splitter=tokenizer.tokenize, stopwords=[], stemmer=None)
    tokens = bm25_tokenizer.tokenize([unit.text for unit in units], show_progress=False, allow_empty=False)
    if not any(tokens):
        raise ValueError("规则文本未产生有效索引 token")
    index = bm25s.BM25(method="lucene", k1=config.k1, b=config.b)
    index.index(tokens, show_progress=False)
    return index, bm25_tokenizer, [unit.unit_id for unit in units]


def build_dense_index(
    units: list[SearchUnit],
    embedding: EmbeddingClient,
) -> tuple[faiss.IndexFlatIP, list[str]]:
    """按传入数据顺序构建归一化内积索引，可独立于 BM25 使用。"""
    if not units:
        raise ValueError("没有可构建的 SearchUnit")
    vectors = normalize_vectors(embedding.embed_batch([unit.text for unit in units]))
    if vectors.shape[0] != len(units):
        raise ValueError("Embedding 文档数与 SearchUnit 数量不一致")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index, [unit.unit_id for unit in units]


@contextmanager
def _replacement_directory(directory: Path, overwrite: bool) -> Iterator[Path]:
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

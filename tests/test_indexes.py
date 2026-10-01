import json
from pathlib import Path

import faiss
import httpx
import numpy as np
import pytest
from openai import OpenAI
from pydantic import TypeAdapter
from zeroentropy import ZeroEntropy

from emergency_rag.clients.embedding import EmbeddingClient, EmbeddingError
from emergency_rag.data.repository import RuleRepository
from emergency_rag.retrieval.fusion.rrf import RRFFusion, RRFFusionConfig
from emergency_rag.retrieval.fusion.union import UnionFusion, UnionFusionConfig
from emergency_rag.retrieval.pipeline import RetrievalPipeline
from emergency_rag.retrieval.rerank.zero_entropy import ZeroEntropyReranker, ZeroEntropyRerankerConfig
from emergency_rag.retrieval.retrievers.bm25 import BM25Retriever, BM25RetrieverConfig
from emergency_rag.retrieval.retrievers.bm25_backend import Tokenizer, bm25s
from emergency_rag.retrieval.retrievers.dense import DenseRetriever, DenseRetrieverConfig
from emergency_rag.retrieval.retrievers.vectors import normalize_vectors
from emergency_rag.retrieval.tokenizer.jieba import JiebaTokenizer
from scripts.build_indexes import build_bm25_index, build_dense_index, save_bm25_index, save_dense_index
from scripts.prepare_rules import prepare_rules


@pytest.fixture
def built_indexes(database, embedding_config, tmp_path):
    source = tmp_path / "rules.json"
    source.write_text(json.dumps([
        {"rule_id": "1", "rule_text": "危险化学品事故应急结束条件"},
        {"rule_id": "2", "rule_text": "海冰灾害航空遥感观测频率"},
        {"rule_id": "3", "rule_text": "海上溢油社会应急队伍"},
    ], ensure_ascii=False), encoding="utf-8")
    prepare_rules(source, database)
    units = RuleRepository(database).load_search_units()

    def respond(request):
        vectors = {
            units[0].text: [3, 0, 0], units[1].text: [0, 4, 0],
            units[2].text: [0, 0, 2], "海冰": [0, 5, 0],
        }
        body = json.loads(request.content)
        return httpx.Response(200, json={"data": [
            {"index": i, "embedding": vectors[text]}
            for i, text in reversed(list(enumerate(body["input"])))
        ]})

    with OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as sdk:
        embedding = EmbeddingClient(embedding_config, sdk)
        sparse = build_bm25_index(units, JiebaTokenizer())
        dense = build_dense_index(units, embedding)
        directory = tmp_path / "indexes"
        save_bm25_index(*sparse, directory / "bm25")
        save_dense_index(*dense, directory / "dense", embedding_model=embedding.model)
        yield units, embedding, sparse, dense, directory


def test_built_objects_and_persisted_objects_have_same_results(built_indexes):
    units, embedding, sparse, dense, directory = built_indexes
    by_id = {unit.unit_id: unit for unit in units}
    # 调用方准备已有资源；Retriever 本身不读取文件、数据库或构建说明。
    tokenizer = Tokenizer(splitter=JiebaTokenizer().tokenize, stopwords=[], stemmer=None)
    tokenizer.load_vocab(str(directory / "bm25"))
    sparse_ids = TypeAdapter(list[str]).validate_json((directory / "bm25/unit_ids.json").read_text())
    sparse_index = bm25s.BM25.load(str(directory / "bm25"), load_corpus=False)
    dense_index = faiss.deserialize_index(np.frombuffer((directory / "dense/faiss.index").read_bytes(), dtype="uint8"))
    dense_ids = TypeAdapter(list[str]).validate_json((directory / "dense/faiss_mapping.json").read_text())
    before = [BM25Retriever(*sparse, by_id), DenseRetriever(*dense, by_id, embedding)]
    after = [BM25Retriever(sparse_index, tokenizer, sparse_ids, by_id), DenseRetriever(dense_index, dense_ids, by_id, embedding)]
    for built, restored in zip(before, after):
        assert restored.retrieve("海冰", 3) == built.retrieve("海冰", 3)
        assert restored.retrieve("海冰")[0].rule_id == "2"
        assert all(item.final_score is None for item in restored.retrieve("海冰"))
    vocab = dict(tokenizer.get_vocab_dict())
    assert after[0].retrieve("qxzvunknownword") == []
    assert after[0].retrieve("！？") == []
    assert tokenizer.get_vocab_dict() == vocab
    np.testing.assert_allclose(np.linalg.norm(dense_index.reconstruct_n(0, 3), axis=1), 1)
    assert after[1].retrieve("海冰")[0].metadata["retrieval"][0]["score"] == pytest.approx(1)


@pytest.mark.parametrize("vectors,dimension", [
    ([[0, 0]], None), ([[float("nan"), 1]], None),
    ([[float("inf"), 1]], None), ([[1, 2]], 3), ([1, 2], None), ([[]], None),
])
def test_invalid_vectors(vectors, dimension):
    with pytest.raises(ValueError):
        normalize_vectors(np.array(vectors), dimension)


def test_normalization_does_not_mutate_input():
    vectors = np.array([[3, 4], [4, 3]], dtype=np.float32)
    normalized = normalize_vectors(vectors)
    np.testing.assert_allclose(normalized, [[0.6, 0.8], [0.8, 0.6]])
    np.testing.assert_array_equal(vectors, [[3, 4], [4, 3]])
    assert normalized.dtype == np.float32 and normalized.flags.c_contiguous


def test_build_and_save_failure_preserve_previous_indexes(built_indexes, monkeypatch):
    units, embedding, sparse, dense, directory = built_indexes
    before = {str(path.relative_to(directory)): path.read_bytes() for path in directory.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError):
        save_bm25_index(*sparse, directory / "bm25")
    with OpenAI(api_key="test", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(500, json={"error": {"message": "failed"}}),
    ))) as sdk:
        with pytest.raises(EmbeddingError):
            build_dense_index(units, EmbeddingClient(embedding.config, sdk))

    original = Path.write_bytes

    def fail_write(path, content):
        if path.name == "faiss.index":
            raise OSError("模拟写入失败")
        return original(path, content)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_bytes", fail_write)
        with pytest.raises(OSError):
            save_dense_index(*dense, directory / "dense", embedding_model=embedding.model, overwrite=True)
    after = {str(path.relative_to(directory)): path.read_bytes() for path in directory.rglob("*") if path.is_file()}
    assert before == after
    assert not list(directory.glob(".indexes-*"))
    save_dense_index(*dense, directory / "dense", embedding_model=embedding.model, overwrite=True)


@pytest.mark.parametrize("kind", ["bm25", "dense"])
@pytest.mark.parametrize("ids,match", [
    (["rule:1"], "映射"), (["rule:1"] * 3, "重复"), (["rule:1", "rule:2", "missing"], "不存在"),
])
def test_invalid_injected_mapping(built_indexes, kind, ids, match):
    units, embedding, sparse, dense, directory = built_indexes
    by_id = {unit.unit_id: unit for unit in units}
    with pytest.raises(ValueError, match=match):
        if kind == "bm25":
            BM25Retriever(sparse[0], sparse[1], ids, by_id)
        else:
            DenseRetriever(dense[0], ids, by_id, embedding)


def test_each_retriever_owns_default_budget_and_override(built_indexes):
    units, embedding, sparse, dense, directory = built_indexes
    by_id = {unit.unit_id: unit for unit in units}
    retrievers = [
        BM25Retriever(*sparse, by_id, BM25RetrieverConfig(top_k=1)),
        DenseRetriever(*dense, by_id, embedding, DenseRetrieverConfig(top_k=1)),
    ]
    for retriever in retrievers:
        assert len(retriever.retrieve("海冰")) == 1
        assert len(retriever.retrieve("海冰", top_k=2)) == 2
        assert retriever.config.top_k == 1


@pytest.mark.parametrize("fusion", [
    RRFFusion(RRFFusionConfig(max_candidates=1)),
    UnionFusion(UnionFusionConfig(per_source_top_k={"bm25": 2, "dense": 2})),
])
def test_caller_assembles_real_retrievers_and_selected_fusion(built_indexes, fusion):
    units, embedding, sparse, dense, directory = built_indexes
    by_id = {unit.unit_id: unit for unit in units}
    calls = []

    def respond(request):
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(200, json={"results": [
            {"index": i, "relevance_score": 0.8 - 0.1 * i}
            for i in reversed(range(len(body["documents"])))
        ]})

    with ZeroEntropy(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as sdk:
        pipeline = RetrievalPipeline(
            retrievers=(BM25Retriever(*sparse, by_id), DenseRetriever(*dense, by_id, embedding)),
            fusion=fusion,
            reranker=ZeroEntropyReranker(client=sdk, config=ZeroEntropyRerankerConfig(model="test")),
        )
        result = pipeline.retrieve("海冰", top_k=1)
    count = result.metadata["candidate_counts"]["fusion"]
    assert count == (1 if fusion.name == "rrf" else 2)
    assert calls[0]["top_n"] == count
    assert result.metadata["retrieval_counts"] == {"bm25": 3, "dense": 3}
    assert len(result.candidates) == 1 and result.candidates[0].final_score == 0.8


def test_dense_uses_injected_row_order(built_indexes):
    units, embedding, sparse, dense, directory = built_indexes
    index, ids = dense
    reordered = faiss.IndexFlatIP(index.d)
    reordered.add(np.ascontiguousarray(index.reconstruct_n(0, 3)[::-1]))
    retriever = DenseRetriever(reordered, ids[::-1], {unit.unit_id: unit for unit in units}, embedding)
    assert retriever.retrieve("海冰")[0].rule_id == "2"


def test_sparse_build_is_independent_of_models(database):
    source = Path(__file__).resolve().parents[1] / "datasets/初赛规则集rules1.json"
    prepare_rules(source, database)
    units = RuleRepository(database).load_search_units()
    resources = build_bm25_index(units, JiebaTokenizer())
    sparse = BM25Retriever(*resources, {unit.unit_id: unit for unit in units})
    assert len(sparse.unit_ids) == 800
    assert "37" in [item.rule_id for item in sparse.retrieve("危险化学品事故应急结束条件", 10)]


def test_retrievers_use_ready_objects_without_file_access(built_indexes, monkeypatch):
    units, embedding, sparse, dense, directory = built_indexes
    by_id = {unit.unit_id: unit for unit in units}

    def fail_read(*args, **kwargs):
        raise AssertionError("检索器不应加载数据")

    monkeypatch.setattr(Path, "read_text", fail_read)
    monkeypatch.setattr(Path, "read_bytes", fail_read)
    monkeypatch.setattr(RuleRepository, "load_search_units", fail_read)
    retrievers = (BM25Retriever(*sparse, by_id), DenseRetriever(*dense, by_id, embedding))
    for retriever in retrievers:
        assert not hasattr(retriever, "load")
        assert retriever.retrieve("海冰")[0].rule_id == "2"

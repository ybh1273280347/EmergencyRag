import json
from pathlib import Path

import bm25s
import httpx
import numpy as np
import pytest
import yaml
from openai import OpenAI

from emergency_rag.clients.embedding import EmbeddingClient, EmbeddingError
from emergency_rag.clients.chat import ChatClient
from emergency_rag.data.indexed_dataset import IndexedDataset
from emergency_rag.data.units.base import Rule, SearchUnit, UnitBuilder
from emergency_rag.data.pipeline import DatasetPipeline
from emergency_rag.data.units.rule import RuleUnitBuilder
from emergency_rag.retrieval.fusion.rrf import RRFFusion
from emergency_rag.retrieval.fusion.union import UnionFusion
from emergency_rag.retrieval.gate.base import EvidenceGate
from emergency_rag.retrieval.pipeline import RetrievalPipeline
from emergency_rag.retrieval.rerank.qwen import QwenReranker
from emergency_rag.retrieval.retrievers.bm25 import BM25Retriever
from emergency_rag.retrieval.retrievers.dense import DenseRetriever
from emergency_rag.retrieval.tokenizer.jieba import JiebaTokenizer
from emergency_rag.retrieval.tokenizer.base import TextTokenizer
from emergency_rag.data.indexing import (
    _build_bm25_index, build_dataset_indexes, _build_dense_index, _normalize_vectors,
    _save_bm25_index, _save_dense_index,
)
from emergency_rag.data.pipeline import read_rules
from emergency_rag.load_pipeline import load_pipeline
from examples.chat_experiment import run_experiment


@pytest.fixture
def built_indexes(tmp_path):
    source = tmp_path / "rules.json"
    source.write_text(json.dumps([
        {"rule_id": "1", "rule_text": "危险化学品事故应急结束条件"},
        {"rule_id": "2", "rule_text": "海冰灾害航空遥感观测频率"},
        {"rule_id": "3", "rule_text": "海上溢油社会应急队伍"},
    ], ensure_ascii=False), encoding="utf-8")
    units = RuleUnitBuilder().build(read_rules(source))
    calls = []

    def respond(request):
        vectors = {
            units[0].text: [3, 0, 0], units[1].text: [0, 4, 0],
            units[2].text: [0, 0, 2], "海冰": [0, 5, 0],
            "海冰观测频率": [0, 5, 0], "海冰观测频率？A. 每日 B. 每小时": [0, 5, 0],
        }
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(200, json={"data": [
            {"index": i, "embedding": vectors[text]}
            for i, text in reversed(list(enumerate(body["input"])))
        ]})

    with OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as sdk:
        embedding = EmbeddingClient(sdk, model="test-embedding")
        dataset = DatasetPipeline(embedding=embedding, index_root=tmp_path).prepare(source, dataset_name="preliminary")
        directory = dataset.directory
        units = list(dataset.units.values())
        calls.clear()
        yield units, embedding, directory, calls


def test_offline_indexes_preserve_scores_and_row_mapping(built_indexes):
    units, embedding, directory, calls = built_indexes
    dataset = IndexedDataset(directory, tokenizer=JiebaTokenizer())
    sparse_index, tokenizer, sparse_ids = dataset.load_bm25()
    before_index, before_tokenizer, before_ids = _build_bm25_index(units, JiebaTokenizer())
    query_tokens = tokenizer.tokenize(["海冰"], update_vocab=False, allow_empty=False, show_progress=False)
    before_rows, before_scores = before_index.retrieve(query_tokens, k=3, show_progress=False)
    after_rows, after_scores = sparse_index.retrieve(query_tokens, k=3, show_progress=False)
    np.testing.assert_array_equal(before_rows, after_rows)
    np.testing.assert_allclose(before_scores, after_scores)
    assert sparse_ids == before_ids == [unit.unit_id for unit in units]
    assert tokenizer.get_vocab_dict() == before_tokenizer.get_vocab_dict()

    dense_index, dense_ids = dataset.load_dense()
    before_index, before_ids = _build_dense_index(units, embedding)
    vector = np.array([[0, 1, 0]], dtype=np.float32)
    before_scores, before_rows = before_index.search(vector, 3)
    after_scores, after_rows = dense_index.search(vector, 3)
    np.testing.assert_array_equal(before_rows, after_rows)
    np.testing.assert_allclose(before_scores, after_scores)
    assert dense_ids == before_ids == sparse_ids
    assert dataset.units == {unit.unit_id: unit for unit in units}
    assert len(dataset.rules) == 3
    assert dataset.rules["2"].text == units[1].text
    np.testing.assert_allclose(np.linalg.norm(dense_index.reconstruct_n(0, 3), axis=1), 1)


def test_directory_loaded_retrievers_and_query_only_embedding(built_indexes):
    units, embedding, directory, calls = built_indexes
    dataset = IndexedDataset(directory, tokenizer=JiebaTokenizer())
    sparse = BM25Retriever()
    dense = DenseRetriever(embedding)
    assert calls == []  # 初始化不访问文档或查询 Embedding API。
    for retriever in (sparse, dense):
        results = retriever.retrieve("海冰", dataset)
        assert results[0].rule_id == "2" and len(results) == 3
        assert all(item.final_score is None for item in results)
    assert [call["input"] for call in calls] == [["海冰"]]
    vocabulary = dataset.load_bm25()[1]
    vocab = dict(vocabulary.get_vocab_dict())
    assert sparse.retrieve("qxzvunknownword", dataset) == []
    assert sparse.retrieve("！？", dataset) == []
    assert vocabulary.get_vocab_dict() == vocab


def test_changing_only_directory_selects_another_dataset(built_indexes, tmp_path):
    units, embedding, directory, calls = built_indexes
    other_units = [units[1].model_copy(update={
        "unit_id": "rule:semifinal", "rule_id": "semifinal",
        "metadata": {"dataset": "semifinal", "unit_builder": {"strategy": "rule"}},
    })]
    other_directory = build_dataset_indexes(
        other_units, tmp_path, rules=[Rule(rule_id="semifinal", text=units[1].text)],
        tokenizer=JiebaTokenizer(), embedding=embedding,
    )
    first = IndexedDataset(directory, tokenizer=JiebaTokenizer())
    second = IndexedDataset(other_directory, tokenizer=JiebaTokenizer())
    retrievers = BM25Retriever(), DenseRetriever(embedding)
    # 同一组组件交替使用不同 Dataset，索引资源随 Dataset 切换。
    for dataset, expected in ((first, "2"), (second, "semifinal"), (first, "2")):
        for retriever in retrievers:
            result = retriever.retrieve("海冰", dataset)[0]
            assert result.rule_id == expected
    assert first.units.keys() != second.units.keys()


def test_query_never_rebuilds_or_rereads_indexes(built_indexes, monkeypatch):
    units, embedding, directory, calls = built_indexes
    dataset = IndexedDataset(directory, tokenizer=JiebaTokenizer())
    sparse = BM25Retriever()
    dense = DenseRetriever(embedding)

    def fail(*args, **kwargs):
        raise AssertionError("查询不应读文件或构建索引")

    monkeypatch.setattr(Path, "read_text", fail)
    monkeypatch.setattr(Path, "read_bytes", fail)
    monkeypatch.setattr(bm25s.BM25, "index", fail)
    for _ in range(2):
        assert sparse.retrieve("海冰", dataset)[0].rule_id == "2"
        assert dense.retrieve("海冰", dataset)[0].rule_id == "2"
    assert [call["input"] for call in calls] == [["海冰"]]  # 第二次查询直接复用向量。


@pytest.mark.parametrize("vectors,dimension", [
    ([[0, 0]], None), ([[float("nan"), 1]], None),
    ([[float("inf"), 1]], None), ([[1, 2]], 3), ([1, 2], None), ([[]], None),
])
def test_invalid_vectors_rejected_offline(vectors, dimension):
    with pytest.raises(ValueError):
        _normalize_vectors(np.array(vectors), dimension)


def test_normalization_does_not_mutate_input():
    vectors = np.array([[3, 4], [4, 3]], dtype=np.float32)
    normalized = _normalize_vectors(vectors)
    np.testing.assert_allclose(normalized, [[0.6, 0.8], [0.8, 0.6]])
    np.testing.assert_array_equal(vectors, [[3, 4], [4, 3]])
    assert normalized.dtype == np.float32 and normalized.flags.c_contiguous


def test_dataset_build_failure_preserves_previous_directory(built_indexes, monkeypatch):
    units, embedding, directory, calls = built_indexes
    rules = list(IndexedDataset(directory, tokenizer=JiebaTokenizer()).rules.values())
    before = {str(path.relative_to(directory)): path.read_bytes() for path in directory.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError):
        build_dataset_indexes(units, directory.parent, rules=rules, tokenizer=JiebaTokenizer(), embedding=embedding)
    with OpenAI(api_key="test", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(500, json={"error": {"message": "failed"}}),
    ))) as sdk:
        with pytest.raises(EmbeddingError):
            build_dataset_indexes(
                units, directory.parent, rules=rules, tokenizer=JiebaTokenizer(),
                embedding=EmbeddingClient(sdk, model=embedding.model, batch_size=embedding.batch_size), overwrite=True,
            )

    original = Path.write_bytes

    def fail_write(path, content):
        if path.name == "faiss.index":
            raise OSError("模拟写入失败")
        return original(path, content)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_bytes", fail_write)
        with pytest.raises(OSError):
            build_dataset_indexes(units, directory.parent, rules=rules, tokenizer=JiebaTokenizer(), embedding=embedding, overwrite=True)
    after = {str(path.relative_to(directory)): path.read_bytes() for path in directory.rglob("*") if path.is_file()}
    assert before == after
    assert not list(directory.parent.glob(".indexes-*"))
    build_dataset_indexes(units, directory.parent, rules=rules, tokenizer=JiebaTokenizer(), embedding=embedding, overwrite=True)
    assert len(IndexedDataset(directory, tokenizer=JiebaTokenizer()).units) == 3


@pytest.mark.parametrize("invalid", ["empty", "duplicate", "blank", "blank_index"])
def test_data_correctness_is_checked_by_offline_builders(built_indexes, invalid):
    units, embedding, directory, calls = built_indexes
    invalid_units = {
        "empty": [], "duplicate": [units[0], units[0]],
        "blank": [units[0].model_copy(update={"text": " "})],
        "blank_index": [units[0].model_copy(update={"index_text": " "})],
    }[invalid]
    with pytest.raises(ValueError):
        _build_bm25_index(invalid_units, JiebaTokenizer())
    with pytest.raises(ValueError):
        _build_dense_index(invalid_units, embedding)


def test_each_retriever_budget_is_fixed_at_initialization(built_indexes):
    units, embedding, directory, calls = built_indexes
    dataset = IndexedDataset(directory, tokenizer=JiebaTokenizer())
    retrievers = (
        BM25Retriever(top_k=1),
        DenseRetriever(embedding, top_k=1),
    )
    for retriever in retrievers:
        assert len(retriever.retrieve("海冰", dataset)) == 1
        with pytest.raises(TypeError):
            retriever.retrieve("海冰", dataset, top_k=2)
        assert retriever.top_k == 1


@pytest.mark.parametrize("fusion", [
    RRFFusion(top_k=1),
    UnionFusion(),
])
def test_pipeline_top_k_does_not_change_recall_or_rerank(built_indexes, fusion):
    units, embedding, directory, embedding_calls = built_indexes
    dataset = IndexedDataset(directory, tokenizer=JiebaTokenizer())
    rerank_calls = []

    def respond(request):
        body = json.loads(request.content)
        rerank_calls.append(body)
        return httpx.Response(200, json={"results": [
            {"index": i, "relevance_score": 0.8 - 0.1 * i}
            for i in reversed(range(len(body["documents"])))
        ]})

    with OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as sdk:
        pipeline = RetrievalPipeline(
            dataset=dataset,
            retrievers=(BM25Retriever(), DenseRetriever(embedding)),
            fusion=fusion,
            reranker=QwenReranker(client=sdk, model="test", instruction="评估相关性"),
        )
        result = pipeline.retrieve("海冰", top_k=1)
        expanded_result = pipeline.retrieve("海冰", top_k=2)
    count = result.metadata["candidate_counts"]["fusion"]
    assert count == (1 if fusion.name == "rrf" else 3)
    assert [call["top_n"] for call in rerank_calls] == [count, count]
    assert [call["input"] for call in embedding_calls] == [["海冰"]]
    assert expanded_result.metadata["retrieval_counts"] == result.metadata["retrieval_counts"]
    assert expanded_result.metadata["candidate_counts"]["fusion"] == count
    assert len(expanded_result.evidence) == min(2, count)
    assert result.metadata["retrieval_counts"] == {"bm25": 3, "dense": 3}
    assert len(result.evidence) == 1 and result.evidence[0].final_score == 0.8


def test_bm25_only_dataset_requires_no_model_or_dense_index(tmp_path):
    source = Path(__file__).resolve().parents[1] / "data/raw/初赛规则集rules1.json"
    rules = read_rules(source)
    units = RuleUnitBuilder().build(rules)
    directory = tmp_path / "bm25_only"
    # 单索引工具仍可独立验证；完整数据集构建入口始终生成两路索引。
    _save_bm25_index(*_build_bm25_index(units, JiebaTokenizer()), directory / "bm25")
    (directory / "rules.json").write_text(json.dumps([rule.model_dump() for rule in rules]), encoding="utf-8")
    (directory / "units.json").write_text(json.dumps([unit.model_dump() for unit in units]), encoding="utf-8")
    dataset = IndexedDataset(directory, tokenizer=JiebaTokenizer())
    sparse = BM25Retriever(top_k=10)
    assert len(dataset.units) == 800
    assert not (directory / "dense").exists()
    assert "37" in [item.rule_id for item in sparse.retrieve("危险化学品事故应急结束条件", dataset)]


def test_dense_only_dataset_requires_no_bm25_index(built_indexes, tmp_path):
    units, embedding, directory, calls = built_indexes
    rules = list(IndexedDataset(directory, tokenizer=JiebaTokenizer()).rules.values())
    directory = tmp_path / "dense_only"
    _save_dense_index(*_build_dense_index(units, embedding), directory / "dense", embedding_model=embedding.model)
    (directory / "rules.json").write_text(json.dumps([rule.model_dump() for rule in rules]), encoding="utf-8")
    (directory / "units.json").write_text(json.dumps([unit.model_dump() for unit in units]), encoding="utf-8")
    dataset = IndexedDataset(directory, tokenizer=JiebaTokenizer())
    dense = DenseRetriever(embedding)
    calls.clear()
    assert not (directory / "bm25").exists()
    assert dense.retrieve("海冰", dataset)[0].rule_id == "2"
    assert [call["input"] for call in calls] == [["海冰"]]


def test_build_notes_are_not_required_for_loading(built_indexes):
    units, embedding, directory, calls = built_indexes
    (directory / "index_meta.json").unlink()
    (directory / "dense/index_meta.json").write_text("invalid json", encoding="utf-8")
    dataset = IndexedDataset(directory, tokenizer=JiebaTokenizer())
    assert BM25Retriever().retrieve("海冰", dataset)[0].rule_id == "2"
    assert DenseRetriever(embedding).retrieve("海冰", dataset)[0].rule_id == "2"


def test_caller_selected_units_are_indexed_and_restore_full_rule(tmp_path):
    class SentenceBuilder(UnitBuilder):
        name = "sentence"

        def build(self, rules):
            return [
                SearchUnit(unit_id=f"sub_rule:{rule.rule_id}:{position}", rule_id=rule.rule_id, text=text, index_text=text)
                for rule in rules
                for position, text in enumerate(rule.text.split("。"), start=1)
            ]

    source = tmp_path / "rules.json"
    source.write_text('[{"rule_id":"1","rule_text":"海冰灾害观测频率。化学品事故结束条件"}]', encoding="utf-8")

    def respond(request):
        texts = json.loads(request.content)["input"]
        return httpx.Response(200, json={"data": [
            {"index": i, "embedding": [1, i + 1]} for i in range(len(texts))
        ]})

    with OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as sdk:
        dataset = DatasetPipeline(
            index_root=tmp_path, unit_builder=SentenceBuilder(), embedding=EmbeddingClient(sdk, model="test-embedding"),
        ).prepare(source, dataset_name="preliminary")
        directory = dataset.directory
        units = list(dataset.units.values())
    assert directory == tmp_path / "preliminary-sentence-jieba-test-embedding"
    assert (directory / "bm25/unit_ids.json").is_file()
    assert (directory / "dense/faiss.index").is_file()
    dataset = IndexedDataset(directory, tokenizer=JiebaTokenizer())
    assert list(dataset.units.values()) == units
    candidates = BM25Retriever().retrieve("化学品事故结束条件", dataset)
    assert candidates[0].unit_id == "sub_rule:1:2"
    assert candidates[0].rule_id == "1"
    assert candidates[0].text == units[1].text
    result = RetrievalPipeline(dataset=dataset, retrievers=[BM25Retriever()], fusion=UnionFusion()).retrieve(
        "化学品事故结束条件", top_k=2,
    )
    assert len(result.evidence) == 1
    assert result.evidence[0].text == "海冰灾害观测频率。化学品事故结束条件"
    assert len(result.evidence[0].matched_units) == 2
    assert result.evidence[0].final_score is None


def test_index_directory_is_determined_by_unit_provenance(built_indexes):
    units, embedding, directory, calls = built_indexes
    assert directory.name == "preliminary-rule-jieba-test-embedding"
    assert (directory / "bm25/unit_ids.json").is_file()
    assert (directory / "dense/faiss.index").is_file()
    notes = json.loads((directory / "index_meta.json").read_text(encoding="utf-8"))
    assert notes["dataset"] == "preliminary"
    assert notes["unit_builder"] == {"strategy": "rule"}
    assert notes["tokenizer"] == {"strategy": "jieba"}
    assert notes["embedding_model"] == embedding.model
    assert notes["embedding_dimension"] == 3
    dense_notes = json.loads((directory / "dense/index_meta.json").read_text(encoding="utf-8"))
    assert dense_notes["embedding_model"] == notes["embedding_model"]
    assert dense_notes["embedding_dimension"] == notes["embedding_dimension"]
    assert notes["document_count"] == 3
    assert notes["rule_count"] == 3
    assert all(unit.metadata["unit_builder"] == {"strategy": "rule"} for unit in IndexedDataset(directory, tokenizer=JiebaTokenizer()).units.values())


def test_dataset_pipeline_loads_existing_directory_before_reading_rules(built_indexes, monkeypatch):
    units, embedding, directory, calls = built_indexes

    def fail(*args, **kwargs):
        raise AssertionError("缓存命中不得读取源规则、构建单元或索引")

    monkeypatch.setattr("emergency_rag.data.pipeline.read_rules", fail)
    monkeypatch.setattr("emergency_rag.data.pipeline.build_dataset_indexes", fail)
    monkeypatch.setattr(RuleUnitBuilder, "build", fail)
    monkeypatch.setattr(embedding, "embed_batch", fail)
    dataset = DatasetPipeline(embedding=embedding, index_root=directory.parent).prepare(
        directory.parent / "missing-rules.json", dataset_name="preliminary",
    )
    assert dataset.directory == directory
    assert list(dataset.units.values()) == units
    assert BM25Retriever().retrieve("海冰", dataset)[0].rule_id == "2"
    assert calls == []


def test_embedding_model_switch_builds_separate_directory(built_indexes):
    units, embedding, directory, calls = built_indexes
    alternate_embedding = EmbeddingClient(embedding.client, model="other-embedding")
    preparation = DatasetPipeline(embedding=alternate_embedding, index_root=directory.parent)
    alternate = preparation.prepare(directory.parent / "rules.json", dataset_name="preliminary")
    assert alternate.directory.name == "preliminary-rule-jieba-other-embedding"
    assert directory.is_dir() and alternate.directory != directory
    notes = json.loads((alternate.directory / "index_meta.json").read_text(encoding="utf-8"))
    assert notes["embedding_model"] == "other-embedding"
    assert len(calls) == 1 and calls[0]["model"] == "other-embedding"
    calls.clear()
    cached = preparation.prepare(directory.parent / "missing.json", dataset_name="preliminary")
    assert cached.directory == alternate.directory and calls == []


def test_embedding_factory_cache_lookup_does_not_create_client(built_indexes):
    units, embedding, directory, calls = built_indexes

    def fail():
        raise AssertionError("读取模型对应缓存不能创建客户端")

    cached = DatasetPipeline(
        embedding=fail, embedding_model=embedding.model, index_root=directory.parent,
    ).prepare(directory.parent / "missing.json", dataset_name="preliminary")
    assert cached.directory == directory and calls == []


def test_embedding_factory_model_declaration_matches_build(built_indexes):
    units, embedding, directory, calls = built_indexes
    preparation = DatasetPipeline(
        embedding=lambda: embedding, embedding_model="different-model", index_root=directory.parent,
    )
    with pytest.raises(ValueError, match="实际模型"):
        preparation.prepare(directory.parent / "rules.json", dataset_name="preliminary")
    assert not (directory.parent / "preliminary-rule-jieba-different-model").exists()
    assert calls == []


def test_dataset_pipeline_builds_missing_directory_and_returns_searchable_dataset(built_indexes, tmp_path):
    units, embedding, directory, calls = built_indexes
    source = tmp_path / "source.json"
    source.write_text(json.dumps([
        {"rule_id": unit.rule_id, "rule_text": unit.text} for unit in units
    ]), encoding="utf-8")
    pipeline = DatasetPipeline(embedding=embedding, index_root=tmp_path / "new-indexes")
    dataset = pipeline.prepare(source, dataset_name="semifinal")
    assert dataset.directory == tmp_path / "new-indexes/semifinal-rule-jieba-test-embedding"
    assert (dataset.directory / "bm25/unit_ids.json").is_file()
    assert (dataset.directory / "dense/faiss.index").is_file()
    assert [call["input"] for call in calls] == [[unit.text for unit in units]]
    assert all(unit.metadata["dataset"] == "semifinal" for unit in dataset.units.values())
    calls.clear()
    cached = pipeline.prepare(tmp_path / "missing.json", dataset_name="semifinal")
    assert cached.units == dataset.units
    assert calls == []
    assert BM25Retriever().retrieve("海冰", dataset)[0].rule_id == "2"
    assert DenseRetriever(embedding).retrieve("海冰", dataset)[0].rule_id == "2"


def test_dataset_pipeline_rebuild_requires_explicit_overwrite(built_indexes, tmp_path):
    units, embedding, directory, calls = built_indexes
    source = tmp_path / "replacement.json"
    source.write_text(json.dumps([{"rule_id": "replacement", "rule_text": units[1].text}]), encoding="utf-8")
    pipeline = DatasetPipeline(embedding=embedding, index_root=directory.parent)
    assert len(pipeline.prepare(source, dataset_name="preliminary").units) == 3
    assert calls == []
    replaced = pipeline.prepare(source, dataset_name="preliminary", overwrite=True)
    assert list(replaced.units) == ["rule:replacement"]
    assert BM25Retriever().retrieve("海冰", replaced)[0].rule_id == "replacement"
    assert [call["input"] for call in calls] == [[units[1].text]]


def test_dataset_pipeline_does_not_rebuild_unreadable_existing_artifact(built_indexes, monkeypatch):
    units, embedding, directory, calls = built_indexes
    (directory / "units.json").write_text("invalid json", encoding="utf-8")

    def fail(*args, **kwargs):
        raise AssertionError("已有产物读取失败不应触发重建")

    monkeypatch.setattr("emergency_rag.data.pipeline.build_dataset_indexes", fail)
    with pytest.raises(json.JSONDecodeError):
        DatasetPipeline(embedding=embedding, index_root=directory.parent).prepare(
            directory.parent / "missing.json", dataset_name="preliminary",
        )
    assert calls == []


def test_tokenizer_is_carried_from_build_through_cache_to_bm25(built_indexes, tmp_path):
    units, embedding, directory, calls = built_indexes

    class WholeTextTokenizer(TextTokenizer):
        name = "whole-text"

        def __init__(self):
            self.observed = []

        def tokenize(self, text):
            self.observed.append(text)
            return [text]

    source = tmp_path / "rules.json"
    source.write_text(json.dumps([
        {"rule_id": unit.rule_id, "rule_text": unit.text} for unit in units
    ]), encoding="utf-8")
    tokenizer = WholeTextTokenizer()
    preparation = DatasetPipeline(embedding=embedding, index_root=tmp_path, tokenizer=tokenizer)
    dataset = preparation.prepare(source, dataset_name="preliminary")
    assert dataset.directory.name == "preliminary-rule-whole-text-test-embedding"
    assert dataset.tokenizer is tokenizer
    assert tokenizer.observed == [unit.text for unit in units]
    retriever = BM25Retriever()
    tokenizer.observed.clear()
    assert retriever.retrieve(units[1].text, dataset)[0].rule_id == "2"
    assert tokenizer.observed == [units[1].text]
    # 检索器没有第二个 tokenizer 入口，不能覆盖 Dataset 的分词策略。
    with pytest.raises(TypeError):
        BM25Retriever(tokenizer=JiebaTokenizer())

    tokenizer.observed.clear()
    calls.clear()
    cached = preparation.prepare(tmp_path / "missing.json", dataset_name="preliminary")
    assert cached.tokenizer is tokenizer
    assert cached.directory == dataset.directory
    assert tokenizer.observed == [] and calls == []
    assert retriever.retrieve(units[1].text, cached)[0].rule_id == "2"
    assert tokenizer.observed == [units[1].text]


def test_different_tokenizers_select_different_index_directories(built_indexes, tmp_path):
    units, embedding, directory, calls = built_indexes

    class CharacterTokenizer(TextTokenizer):
        name = "characters"

        def tokenize(self, text):
            return list(text)

    source = tmp_path / "rules.json"
    source.write_text(json.dumps([
        {"rule_id": unit.rule_id, "rule_text": unit.text} for unit in units
    ]), encoding="utf-8")
    tokenizer = CharacterTokenizer()
    alternate = DatasetPipeline(embedding=embedding, index_root=directory.parent, tokenizer=tokenizer).prepare(
        source, dataset_name="preliminary",
    )
    assert alternate.directory.name == "preliminary-rule-characters-test-embedding"
    assert alternate.directory != directory
    assert (alternate.directory / "bm25/unit_ids.json").is_file()
    assert (alternate.directory / "dense/faiss.index").is_file()
    assert alternate.tokenizer is tokenizer
    assert [call["input"] for call in calls] == [[unit.text for unit in units]]
    assert BM25Retriever().retrieve("海冰", alternate)[0].rule_id == "2"


@pytest.mark.parametrize("metadata", [
    {"dataset": "semifinal", "unit_builder": {"strategy": "rule"}},
    {"dataset": "preliminary", "unit_builder": {"strategy": "sentence"}},
])
def test_mixed_dataset_or_unit_builder_is_rejected_offline(built_indexes, metadata, tmp_path):
    units, embedding, directory, calls = built_indexes
    invalid = [units[0], units[1].model_copy(update={"metadata": metadata})]
    rules = list(IndexedDataset(directory, tokenizer=JiebaTokenizer()).rules.values())
    target = tmp_path / "mixed"
    with pytest.raises(ValueError, match="同一数据集和单元构建策略"):
        build_dataset_indexes(invalid, target, rules=rules, tokenizer=JiebaTokenizer(), embedding=embedding)
    assert not target.exists()
    assert calls == []


@pytest.mark.parametrize("question,instructions,answer", [
    ("海冰观测频率", "根据规则用文字解释，并引用规则编号。", "根据规则 2，观测频率如下。"),
    ("海冰观测频率？A. 每日 B. 每小时", "根据规则只返回选项字母。", "B"),
])
@pytest.mark.parametrize("use_cache", [True, False])
def test_experiment_prepares_retrieves_and_answers(built_indexes, monkeypatch, question, instructions, answer, use_cache):
    units, embedding, directory, calls = built_indexes
    source = directory.parent / "experiment-rules.json"
    if use_cache:
        index_root = directory.parent
    else:
        index_root = directory.parent / "experiment-indexes"
        source.write_text(json.dumps([
            {"rule_id": unit.rule_id, "rule_text": unit.text} for unit in units
        ]), encoding="utf-8")
    config_path = directory.parent / "experiment.yaml"
    config_path.write_text(yaml.safe_dump({
        "dataset": {
            "source": str(source), "dataset_name": "preliminary",
            "index_root": str(index_root),
        },
        "retrieval": {
            "retrievers": [
                "bm25", "dense",
            ],
            "fusion": "rrf",
        },
    }), encoding="utf-8")
    monkeypatch.setattr("emergency_rag.clients.embedding.get_embedding_client", lambda: embedding)
    monkeypatch.setattr("emergency_rag.clients.embedding.settings.embedding_model", embedding.model)
    pipeline = load_pipeline(config_path)
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "chat-1", "object": "chat.completion", "created": 1, "model": "answer-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
        })

    with OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as sdk:
        result, actual = run_experiment(
            pipeline=pipeline, chat=ChatClient(sdk, model="answer-model"), question=question,
            instructions=instructions, top_k=1,
        )
    assert actual == answer
    assert len(result.evidence) == 1
    assert result.evidence[0].rule_id == "2"
    assert result.evidence[0].final_rank == 1
    assert result.evidence[0].final_score is None
    expected_inputs = [[question]] if use_cache else [[unit.text for unit in units], [question]]
    assert [call["input"] for call in calls] == expected_inputs
    assert len(requests) == 1
    messages = requests[0]["messages"]
    assert messages[0] == {"role": "system", "content": instructions}
    assert question in messages[1]["content"]
    assert f"[规则 2]\n{units[1].text}" in messages[1]["content"]
    assert units[0].text not in messages[1]["content"]
    assert units[2].text not in messages[1]["content"]
    assert "response_format" not in requests[0]


def test_enhanced_text_indexes_units_but_reranks_original_and_returns_rules(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps([
        {"rule_id": "37", "rule_text": "条件甲。条件乙", "metadata": {"domain": "化学品", "topic": "应急结束"}},
        {"rule_id": "463", "rule_text": "海冰观测频率"},
    ], ensure_ascii=False), encoding="utf-8")

    class EnhancedBuilder(UnitBuilder):
        name = "rule-sub-rule-context"

        def build(self, rules):
            units = []
            for rule in rules:
                context = "危险化学品 应急终止 " if rule.rule_id == "37" else "海冰 "
                units.append(SearchUnit(
                    unit_id=f"rule:{rule.rule_id}", rule_id=rule.rule_id,
                    text=rule.text, index_text=context + rule.text,
                ))
                if rule.rule_id == "37":
                    units.extend(SearchUnit(
                        unit_id=f"sub_rule:37:{i}", rule_id="37", text=text, index_text=context + text,
                        metadata={"position": i},
                    ) for i, text in enumerate(rule.text.split("。"), start=1))
            return units

    embedding_calls, rerank_calls = [], []

    def embed(request):
        body = json.loads(request.content)
        embedding_calls.append(body)
        return httpx.Response(200, json={"data": [
            {"index": i, "embedding": [0, 1] if "海冰" in text else [1, 0]}
            for i, text in reversed(list(enumerate(body["input"])))
        ]})

    def rerank(request):
        body = json.loads(request.content)
        rerank_calls.append(body)
        scores = {"条件甲": 0.95, "条件甲。条件乙": 0.90, "条件乙": 0.88, "海冰观测频率": 0.85}
        return httpx.Response(200, json={"results": [
            {"index": i, "relevance_score": scores[text]}
            for i, text in reversed(list(enumerate(body["documents"])))
        ]})

    class Gate(EvidenceGate):
        def filter(self, evidence):
            assert [item.rule_id for item in evidence] == ["37", "463"]
            assert [item.final_score for item in evidence] == [0.95, 0.85]
            assert len(evidence[0].matched_units) == 3
            return evidence

    with OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(embed))) as embedding_sdk:
        embedding = EmbeddingClient(embedding_sdk, model="test-embedding")
        preparation = DatasetPipeline(
            embedding=embedding, index_root=tmp_path / "indexes", unit_builder=EnhancedBuilder(),
        )
        dataset = preparation.prepare(source, dataset_name="preliminary")
        assert len(dataset.rules) == 2 and len(dataset.units) == 4
        assert embedding_calls[0]["input"] == [unit.index_text for unit in dataset.units.values()]
        assert all("应急终止" not in unit.text for unit in dataset.units.values())
        sparse = BM25Retriever()
        assert sparse.retrieve("应急终止", dataset)[0].rule_id == "37"
        with OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(rerank))) as rerank_sdk:
            pipeline = RetrievalPipeline(
                dataset=dataset, retrievers=[sparse, DenseRetriever(embedding)],
                fusion=RRFFusion(),
                reranker=QwenReranker(rerank_sdk, model="test-reranker", instruction="", batch_size=2),
                gate=Gate(),
            )
            result = pipeline.retrieve("应急终止", top_k=2)
        assert result.evidence[0].text == "条件甲。条件乙"
        assert result.evidence[0].metadata == {"domain": "化学品", "topic": "应急结束"}
        assert result.evidence[0].sources == ["bm25", "dense"]
        assert [item.final_rank for item in result.evidence] == [1, 2]
        assert sorted(text for call in rerank_calls for text in call["documents"]) == sorted(
            unit.text for unit in dataset.units.values()
        )
        assert [call["top_n"] for call in rerank_calls] == [2, 2]
        assert "index_text" not in json.dumps(result.model_dump())
        assert "危险化学品 应急终止 " not in json.dumps(result.evidence[0].model_dump(), ensure_ascii=False)
        embedding_calls.clear()
        cached = preparation.prepare(tmp_path / "missing.json", dataset_name="preliminary")
        assert cached.rules == dataset.rules and cached.units == dataset.units
        assert cached.tokenizer is preparation.tokenizer
        assert embedding_calls == []


@pytest.mark.parametrize("invalid", ["unknown_rule", "duplicate_rule"])
def test_offline_build_rejects_invalid_rule_links(built_indexes, tmp_path, invalid):
    units, embedding, directory, calls = built_indexes
    rules = list(IndexedDataset(directory, tokenizer=JiebaTokenizer()).rules.values())
    if invalid == "unknown_rule":
        units = [units[0].model_copy(update={"rule_id": "missing"})]
    else:
        rules.append(rules[0])
    with pytest.raises(ValueError, match="不存在|重复"):
        build_dataset_indexes(units, tmp_path / "invalid-rules", rules=rules, tokenizer=JiebaTokenizer(), embedding=embedding)
    assert calls == []
    assert not (tmp_path / "invalid-rules").exists()


@pytest.mark.parametrize("missing", ["rules", "index_text"])
def test_old_dataset_format_requires_explicit_rebuild(built_indexes, missing):
    units, embedding, directory, calls = built_indexes
    if missing == "rules":
        (directory / "rules.json").unlink()
    else:
        records = json.loads((directory / "units.json").read_text(encoding="utf-8"))
        for record in records:
            del record["index_text"]
        (directory / "units.json").write_text(json.dumps(records), encoding="utf-8")
    preparation = DatasetPipeline(embedding=embedding, index_root=directory.parent)
    with pytest.raises((FileNotFoundError, ValueError)):
        preparation.prepare(directory.parent / "rules.json", dataset_name="preliminary")
    assert calls == []
    rebuilt = preparation.prepare(directory.parent / "rules.json", dataset_name="preliminary", overwrite=True)
    assert len(rebuilt.rules) == len(rebuilt.units) == 3
    assert [call["input"] for call in calls] == [[unit.index_text for unit in units]]

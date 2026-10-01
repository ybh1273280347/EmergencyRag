import json
from pathlib import Path

import bm25s
import httpx
import numpy as np
import pytest
from openai import OpenAI
from zeroentropy import ZeroEntropy

from emergency_rag.clients.embedding import EmbeddingClient, EmbeddingError
from emergency_rag.clients.chat import ChatClient
from emergency_rag.chunking.base import Chunker
from emergency_rag.data.models import IndexedDataset, SearchUnit
from emergency_rag.data.pipeline import DatasetPipeline
from emergency_rag.chunking.rule import RuleChunker
from emergency_rag.retrieval.fusion.rrf import RRFFusion
from emergency_rag.retrieval.fusion.union import UnionFusion
from emergency_rag.retrieval.pipeline import RetrievalPipeline
from emergency_rag.retrieval.rerank.zero_entropy import ZeroEntropyReranker
from emergency_rag.retrieval.retrievers.bm25 import BM25Retriever
from emergency_rag.retrieval.retrievers.dense import DenseRetriever
from emergency_rag.retrieval.tokenizer.jieba import JiebaTokenizer
from emergency_rag.retrieval.tokenizer.base import TextTokenizer
from emergency_rag.data.indexing import (
    build_bm25_index, build_dataset_indexes, build_dense_index, normalize_vectors,
    save_bm25_index, save_dense_index,
)
from emergency_rag.data.pipeline import read_rules
from examples.preliminary_experiment import run_experiment


@pytest.fixture
def built_indexes(tmp_path):
    source = tmp_path / "rules.json"
    source.write_text(json.dumps([
        {"rule_id": "1", "rule_text": "危险化学品事故应急结束条件"},
        {"rule_id": "2", "rule_text": "海冰灾害航空遥感观测频率"},
        {"rule_id": "3", "rule_text": "海上溢油社会应急队伍"},
    ], ensure_ascii=False), encoding="utf-8")
    units = RuleChunker().chunk(read_rules(source))
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
    before_index, before_tokenizer, before_ids = build_bm25_index(units, JiebaTokenizer())
    query_tokens = tokenizer.tokenize(["海冰"], update_vocab=False, allow_empty=False, show_progress=False)
    before_rows, before_scores = before_index.retrieve(query_tokens, k=3, show_progress=False)
    after_rows, after_scores = sparse_index.retrieve(query_tokens, k=3, show_progress=False)
    np.testing.assert_array_equal(before_rows, after_rows)
    np.testing.assert_allclose(before_scores, after_scores)
    assert sparse_ids == before_ids == [unit.unit_id for unit in units]
    assert tokenizer.get_vocab_dict() == before_tokenizer.get_vocab_dict()

    dense_index, dense_ids = dataset.load_dense()
    before_index, before_ids = build_dense_index(units, embedding)
    vector = np.array([[0, 1, 0]], dtype=np.float32)
    before_scores, before_rows = before_index.search(vector, 3)
    after_scores, after_rows = dense_index.search(vector, 3)
    np.testing.assert_array_equal(before_rows, after_rows)
    np.testing.assert_allclose(before_scores, after_scores)
    assert dense_ids == before_ids == sparse_ids
    assert dataset.units == {unit.unit_id: unit for unit in units}
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
        "metadata": {"dataset": "semifinal", "chunker": {"strategy": "rule"}},
    })]
    other_directory = build_dataset_indexes(other_units, tmp_path, tokenizer=JiebaTokenizer(), embedding=embedding)
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
    assert [call["input"] for call in calls] == [["海冰"], ["海冰"]]


@pytest.mark.parametrize("vectors,dimension", [
    ([[0, 0]], None), ([[float("nan"), 1]], None),
    ([[float("inf"), 1]], None), ([[1, 2]], 3), ([1, 2], None), ([[]], None),
])
def test_invalid_vectors_rejected_offline(vectors, dimension):
    with pytest.raises(ValueError):
        normalize_vectors(np.array(vectors), dimension)


def test_normalization_does_not_mutate_input():
    vectors = np.array([[3, 4], [4, 3]], dtype=np.float32)
    normalized = normalize_vectors(vectors)
    np.testing.assert_allclose(normalized, [[0.6, 0.8], [0.8, 0.6]])
    np.testing.assert_array_equal(vectors, [[3, 4], [4, 3]])
    assert normalized.dtype == np.float32 and normalized.flags.c_contiguous


def test_dataset_build_failure_preserves_previous_directory(built_indexes, monkeypatch):
    units, embedding, directory, calls = built_indexes
    before = {str(path.relative_to(directory)): path.read_bytes() for path in directory.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError):
        build_dataset_indexes(units, directory.parent, tokenizer=JiebaTokenizer(), embedding=embedding)
    with OpenAI(api_key="test", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(500, json={"error": {"message": "failed"}}),
    ))) as sdk:
        with pytest.raises(EmbeddingError):
            build_dataset_indexes(
                units, directory.parent, tokenizer=JiebaTokenizer(),
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
            build_dataset_indexes(units, directory.parent, tokenizer=JiebaTokenizer(), embedding=embedding, overwrite=True)
    after = {str(path.relative_to(directory)): path.read_bytes() for path in directory.rglob("*") if path.is_file()}
    assert before == after
    assert not list(directory.parent.glob(".indexes-*"))
    build_dataset_indexes(units, directory.parent, tokenizer=JiebaTokenizer(), embedding=embedding, overwrite=True)
    assert len(IndexedDataset(directory, tokenizer=JiebaTokenizer()).units) == 3


@pytest.mark.parametrize("invalid", ["empty", "duplicate", "blank"])
def test_data_correctness_is_checked_by_offline_builders(built_indexes, invalid):
    units, embedding, directory, calls = built_indexes
    invalid_units = {
        "empty": [], "duplicate": [units[0], units[0]],
        "blank": [units[0].model_copy(update={"text": " "})],
    }[invalid]
    with pytest.raises(ValueError):
        build_bm25_index(invalid_units, JiebaTokenizer())
    with pytest.raises(ValueError):
        build_dense_index(invalid_units, embedding)


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

    with ZeroEntropy(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as sdk:
        pipeline = RetrievalPipeline(
            dataset=dataset,
            retrievers=(BM25Retriever(), DenseRetriever(embedding)),
            fusion=fusion,
            reranker=ZeroEntropyReranker(client=sdk, model="test", instruction="评估相关性"),
        )
        result = pipeline.retrieve("海冰", top_k=1)
        expanded_result = pipeline.retrieve("海冰", top_k=2)
    count = result.metadata["candidate_counts"]["fusion"]
    assert count == (1 if fusion.name == "rrf" else 3)
    assert [call["top_n"] for call in rerank_calls] == [count, count]
    assert [call["input"] for call in embedding_calls] == [["海冰"], ["海冰"]]
    assert expanded_result.metadata["retrieval_counts"] == result.metadata["retrieval_counts"]
    assert expanded_result.metadata["candidate_counts"]["fusion"] == count
    assert len(expanded_result.candidates) == min(2, count)
    assert result.metadata["retrieval_counts"] == {"bm25": 3, "dense": 3}
    assert len(result.candidates) == 1 and result.candidates[0].final_score == 0.8


def test_bm25_only_dataset_requires_no_model_or_dense_index(tmp_path):
    source = Path(__file__).resolve().parents[1] / "datasets/初赛规则集rules1.json"
    units = RuleChunker().chunk(read_rules(source))
    directory = tmp_path / "bm25_only"
    # 单索引工具仍可独立验证；完整数据集构建入口始终生成两路索引。
    save_bm25_index(*build_bm25_index(units, JiebaTokenizer()), directory / "bm25")
    (directory / "units.json").write_text(json.dumps([unit.model_dump() for unit in units]), encoding="utf-8")
    dataset = IndexedDataset(directory, tokenizer=JiebaTokenizer())
    sparse = BM25Retriever(top_k=10)
    assert len(dataset.units) == 800
    assert not (directory / "dense").exists()
    assert "37" in [item.rule_id for item in sparse.retrieve("危险化学品事故应急结束条件", dataset)]


def test_dense_only_dataset_requires_no_bm25_index(built_indexes, tmp_path):
    units, embedding, directory, calls = built_indexes
    directory = tmp_path / "dense_only"
    save_dense_index(*build_dense_index(units, embedding), directory / "dense", embedding_model=embedding.model)
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


def test_caller_selected_chunks_are_the_indexed_and_returned_units(tmp_path):
    class SentenceChunker(Chunker):
        name = "sentence"

        def chunk(self, records):
            return [
                SearchUnit(unit_id=f"{rule.rule_id}:{position}", rule_id=rule.rule_id, text=text)
                for rule in records
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
            index_root=tmp_path, chunker=SentenceChunker(), embedding=EmbeddingClient(sdk, model="test-embedding"),
        ).prepare(source, dataset_name="preliminary")
        directory = dataset.directory
        units = list(dataset.units.values())
    assert directory == tmp_path / "preliminary-sentence-jieba"
    assert (directory / "bm25/unit_ids.json").is_file()
    assert (directory / "dense/faiss.index").is_file()
    dataset = IndexedDataset(directory, tokenizer=JiebaTokenizer())
    assert list(dataset.units.values()) == units
    candidates = BM25Retriever().retrieve("化学品事故结束条件", dataset)
    assert candidates[0].unit_id == "1:2"
    assert candidates[0].rule_id == "1"
    assert candidates[0].text == units[1].text


def test_index_directory_is_determined_by_unit_provenance(built_indexes):
    units, embedding, directory, calls = built_indexes
    assert directory.name == "preliminary-rule-jieba"
    assert (directory / "bm25/unit_ids.json").is_file()
    assert (directory / "dense/faiss.index").is_file()
    notes = json.loads((directory / "index_meta.json").read_text(encoding="utf-8"))
    assert notes["dataset"] == "preliminary"
    assert notes["chunker"] == {"strategy": "rule"}
    assert notes["tokenizer"] == {"strategy": "jieba"}
    assert notes["document_count"] == 3
    assert all(unit.metadata["chunker"] == {"strategy": "rule"} for unit in IndexedDataset(directory, tokenizer=JiebaTokenizer()).units.values())


def test_dataset_pipeline_loads_existing_directory_before_reading_rules(built_indexes, monkeypatch):
    units, embedding, directory, calls = built_indexes

    def fail(*args, **kwargs):
        raise AssertionError("缓存命中不得读取规则、分块或构建索引")

    monkeypatch.setattr("emergency_rag.data.pipeline.read_rules", fail)
    monkeypatch.setattr("emergency_rag.data.pipeline.build_dataset_indexes", fail)
    monkeypatch.setattr(RuleChunker, "chunk", fail)
    monkeypatch.setattr(embedding, "embed_batch", fail)
    dataset = DatasetPipeline(embedding=embedding, index_root=directory.parent).prepare(
        directory.parent / "missing-rules.json", dataset_name="preliminary",
    )
    assert dataset.directory == directory
    assert list(dataset.units.values()) == units
    assert BM25Retriever().retrieve("海冰", dataset)[0].rule_id == "2"
    assert calls == []


def test_dataset_pipeline_builds_missing_directory_and_returns_searchable_dataset(built_indexes, tmp_path):
    units, embedding, directory, calls = built_indexes
    source = tmp_path / "source.json"
    source.write_text(json.dumps([
        {"rule_id": unit.rule_id, "rule_text": unit.text} for unit in units
    ]), encoding="utf-8")
    pipeline = DatasetPipeline(embedding=embedding, index_root=tmp_path / "new-indexes")
    dataset = pipeline.prepare(source, dataset_name="semifinal")
    assert dataset.directory == tmp_path / "new-indexes/semifinal-rule-jieba"
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
    assert dataset.directory.name == "preliminary-rule-whole-text"
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
    assert alternate.directory.name == "preliminary-rule-characters"
    assert alternate.directory != directory
    assert (alternate.directory / "bm25/unit_ids.json").is_file()
    assert (alternate.directory / "dense/faiss.index").is_file()
    assert alternate.tokenizer is tokenizer
    assert [call["input"] for call in calls] == [[unit.text for unit in units]]
    assert BM25Retriever().retrieve("海冰", alternate)[0].rule_id == "2"


@pytest.mark.parametrize("metadata", [
    {}, {"dataset": "preliminary"},
    {"dataset": "../escape", "chunker": {"strategy": "rule"}},
    {"dataset": "初赛", "chunker": {"strategy": "rule"}},
    {"dataset": "preliminary", "chunker": {"strategy": "../rule"}},
])
def test_invalid_directory_provenance_is_rejected_offline(built_indexes, metadata, tmp_path):
    units, embedding, directory, calls = built_indexes
    invalid = [units[0].model_copy(update={"metadata": metadata})]
    target = tmp_path / "invalid"
    with pytest.raises(ValueError, match="英文名称"):
        build_dataset_indexes(invalid, target, tokenizer=JiebaTokenizer(), embedding=embedding)
    assert not target.exists()
    assert calls == []


@pytest.mark.parametrize("metadata", [
    {"dataset": "semifinal", "chunker": {"strategy": "rule"}},
    {"dataset": "preliminary", "chunker": {"strategy": "sentence"}},
])
def test_mixed_dataset_or_chunker_is_rejected_offline(built_indexes, metadata, tmp_path):
    units, embedding, directory, calls = built_indexes
    invalid = [units[0], units[1].model_copy(update={"metadata": metadata})]
    target = tmp_path / "mixed"
    with pytest.raises(ValueError, match="同一数据集和分块策略"):
        build_dataset_indexes(invalid, target, tokenizer=JiebaTokenizer(), embedding=embedding)
    assert not target.exists()
    assert calls == []


@pytest.mark.parametrize("question,instructions,answer", [
    ("海冰观测频率", "根据规则用文字解释，并引用规则编号。", "根据规则 2，观测频率如下。"),
    ("海冰观测频率？A. 每日 B. 每小时", "根据规则只返回选项字母。", "B"),
])
@pytest.mark.parametrize("use_cache", [True, False])
def test_experiment_prepares_retrieves_and_answers(built_indexes, question, instructions, answer, use_cache):
    units, embedding, directory, calls = built_indexes
    source = directory.parent / "experiment-rules.json"
    if use_cache:
        index_root = directory.parent
    else:
        index_root = directory.parent / "experiment-indexes"
        source.write_text(json.dumps([
            {"rule_id": unit.rule_id, "rule_text": unit.text} for unit in units
        ]), encoding="utf-8")
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "chat-1", "object": "chat.completion", "created": 1, "model": "answer-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
        })

    with OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as sdk:
        result, actual = run_experiment(
            embedding=embedding, chat=ChatClient(sdk), chat_model="answer-model", question=question,
            instructions=instructions, top_k=1, index_root=index_root, source=source,
        )
    assert actual == answer
    assert len(result.candidates) == 1
    assert result.candidates[0].rule_id == "2"
    assert result.candidates[0].final_rank == 1
    assert result.candidates[0].final_score is None
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

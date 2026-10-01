import json
from functools import cache
from pathlib import Path

import httpx
import pytest
import yaml
from openai import OpenAI
from zeroentropy import ZeroEntropy

from emergency_rag.clients import chat, embedding
from emergency_rag.data.models import SearchUnit
from emergency_rag.data.pipeline import DatasetPipeline
from emergency_rag.load_pipeline import PipelineConfigError, load_pipeline
from emergency_rag.registry import COMPONENT_REGISTRY
from emergency_rag.retrieval.fusion.base import Fusion
from emergency_rag.retrieval.fusion.union import UnionFusion
from emergency_rag.retrieval.gate.base import EvidenceGate
from emergency_rag.retrieval.query.base import QueryProcessor
from emergency_rag.retrieval.rerank import zero_entropy
from emergency_rag.retrieval.rerank.base import Reranker
from emergency_rag.unit_building.base import UnitBuilder


class EnhancedUnitBuilder(UnitBuilder):
    name = "context"

    def __init__(self, *, prefix):
        self.prefix = prefix

    def build(self, rules):
        return [SearchUnit(
            unit_id=f"sub_rule:{rule.rule_id}:1", rule_id=rule.rule_id,
            text=rule.text[:2], index_text=self.prefix + rule.text,
        ) for rule in rules]


class FirstFusion(Fusion):
    name = "first"

    def fuse(self, result_sets):
        return result_sets[0]


@pytest.fixture(autouse=True)
def isolate_model_cache():
    # 模型缓存属于进程，测试之间显式释放连接，避免配置和实例串用。
    getters = (embedding.get_embedding_client, chat.get_chat_client, zero_entropy.get_rerank_model)
    for getter in getters:
        getter.cache_clear()
    yield
    for getter in getters:
        if getter.cache_info().currsize:
            model = getter()
            client = model._client if isinstance(model, chat.ChatClient) else model.client
            client.close()
        getter.cache_clear()


@pytest.fixture
def experiment_yaml(tmp_path, monkeypatch):
    directory = tmp_path / "experiment"
    directory.mkdir()
    (directory / "rules.json").write_text(json.dumps([
        {"rule_id": "1", "rule_text": "条件甲及完整说明", "metadata": {"topic": "应急结束"}},
        {"rule_id": "2", "rule_text": "海冰观测频率"},
    ], ensure_ascii=False), encoding="utf-8")
    config = {
        "dataset": {"source": "rules.json", "dataset_name": "preliminary", "index_root": "indexes"},
        "retrieval": {
            "retrievers": [{"name": "bm25", "params": {"top_k": 2}}, {"name": "dense", "params": {"top_k": 2}}],
            "fusion": {"name": "rrf", "params": {"top_k": 2}},
        },
    }
    calls = []

    def respond(request):
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(200, json={"data": [
            {"index": i, "embedding": [0, 1] if "海冰" in text else [1, 0]}
            for i, text in reversed(list(enumerate(body["input"])))
        ]})

    with OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as sdk:
        model = embedding.EmbeddingClient(sdk, model="test-embedding")
        monkeypatch.setattr(embedding, "get_embedding_client", lambda: model)
        yield directory / "pipeline.yaml", config, model, calls


def test_yaml_builds_indexes_and_shares_embedding_through_cache(experiment_yaml, monkeypatch, tmp_path):
    path, config, model, calls = experiment_yaml
    prepared_embeddings = []
    original_prepare = DatasetPipeline.prepare

    def prepare(self, *args, **kwargs):
        prepared_embeddings.append(self.embedding)
        return original_prepare(self, *args, **kwargs)

    monkeypatch.setattr(DatasetPipeline, "prepare", prepare)
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    pipeline = load_pipeline(path)
    assert pipeline.dataset.directory == path.parent / "indexes/preliminary-rule-jieba"
    assert all((pipeline.dataset.directory / name).is_file() for name in (
        "rules.json", "units.json", "bm25/unit_ids.json", "dense/faiss.index",
    ))
    assert prepared_embeddings[0] is pipeline.retrievers[1].embedding is model
    assert type(pipeline.query_processor) is QueryProcessor
    assert type(pipeline.reranker) is Reranker
    assert type(pipeline.gate) is EvidenceGate
    assert calls[0]["input"] == ["条件甲及完整说明", "海冰观测频率"]
    result = pipeline.retrieve("条件甲", top_k=1)
    assert result.evidence[0].rule_id == "1"
    assert result.evidence[0].text == "条件甲及完整说明"
    assert result.evidence[0].final_score is None

    def fail(*args, **kwargs):
        raise AssertionError("缓存命中不能读源规则或重新构建索引")

    (path.parent / "rules.json").unlink()
    calls.clear()
    monkeypatch.setattr("emergency_rag.data.pipeline.read_rules", fail)
    monkeypatch.setattr("emergency_rag.data.pipeline.build_dataset_indexes", fail)
    cached = load_pipeline(path)
    assert cached.dataset.rules == pipeline.dataset.rules
    assert cached.dataset.units == pipeline.dataset.units
    assert calls == []
    assert prepared_embeddings[1] is cached.retrievers[1].embedding is model
    assert cached.retrieve("条件甲").evidence[0].rule_id == "1"


def test_yaml_custom_registered_components(experiment_yaml, monkeypatch):
    path, config, model, calls = experiment_yaml
    monkeypatch.setitem(COMPONENT_REGISTRY["unit_builder"], EnhancedUnitBuilder.name, EnhancedUnitBuilder)
    monkeypatch.setitem(COMPONENT_REGISTRY["fusion"], FirstFusion.name, FirstFusion)
    config["dataset"]["unit_builder"] = {"name": "context", "params": {"prefix": "应急终止 "}}
    config["retrieval"]["fusion"] = "first"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    pipeline = load_pipeline(path)
    assert pipeline.dataset.directory.name == "preliminary-context-jieba"
    assert calls[0]["input"] == ["应急终止 条件甲及完整说明", "应急终止 海冰观测频率"]
    result = pipeline.retrieve("应急终止 条件甲", top_k=1)
    assert result.evidence[0].matched_units[0].unit_id == "sub_rule:1:1"
    assert result.evidence[0].matched_units[0].text == "条件"
    assert result.evidence[0].text == "条件甲及完整说明"
    assert result.metadata["fusion_strategy"] == "first"


def test_yaml_switches_fusion_and_keeps_model_instance(experiment_yaml):
    path, config, model, calls = experiment_yaml
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    first = load_pipeline(path)
    config["retrieval"]["fusion"] = "union"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    second = load_pipeline(path)
    assert isinstance(second.fusion, UnionFusion)
    assert first.retrievers[1].embedding is second.retrievers[1].embedding is model
    assert second.retrieve("条件甲", top_k=2).metadata["candidate_counts"] == {"fusion": 2, "rules": 2, "final": 2}


def test_shipped_baseline_and_independent_rerank_instructions(experiment_yaml, monkeypatch):
    path, config, model, calls = experiment_yaml
    baseline = yaml.safe_load((Path(__file__).resolve().parents[1] / "config/baseline.yaml").read_text(encoding="utf-8"))
    baseline["dataset"] = config["dataset"]
    rerank_calls = []

    def respond(request):
        body = json.loads(request.content)
        rerank_calls.append(body)
        return httpx.Response(200, json={"results": [
            {"index": i, "relevance_score": 0.9 - 0.1 * i}
            for i in reversed(range(len(body["documents"])))
        ]})

    with ZeroEntropy(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as sdk:
        @cache
        def get_model(**params):
            return zero_entropy.ZeroEntropyReranker(sdk, model="test-reranker", **params)

        monkeypatch.setattr(zero_entropy, "get_rerank_model", get_model)
        path.write_text(yaml.safe_dump(baseline), encoding="utf-8")
        pipeline = load_pipeline(path)
        result = pipeline.retrieve("条件甲", top_k=1)
        assert result.evidence[0].final_score == 0.9
        assert rerank_calls[0]["top_n"] == 2
        assert set(rerank_calls[0]["documents"]) == {"条件甲及完整说明", "海冰观测频率"}
        assert load_pipeline(path).reranker is pipeline.reranker
        baseline["retrieval"]["reranker"]["params"]["instruction"] = "另一实验指令"
        path.write_text(yaml.safe_dump(baseline), encoding="utf-8")
        second = load_pipeline(path)
        assert second.reranker is not pipeline.reranker
        assert load_pipeline(path).reranker is second.reranker
        assert pipeline.reranker.instruction == "请评估规则片段与问题的相关性。"
        assert second.reranker.instruction == "另一实验指令"


@pytest.mark.parametrize("invalid", [
    "root", "extra_root", "source", "overwrite", "unknown_name", "class_path", "params",
    "kwargs", "null_gate", "wrong_builder", "unknown_stage", "dataset_override", "retrievers",
])
def test_invalid_yaml_fails_before_dataset_build(experiment_yaml, monkeypatch, invalid):
    path, config, model, calls = experiment_yaml
    if invalid == "root":
        config = []
    elif invalid == "extra_root":
        config["unknown"] = {}
    elif invalid == "source":
        del config["dataset"]["source"]
    elif invalid == "overwrite":
        config["dataset"]["overwrite"] = "false"
    elif invalid == "unknown_name":
        config["retrieval"]["fusion"] = "unknown"
    elif invalid == "class_path":
        config["retrieval"]["fusion"] = {"target": "os.getenv"}
    elif invalid == "params":
        config["retrieval"]["fusion"]["params"] = []
    elif invalid == "kwargs":
        config["retrieval"]["fusion"]["params"]["secret_typo"] = "test-private-key"
    elif invalid == "null_gate":
        config["retrieval"]["gate"] = None
    elif invalid == "wrong_builder":
        monkeypatch.setitem(COMPONENT_REGISTRY["unit_builder"], "wrong", Reranker)
        config["dataset"]["unit_builder"] = "wrong"
    elif invalid == "unknown_stage":
        config["retrieval"]["unknown"] = 1
    elif invalid == "dataset_override":
        config["retrieval"]["dataset"] = "other"
    elif invalid == "retrievers":
        config["retrieval"]["retrievers"] = "bm25"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    def fail(*args, **kwargs):
        raise AssertionError("配置失败不得开始构建数据集")

    monkeypatch.setattr(DatasetPipeline, "prepare", fail)
    with pytest.raises(PipelineConfigError) as error:
        load_pipeline(path)
    assert "test-private-key" not in str(error.value)
    assert calls == []


@pytest.mark.parametrize("content", ["dataset: [", "!!python/object:os.system {}"])
def test_invalid_or_unsafe_yaml_tags_rejected(tmp_path, content):
    path = tmp_path / "invalid.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(PipelineConfigError, match="YAML"):
        load_pipeline(path)


@pytest.mark.parametrize("kind", ["embedding", "rerank", "chat"])
def test_module_models_created_once_from_environment(monkeypatch, kind):
    prefix = {"embedding": "RAG_EMBEDDING", "rerank": "ZEROENTROPY", "chat": "RAG_CHAT"}[kind]
    module = {"embedding": embedding, "rerank": zero_entropy, "chat": chat}[kind]
    getter = {"embedding": embedding.get_embedding_client, "rerank": zero_entropy.get_rerank_model, "chat": chat.get_chat_client}[kind]
    monkeypatch.setenv(f"{prefix}_API_KEY", "test-private-key")
    monkeypatch.setenv(f"{prefix}_MODEL", "test-model")
    url_variable = "ZEROENTROPY_URL" if kind == "rerank" else f"{prefix}_BASE_URL"
    monkeypatch.setenv(url_variable, "https://model.example/v1")
    monkeypatch.setenv(f"{prefix}_TIMEOUT", "12")
    monkeypatch.setenv(f"{prefix}_MAX_RETRIES", "1")
    constructors = []
    sdk_class = ZeroEntropy if kind == "rerank" else OpenAI

    def construct(**kwargs):
        constructors.append(kwargs)
        return sdk_class(http_client=httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(500),
        )), **kwargs)

    monkeypatch.setattr(module, "ZeroEntropy" if kind == "rerank" else "OpenAI", construct)
    first = getter()
    assert getter() is first
    assert len(constructors) == 1
    assert constructors[0] == {
        "api_key": "test-private-key", "base_url": "https://model.example/v1", "timeout": 12.0, "max_retries": 1,
    }
    assert first.model == "test-model"
    if kind == "rerank":
        # 指令和批大小均参与缓存，注册表无需再包一层重排器实例。
        instructed = getter(instruction="评估相关性", batch_size=2)
        other_batch = getter(instruction="评估相关性", batch_size=3)
        assert getter(instruction="评估相关性", batch_size=2) is instructed
        assert instructed is not first and instructed is not other_batch
        assert instructed.instruction == other_batch.instruction == "评估相关性"
        assert instructed.batch_size == 2 and other_batch.batch_size == 3
        assert first.instruction == ""
        assert len(constructors) == 3
        instructed.client.close()
        other_batch.client.close()


@pytest.mark.parametrize("getter,variable", [
    (embedding.get_embedding_client, "RAG_EMBEDDING_API_KEY"),
    (zero_entropy.get_rerank_model, "ZEROENTROPY_API_KEY"),
    (chat.get_chat_client, "RAG_CHAT_API_KEY"),
])
def test_missing_environment_does_not_create_placeholder_model(getter, variable):
    with pytest.raises(ValueError, match=variable):
        getter()
    assert getter.cache_info().currsize == 0

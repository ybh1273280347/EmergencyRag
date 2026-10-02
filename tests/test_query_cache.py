"""查询缓存跨对象、跨进程复用；默认测试不访问真实模型。"""

import json
import subprocess
import sys
from pathlib import Path

import httpx
import numpy as np
import pytest
import yaml
from openai import OpenAI

from emergency_rag.cache import JsonFileCache
from emergency_rag.clients.embedding import EmbeddingClient, EmbeddingError
from emergency_rag.load_pipeline import load_pipeline, PipelineConfigError
from emergency_rag.registry import COMPONENT_REGISTRY
from emergency_rag.retrieval.models import QueryContext
from emergency_rag.retrieval.query.base import QueryProcessor
from emergency_rag.retrieval.query.cached import CachedQueryProcessor


class RewriteProcessor(QueryProcessor):
    name = "rewrite-test"

    def __init__(self):
        self.calls = 0

    def process(self, query):
        self.calls += 1
        return QueryContext(original_query=query, rewritten_query="海冰观测", sub_queries=["上报时限"])


def test_query_context_persistence_refresh_and_independent_lists(tmp_path):
    processor = RewriteProcessor()
    cached = CachedQueryProcessor(processor, directory=tmp_path)
    first = cached.process("原问题")
    first.sub_queries.append("被调用方修改")
    assert cached.process("原问题").sub_queries == ["上报时限"]
    assert processor.calls == 1
    reloaded = CachedQueryProcessor(processor, directory=tmp_path)
    assert reloaded.process("原问题").queries == ["海冰观测", "上报时限"]
    assert processor.calls == 1
    reloaded.process("原问题", refresh=True)
    assert processor.calls == 2
    version_two = CachedQueryProcessor(processor, directory=tmp_path / "v2")
    version_two.process("原问题")
    assert processor.calls == 3


def test_failed_rewrite_refresh_preserves_previous_result(tmp_path):
    processor = RewriteProcessor()
    cached = CachedQueryProcessor(processor, directory=tmp_path)
    cached.process("原问题")
    before = (tmp_path / "queries.json").read_bytes()

    def fail(query):
        raise RuntimeError("模型调用失败")

    processor.process = fail
    with pytest.raises(RuntimeError):
        cached.process("原问题", refresh=True)
    assert (tmp_path / "queries.json").read_bytes() == before
    assert cached.process("原问题").rewritten_query == "海冰观测"


@pytest.fixture
def embedding_sdk():
    calls = []

    def respond(request):
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(200, json={"data": [
            {"index": i, "embedding": [3, 4]} for i in reversed(range(len(body["input"])))
        ]})

    with OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as sdk:
        yield sdk, calls


def test_query_embeddings_persist_by_text_model_and_version(tmp_path, embedding_sdk):
    sdk, calls = embedding_sdk
    client = EmbeddingClient(sdk, model="Qwen/embedding")
    vector = client.embed_query("海冰观测", cache_directory=tmp_path)
    vector[:] = 0
    np.testing.assert_array_equal(client.embed_query("海冰观测", cache_directory=tmp_path), [3, 4])
    reloaded = EmbeddingClient(sdk, model="Qwen/embedding")
    np.testing.assert_array_equal(reloaded.embed_query("海冰观测", cache_directory=tmp_path), [3, 4])
    assert len(calls) == 1
    reloaded.embed_query("上报时限", cache_directory=tmp_path)
    reloaded.embed_query("海冰观测", cache_directory=tmp_path, refresh=True)
    EmbeddingClient(sdk, model="other-model").embed_query("海冰观测", cache_directory=tmp_path)
    client.embed_query("海冰观测", cache_directory=tmp_path / "v2")
    assert len(calls) == 5
    notes = json.loads((tmp_path / "Qwen-embedding/embeddings.json").read_text(encoding="utf-8"))
    assert notes["model"] == "Qwen/embedding" and notes["dimension"] == 2
    assert set(notes["queries"]) == {"海冰观测", "上报时限"}
    # 文档批量调用不读取查询缓存。
    client.embed_batch(["海冰观测"])
    assert len(calls) == 6


def test_failed_embedding_refresh_preserves_cached_vector(tmp_path, embedding_sdk):
    sdk, calls = embedding_sdk
    client = EmbeddingClient(sdk, model="test")
    client.embed_query("问题", cache_directory=tmp_path)
    path = tmp_path / "test/embeddings.json"
    before = path.read_bytes()

    def fail(text):
        raise EmbeddingError("服务不可用")

    client.embed = fail
    with pytest.raises(EmbeddingError):
        client.embed_query("问题", cache_directory=tmp_path, refresh=True)
    assert path.read_bytes() == before
    np.testing.assert_array_equal(client.embed_query("问题", cache_directory=tmp_path), [3, 4])


@pytest.mark.parametrize("vector", [[0, 0], [float("nan"), 1], [], [[1, 2]]])
def test_invalid_embedding_not_cached(tmp_path, embedding_sdk, vector):
    sdk, calls = embedding_sdk
    client = EmbeddingClient(sdk, model="test")
    client.embed = lambda text: np.asarray(vector, dtype=np.float32)
    with pytest.raises(EmbeddingError):
        client.embed_query("问题", cache_directory=tmp_path)
    assert not (tmp_path / "test/embeddings.json").exists()


def test_atomic_write_failure_keeps_previous_cache(tmp_path, monkeypatch):
    path = tmp_path / "cache.json"
    cache = JsonFileCache(path)
    cache.save({"old": 1})

    def fail(*args):
        raise OSError("模拟磁盘写入失败")

    monkeypatch.setattr("emergency_rag.cache.os.replace", fail)
    with pytest.raises(OSError):
        cache.save({"new": 2})
    assert cache.data == {"old": 1}
    assert json.loads(path.read_text()) == {"old": 1}
    assert list(tmp_path.iterdir()) == [path]


def test_query_caches_can_be_read_in_new_process_without_model_calls(tmp_path, embedding_sdk):
    sdk, calls = embedding_sdk
    EmbeddingClient(sdk, model="test").embed_query("原问题", cache_directory=tmp_path)
    CachedQueryProcessor(RewriteProcessor(), directory=tmp_path / "rewrite").process("原问题")
    # 新进程彻底丢弃内存缓存，API 与 QueryProcessor 都设置为禁止调用。
    code = """
import json,sys
from pathlib import Path
import httpx
from openai import OpenAI
from emergency_rag.clients.embedding import EmbeddingClient
from emergency_rag.retrieval.query.base import QueryProcessor
from emergency_rag.retrieval.query.cached import CachedQueryProcessor
def fail(*args):
    raise AssertionError('cache hit must not call model')
processor=QueryProcessor()
processor.process=fail
root=Path(sys.argv[1])
with OpenAI(api_key='test',http_client=httpx.Client(transport=httpx.MockTransport(fail))) as sdk:
    vector=EmbeddingClient(sdk,model='test').embed_query('原问题',cache_directory=root)
    context=CachedQueryProcessor(processor,directory=root/'rewrite').process('原问题')
print(json.dumps({'vector':vector.tolist(),'queries':context.queries},ensure_ascii=False))
"""
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, encoding="utf-8", check=True)
    assert json.loads(result.stdout) == {"vector": [3, 4], "queries": ["海冰观测", "上报时限"]}


def test_yaml_rewrite_and_dense_caches_skip_remote_calls(tmp_path, monkeypatch, embedding_sdk):
    sdk, calls = embedding_sdk
    client = EmbeddingClient(sdk, model="test")
    monkeypatch.setattr("emergency_rag.clients.embedding.get_embedding_client", lambda: client)
    monkeypatch.setattr("emergency_rag.clients.embedding.settings.embedding_model", "test")
    monkeypatch.setitem(COMPONENT_REGISTRY["query_processor"], "rewrite-test", RewriteProcessor)
    (tmp_path / "rules.json").write_text('[{"rule_id":"1","rule_text":"海冰观测上报时限"}]', encoding="utf-8")
    config = {
        "dataset": {"source": "rules.json", "dataset_name": "test", "index_root": "indexes"},
        "retrieval": {
            "retrievers": [{"name": "dense"}],
            "fusion": "union",
            "query_processor": "rewrite-test",
        },
    }
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    first = load_pipeline(path)
    calls.clear()
    first.retrieve("原问题")
    assert [c["input"] for c in calls] == [["海冰观测"], ["上报时限"]]
    assert first.query_processor.processor.calls == 1
    calls.clear()
    second = load_pipeline(path)
    second.retrieve("原问题")
    assert second.query_processor.processor.calls == 0 and calls == []
    assert (tmp_path / "cache/query_embeddings/test/embeddings.json").is_file()
    assert (tmp_path / "cache/query_processing/rewrite-test/queries.json").is_file()
    client.embed_query("海冰观测", refresh=True)
    assert len(calls) == 1


@pytest.mark.parametrize("cache", [None, {"directory": ""}, {"directory": "cache", "refresh": "false"}, {"directory": "cache", "unknown": 1}])
def test_invalid_cache_yaml_rejected(tmp_path, cache):
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump({
        "dataset": {"source": "missing.json", "dataset_name": "test"},
        "retrieval": {"retrievers": ["bm25"], "fusion": "union", "query_processor": {"name": "default", "cache": cache}},
    }), encoding="utf-8")
    with pytest.raises(PipelineConfigError):
        load_pipeline(path)

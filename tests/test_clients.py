import json
from dataclasses import replace

import httpx
import numpy as np
import pytest
from openai import OpenAI
from zeroentropy import ZeroEntropy

from emergency_rag.clients.chat import ChatLLM, ChatLLMError
from emergency_rag.clients.embedding import EmbeddingClient, EmbeddingError
from emergency_rag.clients.chat import ChatLLMConfig
from emergency_rag.retrieval.rerank.zero_entropy import ZeroEntropyReranker, ZeroEntropyRerankerError


def test_embedding_batches_restore_order(embedding_config):
    calls = []

    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        return httpx.Response(200, json={
            "object": "list", "model": "test-embedding",
            "data": [
                {"object": "embedding", "index": i, "embedding": [float(text), 1.0]}
                for i, text in reversed(list(enumerate(payload["input"])))
            ],
        })

    embedding_config = replace(embedding_config, batch_size=2)
    with OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as sdk:
        client = EmbeddingClient(embedding_config, sdk)
        vectors = client.embed_batch(["1", "2", "3"])
        assert vectors.dtype == np.float32 and vectors.flags.c_contiguous
        np.testing.assert_array_equal(vectors, [[1, 1], [2, 1], [3, 1]])
        assert [len(call["input"]) for call in calls] == [2, 1]
        assert all(call["encoding_format"] == "float" for call in calls)
        assert client.embed("4").shape == (2,)
        assert client.embed_batch([]).shape == (0, 0)
        with pytest.raises(ValueError):
            client.embed(" ")


@pytest.mark.parametrize("data", [
    None, [],
    [{"index": 1, "embedding": [1, 2]}],
    [{"index": 0, "embedding": []}],
    [{"index": 0, "embedding": [float("inf"), 2]}],
    [{"index": 0, "embedding": [1, 2]}, {"index": 0, "embedding": [1, 2]}],
])
def test_embedding_invalid_response(embedding_config, data):
    def respond(request):
        # json= 不允许 inf；用原始 JSON 模拟第三方非规范响应。
        return httpx.Response(200, text=json.dumps({"data": data}))

    with OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as sdk:
        with pytest.raises(EmbeddingError):
            EmbeddingClient(embedding_config, sdk).embed("文档")


def test_embedding_dimension_changes_between_batches(embedding_config):
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0] * len(calls)}]})

    embedding_config = replace(embedding_config, batch_size=1)
    with OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as sdk:
        with pytest.raises(EmbeddingError, match="维度不一致"):
            EmbeddingClient(embedding_config, sdk).embed_batch(["甲", "乙"])


def test_embedding_sdk_retry_is_the_only_retry(embedding_config):
    calls = []

    def respond(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(500, json={"error": {"message": "temporary"}})
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1, 2]}]})

    with OpenAI(api_key="test", max_retries=1, http_client=httpx.Client(transport=httpx.MockTransport(respond))) as sdk:
        EmbeddingClient(embedding_config, sdk).embed("文档")
    assert len(calls) == 2


def test_rerank_batch_alignment_and_full_scores(rerank_config, candidate):
    calls = []

    def respond(request):
        body = json.loads(request.content)
        calls.append(body)
        scores = [0.1, 0.9] if len(calls) == 1 else [0.7]
        return httpx.Response(200, json={"results": [
            {"index": index, "relevance_score": scores[index]}
            for index in reversed(range(len(body["documents"])))
        ]})

    rerank_config = replace(rerank_config, batch_size=2)
    originals = [candidate("a"), candidate("b"), candidate("c")]
    with ZeroEntropy(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as sdk:
        reranker = ZeroEntropyReranker(rerank_config, sdk)
        assert reranker.rerank("问题", []) == []
        results = reranker.rerank("<危险&事故>", originals)
    assert [item.unit_id for item in results] == ["b", "c", "a"]
    assert [item.final_score for item in results] == [0.9, 0.7, 0.1]
    assert all(item.final_rank is None for item in results)
    assert all(item.final_score is None for item in originals)
    assert [body["top_n"] for body in calls] == [2, 1]
    assert "&lt;危险&amp;事故&gt;" in calls[0]["query"]


@pytest.mark.parametrize("results", [
    None, [],
    [{"index": 1, "relevance_score": 0.5}],
    [{"index": 0, "relevance_score": 0.5}, {"index": 0, "relevance_score": 0.6}],
    [{"index": 0, "relevance_score": -0.1}],
    [{"index": 0, "relevance_score": 1.1}],
    [{"index": 0, "relevance_score": float("nan")}],
    [{"index": 0, "relevance_score": "invalid"}],
])
def test_rerank_invalid_response(rerank_config, candidate, results):
    def respond(request):
        return httpx.Response(200, text=json.dumps({"results": results}))

    with ZeroEntropy(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as sdk:
        with pytest.raises(ZeroEntropyRerankerError):
            ZeroEntropyReranker(rerank_config, sdk).rerank("问题", [candidate("a")])


def test_rerank_ties_and_duplicate_input(rerank_config, candidate):
    def respond(request):
        return httpx.Response(200, json={"results": [
            {"index": 0, "relevance_score": 0.5}, {"index": 1, "relevance_score": 0.5},
        ]})

    with ZeroEntropy(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as sdk:
        reranker = ZeroEntropyReranker(rerank_config, sdk)
        assert [item.unit_id for item in reranker.rerank("问题", [candidate("b"), candidate("a")])] == ["a", "b"]
        with pytest.raises(ValueError, match="重复"):
            reranker.rerank("问题", [candidate("a"), candidate("a")])


@pytest.mark.parametrize("client_kind", ["embedding", "rerank", "chat"])
def test_api_errors_have_clear_boundary(embedding_config, rerank_config, candidate, client_kind):
    transport = httpx.MockTransport(lambda request: httpx.Response(401, json={"error": {"message": "unauthorized"}}))
    if client_kind == "rerank":
        with ZeroEntropy(api_key="test", max_retries=0, http_client=httpx.Client(transport=transport)) as sdk:
            with pytest.raises(ZeroEntropyRerankerError, match="API 调用失败"):
                ZeroEntropyReranker(rerank_config, sdk).rerank("问题", [candidate("a")])
    else:
        with OpenAI(api_key="test", max_retries=0, http_client=httpx.Client(transport=transport)) as sdk:
            if client_kind == "embedding":
                with pytest.raises(EmbeddingError, match="API 调用失败"):
                    EmbeddingClient(embedding_config, sdk).embed("问题")
            else:
                cfg = ChatLLMConfig(model="test")
                with pytest.raises(ChatLLMError, match="API 调用失败"):
                    ChatLLM(cfg, sdk).invoke([{"role": "user", "content": "问题"}])


@pytest.mark.parametrize("content,finish,error", [
    ('{"answer":"结果"}', "stop", False),
    ('{"partial":', "length", True),
    ("not json", "stop", True),
    ("[]", "stop", True),
    (None, "stop", True),
])
def test_chat_json_contract(content, finish, error):
    calls = []

    def respond(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "chat-1", "object": "chat.completion", "created": 1, "model": "test-chat",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
        })

    with OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as sdk:
        chat = ChatLLM(ChatLLMConfig(model="test-chat"), sdk)
        if error:
            with pytest.raises(ChatLLMError):
                chat.invoke([{"role": "user", "content": "问题"}])
        else:
            assert chat.invoke([{"role": "user", "content": "问题"}]) == content
    assert calls[0]["response_format"] == {"type": "json_object"}

import json

import httpx
import numpy as np
import pytest
from openai import AuthenticationError, OpenAI
from zeroentropy import ZeroEntropy

from emergency_rag.clients.chat import ChatClient
from emergency_rag.clients.embedding import EmbeddingClient, EmbeddingError
from emergency_rag.retrieval.rerank.zero_entropy import ZeroEntropyReranker, ZeroEntropyRerankerError


def test_embedding_batches_restore_order():
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

    with OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as sdk:
        client = EmbeddingClient(sdk, model="test-embedding", batch_size=2)
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
def test_embedding_invalid_response(data):
    def respond(request):
        # json= 不允许 inf；用原始 JSON 模拟第三方非规范响应。
        return httpx.Response(200, text=json.dumps({"data": data}))

    with OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as sdk:
        with pytest.raises(EmbeddingError):
            EmbeddingClient(sdk, model="test-embedding").embed("文档")


def test_embedding_dimension_changes_between_batches():
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0] * len(calls)}]})

    with OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as sdk:
        with pytest.raises(EmbeddingError, match="维度不一致"):
            EmbeddingClient(sdk, model="test-embedding", batch_size=1).embed_batch(["甲", "乙"])


def test_embedding_sdk_retry_is_the_only_retry():
    calls = []

    def respond(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(500, json={"error": {"message": "temporary"}})
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1, 2]}]})

    with OpenAI(api_key="test", max_retries=1, http_client=httpx.Client(transport=httpx.MockTransport(respond))) as sdk:
        EmbeddingClient(sdk, model="test-embedding").embed("文档")
    assert len(calls) == 2


def test_rerank_batch_alignment_and_full_scores(candidate):
    calls = []

    def respond(request):
        body = json.loads(request.content)
        calls.append(body)
        scores = [0.1, 0.9] if len(calls) == 1 else [0.7]
        return httpx.Response(200, json={"results": [
            {"index": index, "relevance_score": scores[index]}
            for index in reversed(range(len(body["documents"])))
        ]})

    originals = [candidate("a"), candidate("b"), candidate("c")]
    with ZeroEntropy(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as sdk:
        reranker = ZeroEntropyReranker(sdk, model="test-reranker", instruction="评估相关性", batch_size=2)
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
def test_rerank_invalid_response(candidate, results):
    def respond(request):
        return httpx.Response(200, text=json.dumps({"results": results}))

    with ZeroEntropy(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as sdk:
        with pytest.raises(ZeroEntropyRerankerError):
            ZeroEntropyReranker(sdk, model="test-reranker", instruction="评估相关性").rerank("问题", [candidate("a")])


def test_rerank_ties_and_duplicate_input(candidate):
    def respond(request):
        return httpx.Response(200, json={"results": [
            {"index": 0, "relevance_score": 0.5}, {"index": 1, "relevance_score": 0.5},
        ]})

    with ZeroEntropy(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as sdk:
        reranker = ZeroEntropyReranker(sdk, model="test-reranker", instruction="评估相关性")
        assert [item.unit_id for item in reranker.rerank("问题", [candidate("b"), candidate("a")])] == ["a", "b"]
        with pytest.raises(ValueError, match="重复"):
            reranker.rerank("问题", [candidate("a"), candidate("a")])


@pytest.mark.parametrize("client_kind", ["embedding", "rerank", "chat"])
def test_api_errors_have_clear_boundary(candidate, client_kind):
    transport = httpx.MockTransport(lambda request: httpx.Response(401, json={"error": {"message": "unauthorized"}}))
    if client_kind == "rerank":
        with ZeroEntropy(api_key="test", max_retries=0, http_client=httpx.Client(transport=transport)) as sdk:
            with pytest.raises(ZeroEntropyRerankerError, match="API 调用失败"):
                ZeroEntropyReranker(sdk, model="test-reranker", instruction="评估相关性").rerank("问题", [candidate("a")])
    else:
        with OpenAI(api_key="test", max_retries=0, http_client=httpx.Client(transport=transport)) as sdk:
            if client_kind == "embedding":
                with pytest.raises(EmbeddingError, match="API 调用失败"):
                    EmbeddingClient(sdk, model="test-embedding").embed("问题")
            else:
                with pytest.raises(AuthenticationError):
                    ChatClient(sdk).complete(model="test", messages=[{"role": "user", "content": "问题"}])


@pytest.mark.parametrize("content,finish,error", [
    ('{"answer":"结果"}', "stop", False),
    ('{"partial":', "length", False),
    ("应急结束条件如下。", "stop", False),
    ("B", "stop", False),
    ("[]", "stop", False),
    (None, "stop", True),
    ("", "stop", True),
    ("  \n", "stop", True),
])
def test_chat_returns_text_without_imposing_answer_format(content, finish, error):
    calls = []

    def respond(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "chat-1", "object": "chat.completion", "created": 1, "model": "test-chat",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
        })

    with OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as sdk:
        chat = ChatClient(sdk)
        if error:
            with pytest.raises(ValueError, match="empty"):
                chat.complete(model="test-chat", messages=[{"role": "user", "content": "问题"}])
        else:
            assert chat.complete(model="test-chat", messages=[{"role": "user", "content": "问题"}]) == content
    assert calls == [{"model": "test-chat", "messages": [{"role": "user", "content": "问题"}]}]


def test_chat_forwards_explicit_output_options_without_parsing_json():
    calls = []

    def respond(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "chat-1", "object": "chat.completion", "created": 1, "model": "test",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "调用方负责解析"}, "finish_reason": "stop"}],
        })

    messages = [{"role": "system", "content": "按指定格式作答"}, {"role": "user", "content": "问题"}]
    with OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as sdk:
        result = ChatClient(sdk).complete(
            model="test", messages=messages, max_tokens=128, response_format={"type": "json_object"},
        )
    assert result == "调用方负责解析"
    assert calls == [{"model": "test", "messages": messages, "max_tokens": 128, "response_format": {"type": "json_object"}}]


def test_chat_rejects_missing_choices():
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={
        "id": "chat-1", "object": "chat.completion", "created": 1, "model": "test", "choices": [],
    }))
    with OpenAI(api_key="test", http_client=httpx.Client(transport=transport)) as sdk:
        with pytest.raises(ValueError, match="empty"):
            ChatClient(sdk).complete(model="test", messages=[{"role": "user", "content": "问题"}])

"""测试默认禁止真实网络，真实 SDK 通过 MockTransport 接收服务响应。"""

import os
import socket

import pytest

from emergency_rag.clients.embedding import EmbeddingClientConfig
from emergency_rag.retrieval.rerank.zero_entropy import ZeroEntropyRerankerConfig
from emergency_rag.retrieval.models import Candidate


@pytest.fixture(autouse=True)
def isolate_environment_and_network(monkeypatch):
    for name in list(os.environ):
        if name.startswith("RAG_"):
            monkeypatch.delenv(name)

    def forbid_network(*args, **kwargs):
        raise AssertionError("pytest 禁止访问真实网络，请使用 MockTransport")

    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    monkeypatch.setattr(socket, "create_connection", forbid_network)


@pytest.fixture
def database(tmp_path):
    return tmp_path / "emergency.db"


@pytest.fixture
def embedding_config():
    return EmbeddingClientConfig(model="test-embedding")


@pytest.fixture
def rerank_config():
    return ZeroEntropyRerankerConfig(model="test-reranker")


@pytest.fixture
def candidate():
    def create(unit_id, source="bm25", rank=1, score=1.0, query="问题"):
        return Candidate(
            unit_id=unit_id,
            rule_id=unit_id.removeprefix("rule:"),
            text=f"规则 {unit_id}",
            sources=[source],
            metadata={"retrieval": [{"source": source, "query": query, "rank": rank, "score": score}]},
        )
    return create

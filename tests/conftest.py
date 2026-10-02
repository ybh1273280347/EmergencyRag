"""测试默认禁止真实网络，真实 SDK 通过 MockTransport 接收服务响应。"""

import os
import socket
from unittest.mock import Mock

import pytest

from emergency_rag.retrieval.models import Candidate
from emergency_rag.data.indexed_dataset import IndexedDataset
from emergency_rag.data.units.base import Rule
from emergency_rag.clients import chat, choice_qa_client, embedding
from emergency_rag.retrieval.rerank import qwen
from emergency_rag import settings as settings_module


@pytest.fixture(autouse=True)
def isolate_environment_and_network(monkeypatch, tmp_path):
    # 默认测试不读取开发者的真实 .env；自动加载行为在临时目录中单独验证。
    monkeypatch.setattr(settings_module, "find_dotenv", lambda **kwargs: "")
    for name in list(os.environ):
        if name.startswith(("RAG_", "QWEN_RERANKER_", "TYPESAFE_")):
            monkeypatch.delenv(name)

    isolated = settings_module.Settings()
    isolated.query_cache_root = tmp_path / "cache"
    for module in (chat, choice_qa_client, embedding, qwen):
        monkeypatch.setattr(module, "settings", isolated)

    def forbid_network(*args, **kwargs):
        raise AssertionError("pytest 禁止访问真实网络，请使用 MockTransport")

    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    monkeypatch.setattr(socket, "create_connection", forbid_network)


@pytest.fixture
def dataset():
    dataset = Mock(spec=IndexedDataset)
    dataset.rules = {key: Rule(rule_id=key, text=f"完整规则 {key}") for key in ("a", "b", "c")}
    return dataset


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

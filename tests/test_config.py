from dataclasses import FrozenInstanceError, fields, is_dataclass

import pytest

from emergency_rag.clients.chat import ChatLLMConfig
from emergency_rag.clients.embedding import EmbeddingClientConfig
from emergency_rag.retrieval.fusion.rrf import RRFFusionConfig
from emergency_rag.retrieval.fusion.union import UnionFusionConfig
from emergency_rag.retrieval.pipeline import RetrievalPipelineConfig
from emergency_rag.retrieval.rerank.zero_entropy import ZeroEntropyRerankerConfig
from emergency_rag.retrieval.retrievers.bm25 import BM25RetrieverConfig
from emergency_rag.retrieval.retrievers.dense import DenseRetrieverConfig


@pytest.mark.parametrize("config", [
    BM25RetrieverConfig(), DenseRetrieverConfig(),
    RRFFusionConfig(), UnionFusionConfig(),
    EmbeddingClientConfig(model="embedding"), ChatLLMConfig(model="chat"),
    ZeroEntropyRerankerConfig(model="rerank"), RetrievalPipelineConfig(),
])
def test_component_config_is_local_frozen_dataclass(config):
    assert is_dataclass(config)
    assert not hasattr(config, "__dict__")
    assert config.__class__.__module__ != "emergency_rag.config"
    name = fields(config)[0].name
    with pytest.raises(FrozenInstanceError):
        setattr(config, name, getattr(config, name))


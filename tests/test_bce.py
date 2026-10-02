"""BCE 边界使用模型替身，默认测试不加载权重或下载模型。"""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from emergency_rag.retrieval.rerank import bce


@pytest.fixture(autouse=True)
def clear_bce_cache():
    bce.get_rerank_model.cache_clear()
    yield
    bce.get_rerank_model.cache_clear()


def test_bce_scores_all_units_without_filtering_or_rounding(candidate):
    model = Mock()
    model.rerank.return_value = {
        "rerank_ids": [2, 0, 1], "rerank_scores": [0.9123456, 0.5, 0.0012345],
    }
    originals = [candidate("a"), candidate("b"), candidate("c")]
    originals[0].text = "长文原文" * 1000
    reranker = bce.BCEReranker(model, batch_size=8)
    assert reranker.rerank("问题", []) == []
    ranked = reranker.rerank("问题", originals)
    model.rerank.assert_called_once_with(
        query="问题", passages=[item.text for item in originals], batch_size=8,
    )
    model.compute_score.assert_not_called()
    assert [item.unit_id for item in ranked] == ["c", "a", "b"]
    assert [item.final_score for item in ranked] == [0.9123456, 0.5, 0.0012345]
    assert all(item.final_score is None for item in originals)
    assert ranked[1].text == originals[0].text
    assert ranked[0].metadata["retrieval"] == originals[2].metadata["retrieval"]


def test_bce_ties_sort_by_unit_id(candidate):
    model = Mock()
    model.rerank.return_value = {"rerank_ids": [0, 1], "rerank_scores": [0.5, 0.5]}
    ranked = bce.BCEReranker(model).rerank("问题", [candidate("b"), candidate("a")])
    assert [item.unit_id for item in ranked] == ["a", "b"]


@pytest.mark.parametrize("response", [
    None, {},
    {"rerank_ids": [], "rerank_scores": []},
    {"rerank_ids": [1], "rerank_scores": [0.5]},
    {"rerank_ids": [True], "rerank_scores": [0.5]},
    {"rerank_ids": [0, 0], "rerank_scores": [0.5, 0.5]},
    {"rerank_ids": [0], "rerank_scores": [True]},
    {"rerank_ids": [0], "rerank_scores": [-0.1]},
    {"rerank_ids": [0], "rerank_scores": [1.1]},
    {"rerank_ids": [0], "rerank_scores": [float("nan")]},
    {"rerank_ids": [0], "rerank_scores": ["invalid"]},
])
def test_bce_invalid_response(candidate, response):
    model = Mock()
    model.rerank.return_value = response
    with pytest.raises(bce.BCERerankerError):
        bce.BCEReranker(model).rerank("问题", [candidate("a")])


def test_bce_lazy_local_model_cache_and_device_selection(tmp_path, monkeypatch):
    constructors = []

    def construct(**kwargs):
        constructors.append(kwargs)
        return Mock()

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setitem(sys.modules, "BCEmbedding", SimpleNamespace(RerankerModel=construct))
    first = bce.get_rerank_model(model_path=str(tmp_path))
    assert bce.get_rerank_model(model_path=str(tmp_path)) is first
    assert constructors == [{
        "model_name_or_path": str(tmp_path), "device": "cuda", "use_fp16": True, "local_files_only": True,
    }]
    cpu = bce.get_rerank_model(model_path=str(tmp_path), device="cpu")
    assert cpu is not first
    assert constructors[1]["device"] == "cpu"
    assert constructors[1]["use_fp16"] is False


def test_bce_nonpositive_batch_size_rejected():
    with pytest.raises(ValueError, match="batch_size"):
        bce.BCEReranker(Mock(), batch_size=0)

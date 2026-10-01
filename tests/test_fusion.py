import pytest

from emergency_rag.retrieval.fusion.rrf import RRFFusion, RRFFusionConfig
from emergency_rag.retrieval.fusion.union import UnionFusion, UnionFusionConfig


def test_rrf_calculation_dedup_and_score_boundary(candidate):
    a = candidate("a", rank=1, score=8)
    b = candidate("b", rank=2, score=3)
    dense_a = candidate("a", source="dense", rank=2, score=0.8)
    dense_b = candidate("b", source="dense", rank=1, score=0.9)
    results = RRFFusion().fuse([[a, b, a], [dense_b, dense_a]])
    assert [item.unit_id for item in results] == ["a", "b"]
    assert results[0].metadata["fusion"]["score"] == pytest.approx(1 / 61 + 1 / 62)
    assert results[0].sources == ["bm25", "dense"]
    assert [obs["score"] for obs in results[0].metadata["retrieval"]] == [8, 0.8]
    assert [obs["rank"] for obs in results[0].metadata["retrieval"]] == [1, 2]
    assert all(item.final_score is None and item.final_rank is None for item in results)
    assert a.sources == ["bm25"] and "fusion" not in a.metadata


def test_rrf_total_budget(candidate):
    results = RRFFusion(RRFFusionConfig(rank_constant=0, max_candidates=1)).fuse([
        [candidate("a"), candidate("b", rank=2)],
        [candidate("b", source="dense"), candidate("a", source="dense", rank=2)],
    ])
    assert len(results) == 1
    assert results[0].unit_id == "a"
    assert results[0].metadata["fusion"]["score"] == 1.5


def test_union_source_budgets_before_dedup_and_no_refill(candidate):
    sparse = [candidate("a"), candidate("b", rank=2), candidate("c", rank=3)]
    dense = [candidate("a", "dense"), candidate("d", "dense", 2), candidate("e", "dense", 3)]
    fusion = UnionFusion(UnionFusionConfig(per_source_top_k={"bm25": 2, "dense": 1}))
    results = fusion.fuse([sparse, dense])
    assert [item.unit_id for item in results] == ["a", "b"]
    assert results[0].sources == ["bm25", "dense"]
    assert len(results[0].metadata["retrieval"]) == 2
    assert "fusion" not in sparse[0].metadata
    # 调换列表顺序不改变来源预算；每路仍只贡献其配置允许的前缀。
    assert {item.unit_id for item in fusion.fuse([dense, sparse])} == {"a", "b"}


def test_union_no_total_limit_and_round_robin(candidate):
    sparse = [candidate(f"s{i}", rank=i + 1) for i in range(30)]
    dense = [candidate(f"d{i}", "dense", i + 1) for i in range(30)]
    results = UnionFusion().fuse([sparse, dense])
    assert len(results) == 60
    assert [item.unit_id for item in results[:4]] == ["s0", "d0", "s1", "d1"]
    assert all(item.metadata["fusion"]["score"] is None for item in results)


def test_union_overlap_example(candidate):
    sparse = [candidate(str(i), rank=i + 1) for i in range(30)]
    dense = [candidate(str(i), "dense", rank=i - 18 + 1) for i in range(18, 48)]
    assert len(UnionFusion().fuse([sparse, dense])) == 48


def test_union_zero_budget_and_missing_source(candidate):
    fusion = UnionFusion(UnionFusionConfig(per_source_top_k={"bm25": 0}))
    assert fusion.fuse([[candidate("a")], []]) == []
    with pytest.raises(ValueError, match="来源预算"):
        fusion.fuse([[candidate("a", "dense")]])


@pytest.mark.parametrize("fusion", [RRFFusion(), UnionFusion()])
def test_empty_and_conflicting_identity(fusion, candidate):
    assert fusion.fuse([[], []]) == []
    a, b = candidate("a"), candidate("a", "dense")
    b.text = "不同文本"
    with pytest.raises(ValueError, match="不同规则内容"):
        fusion.fuse([[a], [b]])


def test_subqueries_keep_observations(candidate):
    result = RRFFusion().fuse([
        [candidate("a", query="原查询")], [candidate("a", query="子查询")],
    ])[0]
    assert len(result.metadata["retrieval"]) == 2
    assert result.sources == ["bm25"]


def test_union_accepts_caller_defined_sources(candidate):
    fusion = UnionFusion(UnionFusionConfig(per_source_top_k={"archive": 1, "lexical": 2}))
    results = fusion.fuse([
        [candidate("a", "archive"), candidate("ignored", "archive", 2)],
        [candidate("a", "lexical"), candidate("b", "lexical", 2)],
    ])
    assert [item.unit_id for item in results] == ["a", "b"]
    assert results[0].sources == ["archive", "lexical"]


@pytest.mark.parametrize("config", [
    RRFFusionConfig(rank_constant=-1), RRFFusionConfig(max_candidates=0),
    UnionFusionConfig(per_source_top_k={"archive": -1}),
])
def test_invalid_algorithm_budgets_fail_at_component_boundary(config):
    with pytest.raises(ValueError):
        if isinstance(config, RRFFusionConfig):
            RRFFusion(config)
        else:
            UnionFusion(config)

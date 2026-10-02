"""从 YAML 注册名称装配组件，自动准备 Dataset 并返回检索 Pipeline。"""

from pathlib import Path
from typing import Any

import yaml

from emergency_rag.clients import embedding
from emergency_rag.data.pipeline import DatasetPipeline
from emergency_rag.registry import COMPONENT_REGISTRY
from emergency_rag.retrieval.expansion.base import CandidateExpander
from emergency_rag.retrieval.fusion.base import Fusion
from emergency_rag.retrieval.gate.base import EvidenceGate
from emergency_rag.retrieval.pipeline import RetrievalPipeline
from emergency_rag.retrieval.query.base import QueryProcessor
from emergency_rag.retrieval.query.cached import CachedQueryProcessor
from emergency_rag.retrieval.rerank.base import Reranker
from emergency_rag.retrieval.retrievers.base import Retriever
from emergency_rag.retrieval.tokenizer.base import TextTokenizer
from emergency_rag.data.units.base import UnitBuilder


class PipelineConfigError(ValueError):
    """YAML 声明不能用于装配对应阶段。"""


DATASET_FIELDS = {
    "source",
    "dataset_name",
    "index_root",
    "overwrite",
    "unit_builder",
    "tokenizer",
}

RETRIEVAL_FIELDS = {
    "retrievers",
    "fusion",
    "query_processor",
    "expander",
    "reranker",
    "gate",
}


def _build_component(stage: str, declaration: Any, interface: type) -> Any:
    if isinstance(declaration, str):
        declaration = {"name": declaration}

    if not isinstance(declaration, dict):
        raise PipelineConfigError(
            f"{stage}: 使用注册名称或 name/params 对象声明组件"
        )

    unknown = set(declaration) - {"name", "params"}
    if unknown:
        raise PipelineConfigError(
            f"{stage}: 使用注册名称或 name/params 对象声明组件，"
            f"未知字段 {sorted(unknown)!r}"
        )

    name = declaration.get("name")
    params = declaration.get("params", {})

    if not isinstance(name, str) or name not in COMPONENT_REGISTRY[stage]:
        raise PipelineConfigError(f"{stage}: 未注册的组件名称 {name!r}")

    if not isinstance(params, dict):
        raise PipelineConfigError(
            f"{stage}.{name}: params 必须为构造参数对象"
        )

    try:
        component = COMPONENT_REGISTRY[stage][name](**params)
    except (TypeError, ValueError) as exc:
        raise PipelineConfigError(
            f"{stage}.{name}: 构造失败，请检查参数与模型环境配置"
        ) from exc

    if not isinstance(component, interface):
        raise PipelineConfigError(
            f"{stage}.{name}: 必须实现 {interface.__name__}"
        )

    return component


def _load_yaml(path: Path) -> dict:
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except yaml.YAMLError as exc:
        raise PipelineConfigError(f"{path.name}: YAML 格式错误") from exc

    if not isinstance(config, dict) or set(config) != {"dataset", "retrieval"}:
        raise PipelineConfigError("配置必须包含 dataset 和 retrieval 两个对象")

    return config


def _parse_dataset_config(dataset_config: Any) -> dict:
    if not isinstance(dataset_config, dict):
        raise PipelineConfigError("dataset 必须为对象")

    unknown = set(dataset_config) - DATASET_FIELDS
    if unknown:
        raise PipelineConfigError(
            "dataset 只接受数据路径、数据集名称和单元构建、分词组件，"
            f"未知字段 {sorted(unknown)!r}"
        )

    source = dataset_config.get("source")
    dataset_name = dataset_config.get("dataset_name")
    index_root = dataset_config.get("index_root", "data/indexes")
    overwrite = dataset_config.get("overwrite", False)

    if not isinstance(source, str) or not source.strip():
        raise PipelineConfigError("dataset.source 必须为非空路径字符串")
    if not isinstance(index_root, str) or not index_root.strip():
        raise PipelineConfigError("dataset.index_root 必须为非空路径字符串")
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise PipelineConfigError("dataset.dataset_name 必须为非空字符串")
    if type(overwrite) is not bool:
        raise PipelineConfigError("dataset.overwrite 必须为布尔值")

    parsed = {
        "source": source,
        "dataset_name": dataset_name,
        "index_root": index_root,
        "overwrite": overwrite,
    }
    # 组件声明保留到装配阶段，不能在路径参数解析时丢失。
    for stage in ("unit_builder", "tokenizer"):
        if stage in dataset_config:
            parsed[stage] = dataset_config[stage]
    return parsed


def _parse_retrieval_config(retrieval_config: Any) -> dict:
    if not isinstance(retrieval_config, dict):
        raise PipelineConfigError("retrieval 必须为对象")

    unknown = set(retrieval_config) - RETRIEVAL_FIELDS
    if unknown:
        raise PipelineConfigError(
            "retrieval 包含未知阶段；dataset 由加载入口统一注入，"
            f"未知字段 {sorted(unknown)!r}"
        )

    if not isinstance(retrieval_config.get("retrievers"), list):
        raise PipelineConfigError("retrieval 必须声明 retrievers 列表")
    if "fusion" not in retrieval_config:
        raise PipelineConfigError("retrieval 必须声明 fusion")

    return retrieval_config


def load_pipeline(config_path: str | Path) -> RetrievalPipeline:
    """加载一次实验；模型由对应模块共享，作答与评测由调用用例处理。

    YAML 只声明组件注册名称和参数。source、index_root 相对 YAML 所在目录；
    组件装配完成后才开始离线构建，缓存命中和 overwrite 交给 DatasetPipeline。
    """
    path = Path(config_path).expanduser().resolve()
    config = _load_yaml(path)

    dataset_config = _parse_dataset_config(config["dataset"])
    retrieval_config = _parse_retrieval_config(config["retrieval"])

    # Dataset 阶段可选组件
    dataset_kwargs = {}
    for stage, interface in (
        ("unit_builder", UnitBuilder),
        ("tokenizer", TextTokenizer),
    ):
        if stage in dataset_config:
            dataset_kwargs[stage] = _build_component(
                stage, dataset_config[stage], interface
            )

    # Retrieval 必需组件
    retrieval_kwargs = {
        "retrievers": [
            _build_component("retrievers", value, Retriever)
            for value in retrieval_config["retrievers"]
        ],
        "fusion": _build_component(
            "fusion", retrieval_config["fusion"], Fusion
        ),
    }

    # Retrieval 可选组件
    for stage, interface in (
        ("query_processor", QueryProcessor),
        ("expander", CandidateExpander),
        ("reranker", Reranker),
        ("gate", EvidenceGate),
    ):
        if stage in retrieval_config:
            retrieval_kwargs[stage] = _build_component(
                stage, retrieval_config[stage], interface
            )

    # ---- 相对路径解析 ----
    source = Path(dataset_config["source"]).expanduser()
    index_root = Path(dataset_config["index_root"]).expanduser()

    if not source.is_absolute():
        source = path.parent / source
    if not index_root.is_absolute():
        index_root = path.parent / index_root

    # 缓存加载不创建 Embedding；实际构建时与 Dense 工厂取得同一个共享客户端。
    preparation = DatasetPipeline(
        embedding=embedding.get_embedding_client,
        embedding_model=embedding.settings.embedding_model,
        index_root=index_root.resolve(),
        **dataset_kwargs,
    )
    dataset = preparation.prepare(
        source.resolve(),
        dataset_name=dataset_config["dataset_name"],
        overwrite=dataset_config["overwrite"],
    )

    # 查询处理缓存是固定行为；策略名称包含模型与提示词版本时会自动隔离。
    processor = retrieval_kwargs.get("query_processor", QueryProcessor())
    retrieval_kwargs["query_processor"] = CachedQueryProcessor(
        processor,
        directory=embedding.settings.query_cache_root / "query_processing" / processor.name,
    )
    return RetrievalPipeline(dataset=dataset, **retrieval_kwargs)

# Emergency-RAG Retrieval Core

可直接装配的同步检索组件库。各阶段使用独立 ABC，各具体组件在自身模块定义 `dataclass(frozen=True, slots=True)` Config。调用方负责准备数据、索引和模型 SDK，并直接构造 `RetrievalPipeline`。

库没有 CLI、YAML、全局 Settings、通用 Factory 或组件注册表。Pipeline 通过阶段接口执行，不限定检索器来源、Fusion 类型或 Reranker 类型。现有 BM25、Dense、RRF、Union、ZeroEntropy 是可选实现。

## 安装与测试

Python 3.11+，在仓库根目录执行：

```powershell
uv sync --group dev
uv run pytest -q
```

## 调用方直接装配

下面的 `bm25_index`、`bm25_tokenizer`、`unit_ids`、`units_by_id` 都是应用已经准备好的对象。SDK 客户端也由应用创建和关闭，密钥、地址、超时、重试使用应用自己的配置。组件不读取环境变量。

```python
from zeroentropy import ZeroEntropy

from emergency_rag.retrieval.fusion.rrf import RRFFusion, RRFFusionConfig
from emergency_rag.retrieval.pipeline import RetrievalPipeline, RetrievalPipelineConfig
from emergency_rag.retrieval.rerank.zero_entropy import (
    ZeroEntropyReranker,
    ZeroEntropyRerankerConfig,
)
from emergency_rag.retrieval.retrievers.bm25 import BM25Retriever, BM25RetrieverConfig

with ZeroEntropy(
    api_key=settings.ZERO_ENTROPY_API_KEY,
    timeout=60,
    max_retries=2,
) as rerank_client:
    pipeline = RetrievalPipeline(
        retrievers=(
            BM25Retriever(
                index=bm25_index,
                tokenizer=bm25_tokenizer,
                unit_ids=unit_ids,
                units=units_by_id,
                config=BM25RetrieverConfig(top_k=30),
            ),
        ),
        fusion=RRFFusion(RRFFusionConfig(max_candidates=30)),
        reranker=ZeroEntropyReranker(
            client=rerank_client,
            config=ZeroEntropyRerankerConfig(model=settings.RERANKER_MODEL),
        ),
        config=RetrievalPipelineConfig(final_top_k=10),
    )
    result = pipeline.retrieve("危险化学品事故应急结束条件", top_k=10)
    print(result.model_dump_json(indent=2))
```

`settings` 属于调用方应用，库不提供或要求这种配置对象。完整的有类型装配函数见 [examples/assemble_pipeline.py](examples/assemble_pipeline.py)。此示例只选择 BM25；选择 Dense、自定义 Retriever、其他 Fusion 或其他 Reranker 时，直接改变调用方传入的实例即可。

```python
pipeline = RetrievalPipeline(
    retrievers=(archive_retriever, dense_retriever, custom_retriever),
    fusion=my_fusion,
    reranker=my_reranker,
    gate=my_gate,
)
```

扩展组件只需继承相应阶段的 ABC，实现该阶段的方法并提供名称，无需注册、枚举策略或修改 Pipeline。

QueryProcessor、CandidateExpander、CandidateGate、EvidenceProjector 基类直接提供原样返回的默认实现。构造 Pipeline 时这些参数默认为 `None`，表示使用基类默认行为。Chunker、Retriever、Fusion、Reranker 和 TextTokenizer 需要具体算法实现。

## 已准备数据的使用

SQLite 和索引在应用初始化处读取一次，然后复用内存中的对象。Retriever 构造与查询都不加载文件，不访问 SQLite，也没有 `load()` 方法。若应用已经持有构建返回的资源，可以直接注入，完全不经过文件。

已有 BM25 产物可按以下方式在调用方初始化处读取；这是普通显式读取，不属于 Pipeline 或 Retriever 的职责：

```python
from pathlib import Path
from pydantic import TypeAdapter

from emergency_rag.data.repository import RuleRepository
from emergency_rag.retrieval.retrievers.bm25_backend import Tokenizer, bm25s
from emergency_rag.retrieval.tokenizer.jieba import JiebaTokenizer

units = RuleRepository(Path("data/processed/emergency.db")).load_search_units()
units_by_id = {unit.unit_id: unit for unit in units}
directory = Path("data/indexes/bm25")
bm25_index = bm25s.BM25.load(str(directory), load_corpus=False)
bm25_tokenizer = Tokenizer(splitter=JiebaTokenizer().tokenize, stopwords=[], stemmer=None)
bm25_tokenizer.load_vocab(str(directory))
unit_ids = TypeAdapter(list[str]).validate_json(
    (directory / "unit_ids.json").read_text(encoding="utf-8"),
)
```

FAISS 也直接使用调用方已有的 `IndexFlatIP`、行映射及知识单元。Windows Unicode 文件路径可通过读取序列化字节处理：

```python
import faiss
import numpy as np

from emergency_rag.clients.embedding import EmbeddingClient, EmbeddingClientConfig
from emergency_rag.retrieval.retrievers.dense import DenseRetriever, DenseRetrieverConfig

# embedding_sdk 是应用创建的 OpenAI SDK 客户端。
embedding = EmbeddingClient(
    client=embedding_sdk,
    config=EmbeddingClientConfig(model=settings.EMBEDDING_MODEL),
)
directory = Path("data/indexes")
dense_index = faiss.deserialize_index(
    np.frombuffer((directory / "faiss.index").read_bytes(), dtype="uint8"),
)
dense_ids = TypeAdapter(list[str]).validate_json(
    (directory / "faiss_mapping.json").read_text(encoding="utf-8"),
)
dense_retriever = DenseRetriever(
    index=dense_index,
    unit_ids=dense_ids,
    units=units_by_id,
    embedding=embedding,
    config=DenseRetrieverConfig(top_k=30),
)
```

已有 SQLite、BM25S、FAISS 和映射格式继续可用。Retriever 只检查注入索引的行数、映射和可解析的知识单元，不读取构建说明、不比较数据库全库顺序、不计算语料指纹。

## 离线构建

`scripts` 提供普通 Python 函数，没有命令行参数解析，也不创建 SDK 或读取应用配置。BM25、Dense 的构建和保存分别调用，不要求同时存在两路索引。

```python
from pathlib import Path

from scripts.prepare_rules import prepare_rules
from scripts.build_indexes import (
    build_bm25_index, build_dense_index,
    save_bm25_index, save_dense_index,
)

# 已有数据库可以直接用；仅初次准备或显式重建时调用。
prepare_rules(
    Path("datasets/初赛规则集rules1.json"),
    Path("data/processed/emergency.db"),
    overwrite=True,
)
units = RuleRepository(Path("data/processed/emergency.db")).load_search_units()

bm25_index, bm25_tokenizer, unit_ids = build_bm25_index(units, JiebaTokenizer())
# 构建返回值可直接注入 BM25Retriever；保存是可选的独立操作。
save_bm25_index(bm25_index, bm25_tokenizer, unit_ids, Path("data/indexes/bm25"), overwrite=True)

# 使用调用方已经创建的 embedding；只使用 BM25 时不执行以下代码。
dense_index, dense_ids = build_dense_index(units, embedding)
save_dense_index(
    dense_index, dense_ids, Path("data/indexes/dense"),
    embedding_model=embedding.model, overwrite=True,
)
```

目录路径由调用方指定。示例将 Dense 放在独立目录，避免替换 Dense 产物时覆盖 BM25；已有的 `data/indexes/faiss.index` 仍可直接读取。数据库使用事务，重复 ID、非法格式和空白文本会拒绝入库。保存索引完成全部临时写入才替换目标目录，失败保留旧产物。`overwrite` 默认为 `False`。

## 生命周期和预算

```text
QueryProcessor.process
→ 各 Retriever.retrieve
→ Fusion.fuse
→ CandidateExpander.expand
→ Reranker.rerank
→ CandidateGate.filter
→ EvidenceProjector.project
→ 最终 Top-K、连续 final_rank
→ RetrievalResult
```

| 所属 Config | 参数与默认值 | 作用 |
|---|---|---|
| BM25RetrieverConfig | top_k=30；k1=1.5、b=0.75 | 召回预算；k1/b 用于离线构建 |
| DenseRetrieverConfig | top_k=30 | Dense 召回预算 |
| RRFFusionConfig | rank_constant=60；max_candidates=60 | RRF 常数及融合后总上限 |
| UnionFusionConfig | per_source_top_k={bm25:30,dense:30} | 各来源去重前的预算，可配置任意来源名 |
| EmbeddingClientConfig | model；batch_size=64 | Embedding 模型及请求拆分 |
| ZeroEntropyRerankerConfig | model；batch_size=64；instruction | 重排模型、请求拆分及相关性指令 |
| ChatLLMConfig | model；temperature=0 | 独立 Chat JSON 客户端 |
| RetrievalPipelineConfig | final_top_k=10 | 投影后的最终数量 |

Pipeline 调用 `retriever.retrieve(query)`，预算由各 Retriever 拥有；显式调用 `retriever.retrieve(query, top_k=...)` 只覆盖本次召回。`Fusion.fuse(result_sets)` 不接收公共预算。`pipeline.retrieve(query, top_k=...)` 只影响最终返回数量。

RRF 使用 `Σ 1/(rank_constant + rank)`，同一结果集重复单元只计一次，排序后按自身总预算截断，平分按 unit_id 排序。Union 每路先取配置前缀，再交替合并去重，不比较不同召回分数，不补齐重叠候选，不施加总上限。两路各 30 条、重叠 12 条时，Union 输出 48 条。

## 分数和错误边界

召回和融合不写 `Candidate.final_score`；它只承载重排相关性分数。`metadata["retrieval"]` 保留各来源的查询、原始排名和分数；`metadata["fusion"]` 保留融合排名及分数。排名从 1 开始，最终截断发生在 Projector 之后。

ZeroEntropy 按批内 index 恢复候选，完整打分排序，拒绝缺失、重复、越界 index 及非有限或 [0,1] 外分数。Embedding 恢复响应顺序并返回 float32；Dense 在文档构建及查询时 L2 Normalize，拒绝零向量、非有限数值和维度不匹配。超时和重试由应用创建的 SDK 处理，Pipeline 不叠加重试。

返回 metadata 包含总耗时、各路召回数量、融合与最终数量，以及启用的检索器、融合器、重排器名称。

## 当前验收状态

800 条初赛规则已写入 `data/processed/emergency.db`，一条 Rule 对应一条 SearchUnit，保留原文。已有 BM25 索引在 `data/indexes/bm25`，可直接复用。本轮不会因架构调整重建已有产物。

测试使用真实 SQLite、BM25S、FAISS 和模型 SDK HTTP MockTransport，默认禁止网络。覆盖调用方装配、自定义组件、None 默认阶段、独立预算、索引保存及恢复、查询不访问数据文件、模型响应对齐及失败回滚。

真实 Embedding 索引和 ZeroEntropy 端到端验收仍待调用方提供远程模型客户端；mock 测试不替代该验收。本阶段不实现答案生成或统一评测。

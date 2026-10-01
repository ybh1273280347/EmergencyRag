# Emergency-RAG Retrieval Core

规则知识库的同步检索组件库。Rule 通过可插拔 UnitBuilder 生成 SearchUnit，离线使用增强文本构建双路索引。在线先对 Unit 召回和重排，再按 rule_id 聚合成完整规则证据。最终 Top-K 按规则计数，结果由调用方传给作答模型。

各阶段使用独立 ABC，简单参数直接通过构造函数注入。RetrievalPipeline 是 dataclass，Dataset 只注入一次；调用方选择 Retriever、Fusion、Reranker 和 EvidenceGate，无需注册、CLI、YAML、全局 Settings 或 Factory。

## 数据与文本契约

| 模型 | 内容 |
| --- | --- |
| Rule | rule_id、完整原文 text、业务 metadata |
| SearchUnit | unit_id、所属 rule_id、单元原文 text、必填 index_text、单元 metadata |
| IndexedDataset | rules、units、已加载的索引资源，以及准备阶段的 tokenizer |
| Candidate | 中间 Unit 命中、重排分数、来源及召回记录；没有索引文本或最终排名 |
| RuleEvidence | 完整规则、最高重排分数、最终排名、来源、全部 matched_units 及 Rule metadata |
| RetrievalResult | 原始 query、evidence: list[RuleEvidence]、运行统计 |

完整 Rule 与 SubRule 可以放在同一个 Dataset。默认整条规则的 Unit ID 为 `rule:<rule_id>`；自定义策略可以使用 `sub_rule:<rule_id>:<position>`。在线不解析 ID 前缀，始终通过显式 rule_id 关联 Dataset.rules。

- Unit.text 是单元原文，用于重排和命中记录。
- Unit.index_text 用于 BM25 分词和文档 embedding，可以加入 Domain、Topic 或其他上下文。
- RuleEvidence.text 从 Dataset 的完整 Rule 取得；即使索引只有 SubRule，也返回完整原文。
- 作答模型接收最终完整规则文本，索引增强文本不进入结果。

原始规则文件为非空 JSON 数组，每项包含字符串 rule_id 和 rule_text，支持可选的 metadata 对象。Domain、Topic 等业务信息放入 metadata，库不自动分类或生成这些内容。

初赛检索语料是 `datasets/初赛规则集rules1.json` 中的 800 条规则；`datasets/初赛验证集dev.json` 提供查询与答案，不作为规则知识库。

## 自动准备 Dataset

Python 3.11+，使用 uv 管理依赖：

```powershell
uv sync --group dev
```

```python
from pathlib import Path

from emergency_rag.data.pipeline import DatasetPipeline
from emergency_rag.unit_building.rule import RuleUnitBuilder
from emergency_rag.retrieval.tokenizer.jieba import JiebaTokenizer

dataset = DatasetPipeline(
    embedding=embedding_client,
    index_root=Path("data/indexes"),
    unit_builder=RuleUnitBuilder(),
    tokenizer=JiebaTokenizer(),
).prepare(
    Path("datasets/初赛规则集rules1.json"), dataset_name="preliminary",
)
```

DatasetPipeline 流程：

1. 根据 dataset_name、unit_builder.name 和 tokenizer.name 确定目录。
2. 目录存在时直接加载完整规则、单元和已有索引，跳过源文件读取、单元构建和文档 embedding。
3. 目录不存在时读取 Rule、调用 UnitBuilder、合并 metadata、固定构建 BM25 与 Dense，并返回 IndexedDataset。

UnitBuilder 接口是 `build(rules: list[Rule]) -> list[SearchUnit]`，负责单元划分和索引文本增强。默认 RuleUnitBuilder 一条规则生成一个 Unit，text 与 index_text 都等于原文。自定义实现可以只生成 SubRule，也可以同时生成完整 Rule 和 SubRule，不需要 Config 类或注册机制。

准备流程合并 Unit metadata 的顺序为：Rule metadata → Unit metadata 覆盖同名字段 → 写入构建来源。嵌套数据独立复制，不修改输入对象。例如：

```python
{
    "domain": "危险化学品",
    "topic": "应急结束",
    "position": 1,
    "dataset": "preliminary",
    "unit_builder": {"strategy": "rule-context"},
}
```

策略名使用小写英文，可包含数字和连字符。不同划分、增强或分词实验应使用不同 name；相同名称下更换源数据或 embedding 模型时，通过 overwrite=True 显式重建。

```text
data/indexes/
  preliminary-rule-jieba/
    rules.json                  完整规则及业务 metadata
    units.json                  text、index_text 和单元 metadata
    index_meta.json             构建说明，在线不依赖
    bm25/
      ...                       BM25S 矩阵、参数及词表
      unit_ids.json             索引行对应的 unit_id
    dense/
      faiss.index               离线归一化 IndexFlatIP
      faiss_mapping.json        索引行对应的 unit_id
      index_meta.json           Dense 构建说明
  preliminary-rule-context-jieba/
  semifinal-rule-jieba/
```

规则、单元和双路索引在同一次构建中写入临时目录，整体成功才替换目标目录；模型、写入或替换失败保留旧产物。数据格式、重复 ID、空白文本、Unit 到 Rule 的关联以及文档向量正确性由离线边界保证，检索器不重复校验。

已有目录读取失败直接报错，不自动重建。旧产物缺少 rules.json 或 index_text 时，必须使用 overwrite=True 重建，无旧格式兼容层。构建说明只供查看，不进行语料指纹、模型地址或全库顺序审计。

IndexedDataset 加载时读取并缓存已有索引、词表和映射。BM25 查询使用准备阶段携带的同一个 tokenizer，不接受独立 tokenizer 注入。在线只复用内存资源；Dense 仅计算查询 embedding，在线与离线应使用相同 embedding 模型。

data 下保留 models、pipeline、indexing 三个模块。直接构建入口为 `build_dataset_indexes(units, index_root, rules=rules, tokenizer=tokenizer, embedding=embedding)`，必须提供完整规则快照。

## 装配检索组件

```python
from emergency_rag.retrieval.fusion.rrf import RRFFusion
from emergency_rag.retrieval.pipeline import RetrievalPipeline
from emergency_rag.retrieval.retrievers.bm25 import BM25Retriever
from emergency_rag.retrieval.retrievers.dense import DenseRetriever

pipeline = RetrievalPipeline(
    dataset=dataset,
    retrievers=(
        BM25Retriever(top_k=30),
        DenseRetriever(embedding=embedding_client, top_k=30),
    ),
    fusion=RRFFusion(k=60, top_k=60),
)
result = pipeline.retrieve("危险化学品事故应急结束条件", top_k=10)
for evidence in result.evidence:
    print(evidence.final_rank, evidence.rule_id, evidence.final_score)
    print(evidence.text)
```

Retriever 接口仍为 retrieve(query, dataset)，初始化决定召回预算，查询时不能覆盖。Pipeline 向所有检索器传同一个 Dataset；同一个 Retriever 可以在不同 Pipeline 中复用。

默认 Reranker 原样返回 Unit 候选，便于无模型评分测试。需要完整评分时注入 ZeroEntropyReranker：

```python
from emergency_rag.retrieval.rerank.zero_entropy import ZeroEntropyReranker

pipeline.reranker = ZeroEntropyReranker(
    client=rerank_sdk,
    model=rerank_model,
    instruction="请评估规则片段与问题的相关性。",
    batch_size=64,
)
```

模型地址、密钥、超时和重试由调用方构造 SDK 时配置。EmbeddingClient 的模型与批大小直接传入，例如 `EmbeddingClient(embedding_sdk, model=embedding_model, batch_size=64)`。`.env.example` 仅为应用参考，库不读取环境变量。

## 规则聚合与独立预算

```text
QueryProcessor
→ 各 Retriever.retrieve(query, pipeline.dataset)
→ Fusion（按 unit_id 合并）
→ Expansion
→ Reranker（完整评分 Unit.text）
→ 按 rule_id 聚合 RuleEvidence
→ EvidenceGate
→ Top-K 条规则
→ 连续 final_rank 与 RetrievalResult
```

Fusion 保留同一 Rule 的不同 Unit，避免提前丢失重排机会。RRF 使用自己的融合总预算；Union 交替合并各检索器全部有序结果，不再排序或截断。两路各 30 个 Unit、重叠 12 个 Unit 时，Union 输出 48 个 Unit。

| 组件 | 参数及默认值 | 预算含义 |
| --- | --- | --- |
| BM25Retriever | top_k=30 | BM25 召回 Unit 数 |
| DenseRetriever | top_k=30 | Dense 召回 Unit 数 |
| RRFFusion | k=60；top_k=60 | RRF 常数及融合 Unit 总上限 |
| UnionFusion | 无配置 | 合并所有已召回 Unit |
| ZeroEntropyReranker | model；instruction；batch_size=64 | 批大小只拆分请求，不限制总候选数 |
| RetrievalPipeline.retrieve | top_k=10 | Gate 后最多返回的不同 Rule 数 |

按 rule_id 分组后，规则分数取本次命中 Unit 的最高重排分数，按分数降序；平分时按 rule_id 排序。只使用实际命中，不补查兄弟单元，不重新重排完整规则。例如三个规则 37 的单元分数为 0.95、0.90、0.88，规则 463 的单元分数为 0.85，Top-2 返回完整规则 37 和 463，分数分别为 0.95、0.85。

无打分模式保持规则首次出现顺序，final_score=None，不把召回或融合得分当作相关性分数。EvidenceGate 接收聚合后的完整规则，默认原样返回；自定义 Gate 按规则分数、排名或业务信息筛选。Top-K 截断后写入连续排名。

matched_units 保留全部命中片段及分数、来源和 metadata。规则 metadata 来自完整 Rule，片段位置等信息留在 matched_units 内。结果只提供 evidence，没有 candidates 兼容别名，不再提供证据投影阶段。

QueryContext 保存 original_query、rewritten_query 和 sub_queries。queries 属性组合主查询与子查询，Pipeline 逐个召回；重排和 result.query 使用 original_query。运行 metadata 保存总耗时、各路召回数、融合 Unit 数、聚合规则数、最终规则数及组件名称。

## 一次实验与模型作答

[examples/preliminary_experiment.py](examples/preliminary_experiment.py) 是唯一实验示例。UnitBuilder、Tokenizer 和检索组件集中配置，Dataset 自动构建或复用，执行后返回检索结果与答案：

```python
from emergency_rag.clients.chat import ChatClient
from examples.preliminary_experiment import run_experiment

result, answer = run_experiment(
    embedding=embedding_client,
    chat=ChatClient(chat_sdk),
    chat_model=chat_model,
    question="危险化学品事故应急结束需要满足哪些条件？",
    instructions="请根据参考规则回答问题，并注明规则编号。",
    top_k=10,
)
```

作答上下文只使用 result.evidence 中完整规则原文与规则 ID。选择题将选项放入 question，并在 instructions 中要求返回选项；问答题可以要求文字说明。

ChatClient 只发送调用方 messages 并检查非空文本，不注入业务提示词、不强制 JSON、不解析业务响应。max_tokens、response_format 仅在显式传入时发送；SDK 错误原样抛出，超时与重试由 SDK 配置。

## 验证与真实模型验收

```powershell
uv run pytest -q
```

测试默认禁止真实网络，使用实际 BM25S、FAISS 和 SDK HTTP MockTransport。覆盖默认 800 条规则、整条规则与子规则共存、仅子规则索引、metadata、增强文本索引、原文重排、规则最高分聚合、规则级 Gate 和 Top-K、完整命中记录、缓存复用、显式重建、失败保留旧产物，以及问答题和选择题的完整实验调用。

正式双路产物由真实实验首次生成。调用方需提供 Embedding、Chat 客户端；使用 ZeroEntropy 重排时提供对应 SDK。pytest 的 mock 响应验证代码与调用契约，不替代真实模型链路验收。

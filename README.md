# Emergency-RAG Retrieval Core

规则知识库的同步检索组件库。原始数据通过 Chunker 生成 SearchUnits，离线构建数据集目录，再初始化检索组件并直接查询。最终候选由调用方传给作答模型。切换初赛或复赛数据集时，调用方只需更换目录。

各阶段使用独立 ABC。当前组件的参数都较简单，直接通过构造函数注入。Pipeline 是 dataclass，调用方直接选择并注入 Retriever、Fusion、Reranker 等组件。

src 中的 DatasetPipeline 把读取规则、可插拔 Chunker、双路索引构建和 IndexedDataset 加载串成一条流水线。开始处理之前先确定目录，已有目录直接加载；目录缺失时才读取规则并构建。读取与流程放在 data/pipeline.py，索引构建放在 data/indexing.py。

## 数据集目录

```text
data/indexes/
  preliminary-rule-jieba/
    units.json                 完整知识单元快照
    index_meta.json            构建说明，在线不依赖
    bm25/
      ...                      BM25S 矩阵、参数及词表
      unit_ids.json            索引行对应的 unit_id
    dense/
      faiss.index              离线生成的单位向量 IndexFlatIP
      faiss_mapping.json       索引行对应的 unit_id
      index_meta.json          Dense 构建说明
  preliminary-sentence-jieba/  使用 sentence 分块策略、jieba 分词器的初赛产物
  preliminary-rule-characters/ 使用 rule 分块策略、characters 分词器的初赛产物
  semifinal-rule-jieba/        使用 rule 分块策略、jieba 分词器的复赛产物
```

完整构建入口始终生成 BM25 和 Dense，每个数据集目录的知识单元和两路索引由同一次离线构建生成。在线 Pipeline 仍由调用方选择检索组件，可只使用其中一路。

IndexedDataset 保存知识单元快照和准备流水线使用的 tokenizer，加载时一次性读取目录中已有的索引、词表和映射并缓存，供多个检索器共用。BM25 从 Dataset 取得分词器，不接受独立 tokenizer 参数。查询时使用已经加载的对象，不读取索引文件，不重建索引。units.json 是该次索引对应的知识单元快照，无需额外数据库中转。

通常由 DatasetPipeline.prepare 返回 IndexedDataset，再将对象注入 RetrievalPipeline。Pipeline 向所有检索器统一传递同一份 Dataset，不需要实验代码逐个绑定数据集、判断目录、手动分块或拼接索引路径。

初赛目录的检索语料是 `datasets/初赛规则集rules1.json` 中的 800 条规则；`datasets/初赛验证集dev.json` 提供 500 条查询，用于验证检索结果。

## 调用方直接装配

```python
from pathlib import Path

from emergency_rag.data.pipeline import DatasetPipeline
from emergency_rag.retrieval.fusion.rrf import RRFFusion
from emergency_rag.retrieval.pipeline import RetrievalPipeline
from emergency_rag.retrieval.rerank.base import Reranker
from emergency_rag.retrieval.retrievers.bm25 import BM25Retriever
from emergency_rag.retrieval.tokenizer.jieba import JiebaTokenizer

# 根据数据集、分块策略和分词策略自动构建或复用目录。
dataset = DatasetPipeline(
    embedding=embedding_client,
    tokenizer=JiebaTokenizer(),
).prepare(
    Path("datasets/初赛规则集rules1.json"), dataset_name="preliminary",
)
pipeline = RetrievalPipeline(
    dataset=dataset,
    retrievers=(
        BM25Retriever(
            top_k=30,
        ),
    ),
    fusion=RRFFusion(top_k=30),
    reranker=Reranker(),
)
result = pipeline.retrieve("危险化学品事故应急结束条件", top_k=10)
```

Pipeline 接口是 `retrieve(query, top_k=10)`。Reranker 默认原样返回，省略 reranker 参数也使用这一行为，方便检索测试；需要模型评分时可直接注入 ZeroEntropyReranker。一次完整实验见 [examples/preliminary_experiment.py](examples/preliminary_experiment.py)，包含 Dataset 准备、组件装配、检索和模型作答。

需要 Dense 召回时，选择对应组件加入 retrievers：

```python
from emergency_rag.retrieval.retrievers.dense import DenseRetriever

dense = DenseRetriever(
    embedding=embedding_client,
    top_k=30,
)
candidates = dense.retrieve(query, dataset)
```

Dense 在线只调用一次查询 embedding，然后归一化并搜索已加载的 FAISS 索引；文档 embedding 在离线构建时计算。在线与离线应使用相同的 embedding 模型。BM25 的查询分词器从 Dataset 取得，与离线构建使用同一个 TextTokenizer 实例。

Dataset 注入 Pipeline；Retriever 构造函数只接收自身参数与模型依赖，统一接口为 retrieve(query, dataset)。同一个 Retriever 可以用于不同 Pipeline，数据和索引资源由各自的 Dataset 持有。模型地址、密钥、超时和重试由调用方创建 SDK 时配置；模型名、批大小和重排指令直接传给组件构造函数。例如 EmbeddingClient(embedding_sdk, model=embedding_model, batch_size=64)。`.env.example` 是可选的应用环境配置参考，库不读取环境变量。

## 离线构建

Python 3.11+，使用 uv 管理依赖：

```powershell
uv sync --group dev
```

调用方选择 Chunker，准备一次数据集，然后直接装配检索组件：

```python
from pathlib import Path

from emergency_rag.data.pipeline import DatasetPipeline
from emergency_rag.chunking.rule import RuleChunker
from emergency_rag.retrieval.tokenizer.jieba import JiebaTokenizer

# embedding_client 由应用创建，也可供后续 DenseRetriever 使用。
preparation = DatasetPipeline(
    index_root=Path("data/indexes"),
    chunker=RuleChunker(),
    tokenizer=JiebaTokenizer(),
    embedding=embedding_client,
)
dataset = preparation.prepare(
    Path("datasets/初赛规则集rules1.json"),
    dataset_name="preliminary",
)
# 目录为 data/indexes/preliminary-rule-jieba；存在时直接加载，不调用 embedding。
sparse = BM25Retriever()
dense = DenseRetriever(embedding=embedding_client)
# 接着由调用方选择 fusion、reranker 等，装配 RetrievalPipeline。
# Dataset 只在 RetrievalPipeline(dataset=dataset, ...) 注入一次。
```

DatasetPipeline 使用默认 RuleChunker、JiebaTokenizer 和 data/indexes 根目录，也支持显式注入。embedding 是双路构建所需依赖。prepare 始终返回 IndexedDataset，流程如下：

1. 根据 dataset_name、chunker.name 和 tokenizer.name 确定目录。
2. 目录存在时直接加载并携带当前 tokenizer，跳过原始文件读取、分块及文档 embedding。
3. 目录不存在时读取规则、分块，用当前 tokenizer 构建 BM25，同时构建 Dense，再将同一 tokenizer 随 IndexedDataset 返回。

需要重建时显式传入 `overwrite=True`。知识单元和两路索引先写入临时目录，整体成功后才替换目标；模型或写入失败保留旧目录。已有目录读取失败直接报错，不自动重建或覆盖。

DatasetPipeline 直接读取规则并调用 Chunker，保留其产生的 metadata，并加入：

```python
{"dataset": "preliminary", "chunker": {"strategy": "rule"}}
```

目录名称为 `<dataset>-<chunker>-<tokenizer>`，不再单独传入目录名。准备流水线在读取源文件前用 dataset_name、chunker.name 和 tokenizer.name 定位目录；生成 units 后将数据集与分块信息写入 metadata，构建入口结合 tokenizer.name 按同一规则确定路径。使用 preliminary 表示初赛，semifinal 表示复赛；名称使用小写英文，可包含数字和连字符。例如不同窗口实验可分别使用 window-256、window-512。同批 units 必须来自同一数据集和分块策略，构建入口在写入前检查这些来源字段。index_meta.json 记录使用的 tokenizer 策略名，在线仍不依赖该构建说明启动。

数据格式、重复 ID、空白文本由原始数据处理和离线构建检查，文档向量的有效性、数量及归一化由离线构建负责。加载目录使用普通文件读取，不做语料指纹、模型地址或全库顺序审计。构建说明不影响启动或查询。

data 保留三个文件：models.py 定义 Rule、SearchUnit、IndexedDataset，并提供 Dataset 的索引加载与目录命名；pipeline.py 包含原始规则读取和 DatasetPipeline；indexing.py 构建并保存索引。DatasetPipeline 直接执行读取、分块和来源 metadata 写入。需要单独分块时可直接调用 chunker.chunk(read_rules(source))。旧离线脚本已删除，实验直接使用 DatasetPipeline，无需单独执行构建脚本。

融合共用的候选合并工具位于 retrieval/fusion/utils/merge.py，由 RRF 和 Union 调用。

缓存复用只依据目录存在。更换分词器名称会自动使用另一个目录。不同分块参数或分词配置的实验应提供不同 name；在同名目录下更换原始数据、分词参数或 embedding 模型时，需要显式 overwrite。

## 生命周期与独立预算

```text
QueryProcessor
→ 各 Retriever.retrieve(query, pipeline.dataset)
→ Fusion
→ Expansion
→ Reranker
→ Gate
→ Top-K 截断
→ Projector
→ 连续 final_rank 与 RetrievalResult
```

Retriever 的 top_k 在初始化时指定，retrieve 不接收覆盖参数。Pipeline 的 top_k 只截断通过 Gate 的有序候选，不改变召回、融合或重排数量。

| 所属组件                  | 参数及默认值                        | 作用                             |
| ------------------------- | ----------------------------------- | -------------------------------- |
| BM25Retriever             | top_k=30                            | BM25 召回数量                    |
| DenseRetriever            | top_k=30                            | Dense 召回数量                   |
| RRFFusion                 | k=60；top_k=60                      | RRF 常数及融合后总上限           |
| UnionFusion               | 无配置                              | 交替合并全部召回结果并去重        |
| EmbeddingClient           | model；batch_size=64                | 模型与请求拆分                   |
| ZeroEntropyReranker       | model；instruction；batch_size=64   | 模型、相关性指令和请求拆分       |
| Reranker                  | 无参数                              | 默认保持候选顺序与分数           |
| ChatClient.complete       | model；messages；可选输出参数       | 通用文本请求，作答格式由调用方决定 |
| Pipeline.retrieve         | top_k=10                            | Gate 后有序候选的截断数量        |

RRF 去重计分后按总预算截断。Union 忠实使用各检索器已经排好的全部结果，交替合并去重，不重新排序或截断。每路数量由检索器初始化参数决定；两路各返回 30 条、重叠 12 条时输出 48 条。

QueryContext 保存 original_query、可选的 rewritten_query 和 sub_queries。queries 属性返回主查询与子查询列表，主查询优先使用 rewritten_query，未改写时使用 original_query。Pipeline 对该列表执行召回，重排仍使用 original_query。

召回与融合不写 final_score。各路查询、原始得分和排名保存在 metadata.retrieval，融合信息保存在 metadata.fusion。默认 Reranker 原样返回输入列表，不排序、不打分，检索结果保留融合顺序；ZeroEntropyReranker 完整打分排序，final_score 只承载重排相关性分数。返回 metadata 包含总耗时、各路召回数量、融合及最终数量和启用组件名称。

默认阶段由其 ABC 提供原样返回行为，使用 Pipeline 字段的 default_factory 创建。Pipeline 没有 Config 或手写 __init__。自定义 Retriever、Fusion、Reranker、Gate 等实现可直接注入，无需注册；库没有 CLI、YAML、全局 Settings 或通用 Factory。

## 检索结果用于作答

完整调用顺序为：DatasetPipeline（已有目录直接加载，或读取规则 → Chunker → SearchUnits → 离线双路索引）→ IndexedDataset → RetrievalPipeline（向各路传递同一 Dataset）→ 最终候选 → 作答模型。RetrievalPipeline 返回 RetrievalResult；模型作答由调用方用例执行，不加入检索阶段。

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

[examples/preliminary_experiment.py](examples/preliminary_experiment.py) 是 examples 中唯一的实验。Chunker、Tokenizer、Retriever、Fusion 和 Reranker 的选择集中在实验函数中；执行后直接返回检索结果与答案。Dataset 自动构建或复用，最终候选文本和规则 ID 作为参考资料交给模型。选择题可把选项放入 question，并在 instructions 中要求返回选项；问答题可要求文字说明。

ChatClient 仅发送原样 messages 并检查返回文本非空，不注入提示词、不强制 JSON、不解析业务响应。complete 的 model、messages 由本次调用指定，max_tokens 和 response_format 仅在显式传入时发送。需要结构化答案时，调用方指定 response_format 并解析返回文本。SDK 错误原样抛出，重试与超时由调用方创建 SDK 时配置。

## 验证与当前产物

```powershell
uv run pytest -q
```

测试默认禁止网络，使用真实 BM25S、FAISS 和 SDK HTTP MockTransport。覆盖缓存命中跳过离线处理、缺失时自动构建、显式重建、缓存读取失败直接报错、可替换分块器、Tokenizer 随 Dataset 传递、不同 Tokenizer 的目录隔离、来源 metadata、确定性目录名、双路构建、混合来源拒绝、目录切换、保存与加载一致性、查询不读文件或建索引、Dense 只计算查询向量、独立预算、模型响应对齐、构建失败保留旧目录及最终候选交给作答模型。

旧 BM25 索引、旧 preliminary 目录和旧验证报告已删除。首次实验将按当前 Chunker 和 Tokenizer 在 data/indexes/preliminary-rule-jieba 自动生成完整双路产物，后续复用该目录。真实模型实验需要调用方提供 Embedding 和 Chat 客户端；使用 ZeroEntropy 重排时再配置对应客户端。pytest 使用 mock 响应验证链路，不替代真实模型验收。

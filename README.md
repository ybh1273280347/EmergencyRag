# Emergency-RAG：配置检索流水线与开展实验

本项目把规则数据准备和检索封装为可组合的流水线。实验通常只需配置 YAML、在 settings.py 填写模型参数、在 .env 填写密钥，再调用 `load_pipeline()`，就能得到可直接检索的对象。

项目自动完成规则读取、检索单元生成、BM25 / Dense 索引构建或复用、检索组件装配，以及完整规则证据输出。[experiments/README.md](experiments/README.md) 提供选择题评估 CLI，支持按题完整 JSONL 分片保存、独立检查点与中断恢复、按 question id 重跑，以及答案准确率、规则 Recall、完整规则覆盖率和 nDCG 四项指标。

## 批量实验使用说明

完整的命令、参数默认值、结果影响、恢复与重跑方式见 **[实验 CLI 参数与结果影响说明](experiments/README.md#实验-cli命令参数与结果影响)**。

**重点：实验最终检索 Top-K 为 `max(max(metric_top_k), answer_top_k)`。** 默认最终检索最多 20 条规则，作答使用前 3 条；修改 `--metric-top-k` 也可能改变最终检索与保存的规则数量。例如 `--metric-top-k 1 3 5 --answer-top-k 3` 最终检索最多 5 条，作答仍使用前 3 条。详细预算关系和对照实验目录隔离要求见实验 README。

## 项目目录与核心入口

```text
.
├── config/
│   └── preliminary/         初赛集 Pipeline YAML：baseline.yaml、example.yaml
├── data/
│   ├── raw/                 原始规则语料与验证题目
│   ├── processed/           保存已拆分选项、修复后的验证集
│   ├── indexes/             自动生成并复用的 BM25 / Dense 索引
│   └── cache/               查询处理结果与查询向量缓存
├── examples/                单题检索、Chat 作答、TypeSafe 单选示例
├── experiments/             批量选择题评估 CLI、完整日志与恢复检查点、指标报告
├── scripts/                 一次性离线数据整理、修复与旧实验迁移脚本
├── results/
│   └── preliminary/         按数据集隔离的结果，如 baseline-typesafe-answer-k3/
├── models/                  本地模型权重（如 BCE Reranker）
├── src/emergency_rag/
│   ├── load_pipeline.py      YAML 加载入口，构造组件、准备数据并返回检索流水线
│   ├── settings.py           统一模型参数和密钥配置
│   ├── registry.py           YAML 可用的组件名称及构造工厂
│   ├── clients/              Embedding、Chat、TypeSafe 客户端
│   ├── data/                 规则读取、检索单元构建、索引构建与加载
│   └── retrieval/            查询处理、召回、融合、重排、规则聚合与过滤
├── tests/                    单元与集成测试
├── .env.example              环境变量模板
└── pyproject.toml            项目依赖、构建与 pytest 配置
```

常用入口与职责：

| 文件 | 作用 |
| --- | --- |
| [config/preliminary/baseline.yaml](config/preliminary/baseline.yaml) | 默认实验配置：BM25 + Dense 召回、RRF 融合、本地 BCE 重排。 |
| [src/emergency_rag/load_pipeline.py](src/emergency_rag/load_pipeline.py) | `load_pipeline(path)` 的实现入口，读取 YAML、装配注册组件、准备或复用索引。 |
| [src/emergency_rag/retrieval/pipeline.py](src/emergency_rag/retrieval/pipeline.py) | 编排查询处理、召回、融合、扩展、重排、规则聚合、Gate 和最终 Top-K。 |
| [src/emergency_rag/data/pipeline.py](src/emergency_rag/data/pipeline.py) | 读取规则、生成 SearchUnit、合并 metadata，并构建或加载数据集索引。 |
| [src/emergency_rag/settings.py](src/emergency_rag/settings.py) | 从环境变量 / `.env` 读取共享模型配置；客户端按需创建。 |
| [src/emergency_rag/registry.py](src/emergency_rag/registry.py) | 维护 YAML 组件注册表；新增组件时从对应阶段扩展。 |
| [examples/retrieve_experiment.py](examples/retrieve_experiment.py) | 只检索的单题演示，不调用作答模型。 |
| [examples/chat_experiment.py](examples/chat_experiment.py) | 检索后调用 Chat 作答的单题示例。 |
| [examples/choice_qa_experiment.py](examples/choice_qa_experiment.py) | 检索后调用 TypeSafe / Jev 选择单项答案的示例。 |
| [experiments/__main__.py](experiments/__main__.py) | 批量评估入口：run、resume、rerun、summarize。 |
| [experiments/metrics.py](experiments/metrics.py) | 计算四项核心指标和规则命中分布，生成 JSON / Markdown 总体摘要。 |
| [experiments/negative.py](experiments/negative.py) | 按题汇总答错与调用失败，生成 JSON / Markdown 负样本分析。 |
| [scripts/normalize_dev.py](scripts/normalize_dev.py) | 一次性拆分题干与选项、修复原始验证集，保存整理后的数据。 |
| [tests/](tests/) | 覆盖数据、索引、组件装配、检索流程、客户端与缓存行为。 |

`retrieval/` 内按职责拆分阶段：`retrievers/` 负责 BM25 与 Dense 召回，`fusion/` 提供 RRF 和 Union，`rerank/` 提供 BCE 与 Qwen，`query/` 负责查询处理及缓存；`models.py` 定义查询上下文、Unit 候选、完整规则证据和检索结果。扩展规则单元构建策略见 `data/units/`，扩展检索组件则实现相应阶段的 ABC 并登记到 `registry.py`。

## 1. 第一次运行

需要 Python 3.11+ 和 uv，在项目根目录安装依赖：

```powershell
uv sync --group dev
```

将 [.env.example](.env.example) 复制为项目根目录的 `.env`，填写密钥；再到 [src/emergency_rag/settings.py](src/emergency_rag/settings.py) 的 `Settings` 中填写模型名与服务地址，无需在实验代码中手动加载：

```powershell
Copy-Item .env.example .env
```

`settings.py` 在首次导入时创建唯一的 `settings` 实例，统一读取配置；模型模块不各自加载文件。它从当前工作目录向上查找最近的 `.env`，从项目根目录或其子目录启动即可。配置优先级为：**进程环境变量 → `.env` → Settings 中的默认值**。查找位置不随 YAML 文件位置变化。


| 模型用途                              | `.env` 中的密钥         | `Settings` 中的普通配置                                                                               |
| ------------------------------------- | ----------------------- | ----------------------------------------------------------------------------------------------------- |
| Embedding：离线文档向量、在线查询向量 | `RAG_EMBEDDING_API_KEY` | `embedding_model`、`embedding_base_url`、`embedding_timeout`、`embedding_max_retries`                 |
| Qwen Reranker：Unit 重排              | `QWEN_RERANKER_API_KEY` | `qwen_reranker_model`、`qwen_reranker_base_url`、`qwen_reranker_timeout`、`qwen_reranker_max_retries` |
| Chat：实验作答                        | `RAG_CHAT_API_KEY`      | `chat_model`、`chat_base_url`、`chat_timeout`、`chat_max_retries`                                     |
| TypeSafe / Jev：单选题作答            | `TYPESAFE_API_KEY`      | `typesafe_model`、`typesafe_base_url`、`typesafe_timeout`、`typesafe_max_retries`                     |

Embedding 默认 `qwen3-embedding-4b`，Qwen Reranker 默认 `qwen3-reranker-4b`，TypeSafe 默认 `jev-1.13.0`；Chat 模型名需要填写。未使用的模型可以保持未配置。超时默认 60 秒，最大重试默认 2 次。普通字段仍支持环境变量覆盖，例如 `RAG_EMBEDDING_MODEL`、`RAG_EMBEDDING_BASE_URL`、`RAG_CHAT_MODEL`、`QWEN_RERANKER_MODEL`、`QWEN_RERANKER_BASE_URL`。密钥只从环境读取，不写入源码，也不出现在 Settings 的 repr 中。

云端重排未配置独立密钥与地址时，复用 `RAG_EMBEDDING_API_KEY` 和 `RAG_EMBEDDING_BASE_URL`；不同服务部署时可用 `QWEN_RERANKER_API_KEY`、`QWEN_RERANKER_BASE_URL` 单独覆盖。现有网关使用 `/v1/rerank`，通过 OpenAI SDK 的自定义请求入口调用，超时与重试仍由 SDK 处理。

配置在 `settings` 实例创建时读取；SDK 客户端仍首次使用时才创建并缓存。修改配置后重启实验进程。未使用的模型不要求密钥，实际获取该模型时才检查模型名与密钥是否齐全。

使用 [config/preliminary/baseline.yaml](config/preliminary/baseline.yaml) 加载检索流水线：

```python
from emergency_rag.load_pipeline import load_pipeline

pipeline = load_pipeline("config/preliminary/baseline.yaml")
result = pipeline.retrieve("危险化学品事故应急结束需要满足哪些条件？", top_k=10)

for rule in result.evidence:
    print(rule.final_rank, rule.rule_id, rule.final_score)
    print(rule.text)
```

第一次运行会构建索引，需要访问 Embedding 服务；后续复用已有目录。Baseline 使用本地 BCE 重排，无需重排 API 密钥，不产生云端重排请求；Dense 查询仍会请求 Embedding。只需无打分检索时，将 YAML 的 `reranker` 改为 `default` 或省略它。

**已有索引下的纯 BM25 检索不需要 Embedding 密钥。** 加载入口将 Embedding 客户端工厂延迟传给 DatasetPipeline，缓存命中时不调用它。启用 Dense 检索时需要查询 Embedding；首次构建或 `overwrite: true` 时仍固定构建 BM25、Dense 两路索引，因此需要文档 Embedding 配置。Chat 只在实验作答时创建，不是检索加载的依赖。

## 2. 如何配置 YAML

复制 Baseline 为一个实验配置文件，再调整数据与组件参数。例如：

```yaml
dataset:
  source: ../../data/raw/初赛规则集rules1.json
  dataset_name: preliminary
  index_root: ../../data/indexes
  overwrite: false
  unit_builder: rule
  tokenizer: jieba

retrieval:
  retrievers:
    - name: bm25
      params:
        top_k: 30
    - name: dense
      params:
        top_k: 30
  fusion:
    name: rrf
    params:
      k: 60
      top_k: 60
  reranker:
    name: bce
    params:
      model_path: models/bce-reranker-base_v1
      batch_size: 32
  query_processor: default
  expander: default
  gate: default
```

YAML 顶层只有 `dataset` 和 `retrieval`。无参数的组件直接写注册名称，例如 `fusion: union`；有参数的组件写 `name` 和 `params`。`params` 是对应构造函数或注册工厂接受的关键字参数，不是任意配置字典。模型连接参数由 `settings.py` 管理，密钥放在环境变量中。


| Dataset 字段   | 含义                                                 |
| -------------- | ---------------------------------------------------- |
| `source`       | 必填，规则 JSON 文件路径；不是验证题目文件           |
| `dataset_name` | 必填，英文数据集名称，例如`preliminary`、`semifinal` |
| `index_root`   | 索引根目录；省略时为 YAML 所在目录下的`data/indexes` |
| `overwrite`    | 默认`false`；设为 `true` 时重新读取规则并重建产物    |
| `unit_builder` | 默认`rule`；控制单元划分和索引文本增强               |
| `tokenizer`    | 默认`jieba`；离线 BM25 与在线查询共用它              |

`source` 和 `index_root` 的相对路径都以 **YAML 所在目录** 为起点，不以启动目录为起点。


| Retrieval 阶段            | 内置名称                 | 常用参数 / 默认行为                     |
| ------------------------- | ------------------------ | --------------------------------------- |
| `retrievers`，必填列表    | `bm25`、`dense`          | 各自`top_k=30`，表示召回 Unit 数        |
| `fusion`，必填            | `rrf`、`union`           | RRF：`k=60`、`top_k=60`；Union：无参数  |
| `query_processor`，可省略 | `default`                | 原查询直接用于检索                      |
| `expander`，可省略        | `default`                | 原样返回 Unit 候选                      |
| `reranker`，可省略        | `default`、`bce`、`qwen` | 默认不打分；BCE 本地推理，Qwen 云端评分 |
| `gate`，可省略            | `default`                | 放行所有聚合后的规则证据                |

可选阶段省略时使用默认组件；不要写 `null`。最终返回数量由 `pipeline.retrieve(query, top_k=10)` 控制，不写入 YAML 的 Pipeline 构造参数。

BCE 参数为 `model_path`、`device`、`use_fp16`、`batch_size=32`。本地权重默认位于 `models/bce-reranker-base_v1`，相对模型路径以进程工作目录为起点，从项目根目录启动即可。首次选择 BCE 时才加载本地模型并缓存，后续题目复用；不自动下载权重。`device` 省略时自动选择 CUDA 或 CPU，CPU 使用 FP32，CUDA 默认使用 FP16。BCE 不接受 instruction；重排调用 SDK 的 `rerank()` 执行长文本滑窗评分，保留全部候选和完整精度分数，不在重排阶段过滤或截取 Top-K。SDK 的 `compute_score()` 会截断长文本，不能替代此路径。

注册名 `qwen` 用于云端 Qwen 对照，接受 `instruction`、`batch_size=64`。Qwen 的 `instruction` 会以 `<Instruct>: ...\n<Query>: ...` 拼入发送给重排服务的查询，文档仍使用 Unit 原始 `text`。这是查询上下文增强：当前网关的独立 `instruction` / `instruct` 字段未表现出被使用，不能将此方式视为模型原生自定义 prompt。Qwen 模型原生指令格式见[官方模型卡](https://huggingface.co/Qwen/Qwen3-Reranker-4B)。修改重排指令不改变召回查询，也不需要重建索引。

三个预算分别生效：Retriever 的 `top_k` 限制原始召回 Unit 数；RRF 的 `top_k` 限制融合后 Unit 数；查询入口的 `top_k` 限制最终不同规则数。Union 交替合并各路已有结果并按 Unit 去重，没有额外预算。两路各召回 30 个 Unit，重叠 12 个时，Union 输出 48 个。重排 `batch_size` 只拆分请求，不截断候选。

### 查询缓存是固定行为

通过 `load_pipeline()` 加载时，QueryProcessor 的处理结果和 Dense 查询向量都会自动持久化，不需要在 YAML 中增加缓存字段。默认目录由 `Settings.query_cache_root` 决定，初始值为相对进程工作目录的 `data/cache`：

```text
data/cache/
  query_processing/default/queries.json
  query_embeddings/qwen3-embedding-4b/embeddings.json
```

查询处理以原始问题作键，保存完整 `QueryContext`，包括改写问题和子查询；查询向量以实际参与检索的文本作键，记录完整模型名和实际维度。相同子查询即使来自不同题目，也可以复用向量。改变 BM25、Fusion、Reranker、Gate 或最终 Top-K 不需要重算这些内容。离线文档 Embedding 继续使用 `embed_batch()`，不混用查询缓存。

文件首次使用时读取到内存，后续查找不反复读盘；成功结果立即保存，重新启动实验也能复用。Dense 命中缓存时不发起 Embedding 请求，但加载 Dense 仍需要有效的模型客户端配置。失败请求、非法向量或写入失败不会替换旧缓存；文件完整写入后原子替换。缓存目录已加入 Git 忽略，没有 hash、语料指纹或数据库。

**查询处理的缓存目录按组件 `name` 隔离。** 自定义改写组件应让 `name` 表达模型和策略版本，例如 `rewrite-qwen3-v2`；更换 Rewrite 模型、提示词或拆分规则时换用新名称。加载器通过接口名称判断缓存身份，不窥探具体组件内部字段。Embedding 模型名变化自动使用不同子目录；相同模型名下权重或查询向量生成方式改变时，修改 `settings.query_cache_root` 为新版本目录。新策略生成的查询文本与旧文本相同且共用查询向量目录时，仍能复用向量。

需要显式更新某个问题时，直接刷新对应缓存：

```python
pipeline = load_pipeline("config/preliminary/baseline.yaml")
pipeline.query_processor.process("原始问题", refresh=True)

from emergency_rag.clients.embedding import get_embedding_client
get_embedding_client().embed_query("实际检索文本", refresh=True)
```

平时继续调用 `pipeline.retrieve(query)` 即可。缓存刷新成功后后续查询复用新结果；刷新失败保留旧结果。当前文件缓存适用于单写入进程和后续跨进程复用；并行写入的实验应使用不同的 `query_cache_root`。缓存损坏明确报错，不静默调用付费模型重算。

## 3. 加载器会自动完成什么

`load_pipeline(path)` 返回一个 `RetrievalPipeline`，可以复用于整个验证集：

1. 读取 YAML，按阶段注册表构造组件，检查名称、参数和 ABC 类型。
2. 注入模型依赖：DatasetPipeline 接收延迟创建 Embedding 的工厂，仅实际构建时调用；启用 Dense 时使用同一共享客户端。BCE 或 Qwen 从各自模块的 `get_rerank_model()` 获取缓存的重排器，未选择 BCE 时不加载 Torch 或本地权重。
3. 运行 DatasetPipeline，准备或加载完整 Rule、SearchUnit 和索引。
4. 将同一个 Dataset 注入 RetrievalPipeline，返回可调用的检索对象。

Dataset 目录固定为：

```text
<index_root>/<dataset_name>-<unit_builder.name>-<tokenizer.name>-<embedding_model>/
  rules.json              完整规则快照及 metadata
  units.json              检索单元的 text、index_text 及 metadata
  index_meta.json         构建说明
  bm25/                   BM25 索引、词表和 Unit 行映射
  dense/                  FAISS 索引、Unit 行映射和构建说明
```

默认目录名为 `preliminary-rule-jieba-qwen3-embedding-4b`。切换 Embedding 模型后自动选择不同目录；模型名中的斜杠、空格等目录不适用字符转为 `-`，构建说明中的 `embedding_model` 仍保存完整模型名。目录存在且 `overwrite: false` 时，直接加载产物，不读取源规则、不调用 UnitBuilder、不请求文档 Embedding，也不创建 Embedding 客户端。不存在时，读取规则、生成 Unit、合并 metadata，并构建两路索引。重建全部成功才替换目标目录；失败保留旧产物。已有产物读取失败会报错，不自动重建。

直接装配 DatasetPipeline 时，传入 EmbeddingClient 会自动使用它的 `model`；传入延迟客户端工厂时，同时传入 `embedding_model`。`load_pipeline()` 自动从 Settings 注入模型名，无需在 YAML 重复配置。实际构建继续在 `index_meta.json` 记录模型名和向量维度。

**目录是否存在决定缓存是否复用，不会自动识别配置变化。** 更换源数据、Embedding 模型、单元增强参数或分词参数时，使用新的数据集 / 策略名称，或设置 `overwrite: true` 显式重建。完成重建后将其恢复为 `false`。离线文档向量与在线查询必须使用相同 Embedding 模型；分词器必须保持相同策略。

配置错误抛出 `PipelineConfigError`；文件、索引构建和模型调用错误向调用方传播。实验代码负责记录失败，不要把失败当作空答案或静默切换算法。

## 4. 检索结果与数据约定

```text
规则文件 → UnitBuilder → index_text 构建双路索引 → IndexedDataset

查询 → QueryProcessor → Retrievers → Fusion → Expansion → Reranker
     → 按 rule_id 聚合完整规则 → EvidenceGate → Top-K 条规则
```

原始规则文件是非空 JSON 数组，`rule_id` 是唯一非空字符串，`rule_text` 是非空原文，`metadata` 可省略：

```json
[
  {
    "rule_id": "37",
    "rule_text": "这里是完整规则原文。",
    "metadata": {"domain": "危险化学品", "topic": "应急结束"}
  }
]
```

初赛规则文件为 `data/raw/初赛规则集rules1.json`，含 800 条规则；`data/raw/初赛验证集dev.json` 是实验题目与答案，不作为索引语料。


| 对象              | 使用约定                                                                             |
| ----------------- | ------------------------------------------------------------------------------------ |
| `Rule`            | `rule_id`、完整原文 `text`、业务 `metadata`                                          |
| `SearchUnit`      | 唯一`unit_id`、关联 `rule_id`、单元原文 `text`、索引文本 `index_text`、`metadata`    |
| `IndexedDataset`  | 完整`rules`、`units`、已加载索引及同一个 tokenizer                                   |
| `Candidate`       | Unit 层命中；保存单元原文、来源、召回记录及重排分数                                  |
| `RuleEvidence`    | 最终完整规则；保存`final_score`、`final_rank`、来源、`matched_units` 及规则 metadata |
| `RetrievalResult` | 原始`query`、最终 `evidence`、耗时与候选数量等 `metadata`                            |

`index_text` 可以加入 Domain、Topic 或上下文；它只用于 BM25 索引和文档 Embedding。`text` 保留单元原文，供重排和命中分析。最终证据文本始终取自 `dataset.rules[rule_id].text`，不是增强文本，也不是子规则片段。

完整 Rule 与多个 SubRule 可以共存。默认 Unit ID 为 `rule:<rule_id>`；自定义 ID 格式不影响检索，关联规则只使用显式 `rule_id`。DatasetPipeline 合并 metadata 的顺序是：Rule metadata → Unit metadata 覆盖同名字段 → 写入 `dataset` 和 `unit_builder` 构建信息。

重排后按规则聚合，分数取该规则本次命中 Unit 的最高重排分数，不累加；同分按 `rule_id` 排序。默认无打分模式下 `final_score=None`，保留规则首次出现顺序。Gate 收到完整规则证据，最终 Top-K 按不同规则计数，排名从 1 开始。`matched_units` 保留全部命中单元及召回记录。

## 5. 如何实现和使用新组件

继承对应阶段的 ABC，提供稳定的英文 `name`，实现下表接口。简单参数直接写构造函数；只有确实复杂的初始化才需要组件自己的 Config 类。


| 扩展点与接口模块                                          | 方法                                                                 | 需要保持的语义                                                |
|---------------------------------------------------| -------------------------------------------------------------------- | ------------------------------------------------------------- |
| `data/units/base.py`：`UnitBuilder`                | `build(rules: list[Rule]) -> list[SearchUnit]`                       | 每个 Unit 关联已有 Rule；可切分子规则或增强索引文本           |
| `retrieval/tokenizer/base.py`：`TextTokenizer`     | `tokenize(text: str) -> list[str]`                                   | 索引与查询共用同一策略                                        |
| `retrieval/query/base.py`：`QueryProcessor`        | `process(query: str) -> QueryContext`                                | 保留`original_query`；可设置 `rewritten_query`、`sub_queries` |
| `retrieval/retrievers/base.py`：`Retriever`        | `retrieve(query: str, dataset: IndexedDataset) -> list[Candidate]`   | 返回有序 Unit 候选，预算在初始化时确定                        |
| `retrieval/fusion/base.py`：`Fusion`               | `fuse(result_sets: list[list[Candidate]]) -> list[Candidate]`        | 按 Unit 合并，保留来源和召回记录                              |
| `retrieval/expansion/base.py`：`CandidateExpander` | `expand(candidates: list[Candidate]) -> list[Candidate]`             | 输入输出仍是 Unit 候选                                        |
| `retrieval/rerank/base.py`：`Reranker`             | `rerank(query: str, candidates: list[Candidate]) -> list[Candidate]` | 评分模式完整打分排序；默认模式原样返回                        |
| `retrieval/gate/base.py`：`EvidenceGate`           | `filter(evidence: list[RuleEvidence]) -> list[RuleEvidence]`         | 在规则聚合后筛选，不处理 Unit 候选                            |

查询处理返回的 `QueryContext.queries` 为改写后的主查询（未改写时为原查询）加子查询。Pipeline 对每个查询调用所有 Retriever；重排与最终结果仍使用 `original_query`。

例如，将规则主题加入索引文本，同时保留原文：

```python

from emergency_rag.data.units.base import Rule, SearchUnit
from emergency_rag.registry import COMPONENT_REGISTRY
from emergency_rag.data.units import UnitBuilder
from emergency_rag.load_pipeline import load_pipeline


class TopicUnitBuilder(UnitBuilder):
    name = "rule-topic"

    def build(self, rules: list[Rule]) -> list[SearchUnit]:
        return [
            SearchUnit(
                unit_id=f"rule:{rule.rule_id}",
                rule_id=rule.rule_id,
                text=rule.text,
                index_text=f"{rule.metadata.get('topic', '')}\n{rule.text}",
            )
            for rule in rules
        ]


# 在加载配置之前登记一次，YAML 才能识别该名称。
COMPONENT_REGISTRY["unit_builder"][TopicUnitBuilder.name] = TopicUnitBuilder
pipeline = load_pipeline("config/topic_experiment.yaml")
```

在复制的 YAML 中将 `dataset.unit_builder` 改为 `rule-topic`，其余配置可以保持不变。索引将写入 `preliminary-rule-topic-jieba`。不同切分、增强或分词策略应使用不同名称，避免误复用缓存。

注册表位于 [src/emergency_rag/registry.py](src/emergency_rag/registry.py)，按阶段保存名称到构造函数的映射。新 Fusion 例如登记到 `COMPONENT_REGISTRY["fusion"][MyFusion.name]`，新 Gate 登记到 `COMPONENT_REGISTRY["gate"][MyGate.name]`。名称只需在所属阶段内唯一。

组件需要模型等依赖时，登记一个显式注入依赖的工厂；工厂的参数对应 YAML 的 `params`：

```python
def build_my_reranker(*, instruction: str, batch_size: int = 64):
    return MyReranker(
        client=get_my_model_client(),
        instruction=instruction,
        batch_size=batch_size,
    )


COMPONENT_REGISTRY["reranker"][MyReranker.name] = build_my_reranker
```

`MyReranker` 和 `get_my_model_client` 由实验实现提供。无需给加载器增加算法分支。新组件也可以直接用 Python 装配，不必注册：

```python
from emergency_rag.retrieval.pipeline import RetrievalPipeline
from emergency_rag.retrieval.retrievers.bm25 import BM25Retriever
from emergency_rag.retrieval.fusion.union import UnionFusion

pipeline = RetrievalPipeline(
    dataset=dataset,
    retrievers=(BM25Retriever(top_k=30),),
    fusion=UnionFusion(),
    gate=MyGate(),
)
```

这里的 `dataset` 是已准备的 IndexedDataset，`MyGate` 是继承 EvidenceGate 的实现。需要自行准备 Dataset 时使用 `DatasetPipeline(...).prepare(source, dataset_name=...)`，接口见 [src/emergency_rag/data/pipeline.py](src/emergency_rag/data/pipeline.py)。

## 6. 运行检索示例与完整实验

仓库提供三个单题入口，均从项目根目录运行。它们使用仓库规则集和对应的默认 YAML 配置；第一次运行会建立 BM25 与 Dense 两路索引，因此需要 Embedding 服务。三个示例入口当前都启用本地 BCE 重排，还需要 `models/bce-reranker-base_v1` 权重；BCE 不会自动下载模型。输出为单题演示，不会读取验证集或计算完整数据集准确率。

只运行检索可使用 [examples/retrieve_experiment.py](examples/retrieve_experiment.py)：它加载 [config/preliminary/example.yaml](config/preliminary/example.yaml)，用 BM25 召回、RRF 融合和 BCE 重排，并连续运行同一问题两次展示冷启动与预热查询。需要 Embedding 配置和本地 BCE 权重，不需要 Chat 或 TypeSafe 密钥：

```powershell
uv run python examples/retrieve_experiment.py
```

检索并调用 Chat 作答可运行 [examples/chat_experiment.py](examples/chat_experiment.py)。它加载 [config/preliminary/baseline.yaml](config/preliminary/baseline.yaml)，需要本地 BCE 权重、Embedding 配置，以及 `RAG_CHAT_API_KEY` 和 `RAG_CHAT_MODEL`：

```powershell
uv run python examples/chat_experiment.py
```

```python
from emergency_rag.clients.chat import get_chat_client
from emergency_rag.load_pipeline import load_pipeline
from examples.chat_experiment import run_experiment

pipeline = load_pipeline("config/preliminary/baseline.yaml")
chat = get_chat_client()

result, answer = run_experiment(
    pipeline=pipeline,
    chat=chat,
    question="危险化学品事故应急结束需要满足哪些条件？",
    instructions="请根据参考规则回答问题，并注明规则编号。",
    top_k=10,
)
```

`pipeline` 和 `chat` 在题目循环之外创建一次。问答题通过指令要求文字说明；选择题将选项放入 `question`，通过指令约定选项输出格式。


| 实验代码需要补齐   | 建议输出 / 行为                                                         |
| ------------------ | ----------------------------------------------------------------------- |
| 读取验证集         | 题目 ID、题干、选项、标准答案；按验证文件真实格式适配                   |
| 业务提示词         | 明确任务、证据使用方式和回答格式                                        |
| Response 解析      | 将原始文本转为可比较答案；明确非法输出如何处理                          |
| 准确率计算         | 按题型定义答案归一化和判定方式，记录分子、分母                          |
| 批量运行与异常记录 | 复用 Pipeline / Chat，记录请求失败和解析失败，明确是否计入分母          |
| 实验结果保存       | 配置、题目 ID、检索规则及分数、原始回答、解析答案、标准答案、判定与耗时 |

`ChatClient` 初始化时确定模型名，`complete(messages=..., max_tokens=..., response_format=...)` 返回非空文本。`get_chat_client()` 使用共享 `settings`；直接注入 SDK 时使用 `ChatClient(sdk, model="模型名")`。提示词、JSON 解析、请求节流和答案判定都属于实验代码。需要结构化输出时由调用方显式传 `response_format`，客户端不会默认要求 JSON。

### 使用 TypeSafe / Jev 回答单选题

[ChoiceQAClient](src/emergency_rag/clients/choice_qa_client.py) 使用官方同步 `typesafe-sdk`，通过 `Choice` 请求返回类型化 `ChoiceAnswer`，包含选项标签、置信度和各选项概率，接口依据 [TypeSafe 官方文档](https://docs.typesafe.ai/introduction/quickstart)。它是作答客户端，由实验调用，不属于检索组件注册表。

在 `.env` 填写 `TYPESAFE_API_KEY`，模型和地址可以直接修改 Settings 的默认值，也可通过以下环境变量覆盖：

```dotenv
TYPESAFE_MODEL=jev-1.13.0
TYPESAFE_BASE_URL=your-bse-url
```

`typesafe_base_url` / `TYPESAFE_BASE_URL` 填 API 根地址。SDK 自行追加 `/v1/systemone`，此网关不要填写末尾的 `/v1`。超时与重试可用 `TYPESAFE_TIMEOUT`、`TYPESAFE_MAX_RETRIES` 覆盖，重试由 SDK 负责。

```python
from emergency_rag.clients.choice_qa_client import get_choice_qa_client
from emergency_rag.load_pipeline import load_pipeline

pipeline = load_pipeline("config/preliminary/baseline.yaml")
choice_client = get_choice_qa_client()
result = pipeline.retrieve("事故现场应急处置工作结束需要谁确认和批准？", top_k=10)

answer = choice_client.choose(
    state={
        "question": "事故现场应急处置工作结束需要谁确认和批准？",
        "evidence": [
            {"rule_id": rule.rule_id, "text": rule.text}
            for rule in result.evidence
        ],
    },
    instructions="仅根据参考规则选择唯一正确选项。",
    choices={"A": "现场应急救援指挥部", "B": "任意救援队员"},
)
print(answer.choice, answer.confidence, answer.probabilities)
```

`state` 可以是文本或 JSON 对象 / 数组；题干、完整规则证据和指令由实验提供。`choices` 的键是答案标签，值是选项内容。该接口每次选择一个标签，不直接处理多选答案集合。客户端校验响应类型、选项对应关系和概率范围；请求失败原样抛出 SDK 异常。准确率判定由实验将 `answer.choice` 与标准答案比较，置信度不是准确率。

[examples/choice_qa_experiment.py](examples/choice_qa_experiment.py) 提供完整单题示例，加载同一 Baseline 检索配置。准备本地 BCE 权重，配置 Embedding 和 TypeSafe 密钥后，从项目根目录运行：

```powershell
uv run python examples/choice_qa_experiment.py
```

输出包含检索规则 ID、选项、置信度、概率分布和本题判定。批量实验可导入 `run_choice_experiment()`，传入已加载的 Pipeline、ChoiceQAClient、题干、选项和指令；返回 `(RetrievalResult, ChoiceAnswer)`。Pipeline 与客户端在题目循环外创建一次，逐题保存结果并累计准确率。也可传入使用默认无打分重排器的自定义 Pipeline，无需 Qwen Reranker 配置。

## 7. AI 辅助开发必须遵守的边界

给 AI 分配新策略或实验时，先说明要扩展哪个接口、输入输出是什么、如何验证。请将以下约束一并提供给 AI：

- **保持组件可直接构造。** 组件不读取 YAML，也不依赖加载器或顶层实验配置；外部模型通过构造参数注入。YAML 的解析与装配留在 `load_pipeline` 和注册表。
- **统一模型配置入口。** 新增服务的连接参数与环境读取放在 `settings.py`，模型获取函数读取共享 `settings`。不要在组件中复制 `.env` 查找、读取环境变量或实例化另一份 Settings；密钥不得写入源码。
- **保持单 Dataset 检索。** Dataset 在 Pipeline 注入一次，Retriever 接口保持 `retrieve(query, dataset)`。不要让每个 Retriever 重新读取源文件、构建索引或管理数据目录。
- **保持文本用途。** 索引使用 `index_text`，重排使用 Unit `text`，作答使用完整 Rule `text`。不得用增强文本替换规则原文。
- **保持 Unit 与 Rule 的关联。** 每个 Unit ID 唯一，`rule_id` 指向已保存的完整 Rule；不要通过 ID 前缀推断关联。Fusion 不提前按 Rule 去重。
- **保持分数语义。** `Candidate.final_score` 只表示重排分数，评分重排器为全部输入候选打分，分数必须是 `[0, 1]` 内有限数值。召回记录写入 `metadata["retrieval"]`，融合信息写入 `metadata["fusion"]`，来源写入 `sources`；排名从 1 开始。
- **保持预算独立。** Retriever 预算只在初始化配置；Union 不额外截断；重排批大小不充当候选上限；最终 Top-K 在规则聚合、Gate 之后执行。
- **保持缓存规则明确。** 策略变更使用新名称或显式重建，不添加自动重建、hash / 指纹审计或模型地址比对。不要在每层重复校验已由离线构建保证的数据事实。
- **保持职责集中。** Pipeline 只编排阶段和聚合规则；作答提示词、Response 解析、评测和实验文件输出留在用例中。不要增加插件自动发现、通用 Manager / Service、空 Config 或额外 Projector 阶段。
- **保持失败可见。** 不静默降级，不把 API、索引或解析失败伪装成正常结果；超时和重试由模型客户端边界配置。
- **验证真实接口。** 新策略用对应 ABC 和共享模型，补充行为测试；远程调用使用 mock，BM25 / FAISS 持久化使用实际实现。不得将 mock 测试通过表述为真实模型或答案准确率验收通过。

运行测试：

```powershell
uv run pytest -q
```

测试默认禁止真实网络。涉及新索引策略时，至少验证保存加载、增强文本与原文分离、Rule / SubRule 聚合和缓存重建；涉及新检索或重排策略时，验证预算、来源记录、排序和候选集合。真实模型链路与完整实验准确率需要配置服务后另行运行，并保存实际结果。

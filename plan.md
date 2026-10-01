你现在位于一个 Emergency-RAG 项目仓库中。请从零开始实现第一阶段的核心检索框架和 Baseline。

仓库中已经放入初赛规则数据集和初赛验证集。当前阶段使用初赛规则数据集构建知识库；验证集后续用于统一评测体系，本轮重点是建立可扩展的 Retrieval Core 和可运行的 Baseline。

请先检查仓库中的数据文件格式和现有目录，再根据下面的设计直接完成代码实现。不要停留在设计文档阶段，完成可运行代码、配置、脚本和必要测试。

# 1. 本轮目标

第一阶段实现一套组件化、可插拔的 Retrieval Pipeline。

对外核心接口统一为：

```
result = pipeline.retrieve(query, top_k)
```

本轮 Baseline：

```
Rule Dataset
    ↓
Rule-level Search Units
    │
    ├── BM25s Sparse Retrieval
    │
    └── Embedding + FAISS Dense Retrieval
             ↓
           Fusion
             ↓
          Reranker
             ↓
       RetrievalResult
```

Baseline 的索引文本直接使用原始 `rule_text`。

当前数据中的 Rule 本身就是具有明确语义边界的知识单元，因此：

```
1 Rule = 1 SearchUnit
```

同时保留独立的 `Chunker` 抽象，为后续父子分块、多粒度索引等实验提供扩展点。

---

# 2. 技术栈

建议使用：

- Python 3.11+

- `openai` 官方 Python SDK：统一访问 OpenAI-compatible Embedding / Chat API

- `bm25s`：Sparse Retrieval

- `faiss-cpu`：Dense Vector Index

- `numpy`

- `pydantic`：配置与结构化数据模型

- `pydantic-settings`：环境变量和配置管理

- `python-dotenv`

- `pytest`

- 标准库 `sqlite3`：处理后的结构化数据存储

- JSON / JSONL：必要的中间产物和调试输出

项目使用 `src` layout。

建议包名：

```
src/emergency_rag/
```

数据处理、索引构建等一次性脚本统一放：

```
scripts/
```

这些代码属于离线数据工程，不放入 `src/emergency_rag`。

---

# 3. Pipeline 抽象

整体 Pipeline 按以下概念设计：

```
Query
  ↓
QueryProcessor
  ↓
Retrievers[]
  ↓
Fusion
  ↓
Expansion
  ↓
Reranker
  ↓
Gate
  ↓
Projector
  ↓
RetrievalResult
```

本轮实现时，核心 Baseline 实际使用：

```
Query
  ↓
BM25Retriever + DenseRetriever
  ↓
Fusion
  ↓
Reranker
  ↓
Identity Projector
  ↓
RetrievalResult
```

架构上保留统一组件接口，使未来能够自然增加 Expansion、Gate、Hierarchy、Subquery、父子 SearchUnit 等机制。

Pipeline 本身只负责编排，不负责实现具体检索算法。

不要通过模式枚举组织多条平行 Pipeline。所有实验能力都应该通过组件替换实现。

外部始终只调用：

```
pipeline.retrieve(query, top_k)
```

---

# 4. QueryProcessor

定义统一接口，例如：

```
class QueryProcessor(Protocol):
    def process(self, query: str) -> QueryContext:
        ...
```

本轮实现：

```
IdentityQueryProcessor
```

其行为只是保留原始 Query。

建议：

```
class QueryContext(BaseModel):
    original_query: str
    queries: list[str]
```

Baseline：

```
QueryContext(
    original_query=query,
    queries=[query],
)
```

这个接口未来将用于 Query Rewrite 和 Subquery。

---

# 5. Chunker

定义通用：

```
class Chunker(Protocol):
    def chunk(self, records: list[Rule]) -> list[SearchUnit]:
        ...
```

本轮实现：

```
RuleChunker
```

行为：

```
1 Rule
→
1 SearchUnit
```

不对 `rule_text` 做固定长度切分。

例如：

```
SearchUnit(
    unit_id="rule:37",
    rule_id="37",
    text="<原始 rule_text>",
)
```

Chunker 只负责：

```
Knowledge Record
→ SearchUnit
```

后续可以增加 SubRuleChunker，而无需修改 Retrieval Pipeline。

---

# 6. 核心数据模型

请集中定义稳定的数据模型，避免各 Retriever 自己返回任意 dict。

至少包括：

## Rule

```
class Rule(BaseModel):
    rule_id: str
    text: str
```

## SearchUnit

```
class SearchUnit(BaseModel):
    unit_id: str
    rule_id: str
    text: str
    metadata: dict = {}
```

## QueryContext

```
class QueryContext(BaseModel):
    original_query: str
    queries: list[str]
```

## Candidate

统一承载各阶段产生的检索信息，例如：

```
class Candidate(BaseModel):
    unit_id: str
    rule_id: str
    text: str
	final_rank: int | None = None
    final_score: float | None = None

    sources: list[str] = []
    metadata: dict = {} // 此处可存放各retrieve阶段的独立得分或排名和fusion得分或排名，但顶层score只展示最终rerank得分
```

这里的 rank 统一从 1 开始。

## RetrievalResult

```
class RetrievalResult(BaseModel):
    query: str
    candidates: list[Candidate]
    metadata: dict = {}
```

metadata 至少可以记录：

```
retrieval latency
candidate count
active retrievers
fusion strategy
reranker
```

---

# 7. Retriever 抽象

所有召回算法实现统一接口：

```
class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> list[Candidate]:
        ...
```

本轮实现：

```
BM25Retriever
DenseRetriever
```

Pipeline 接收：

```
retrievers: list[Retriever]
```

因此后续新的检索策略可以直接作为新的 Retriever 插入。

---

# 8. BM25Retriever

Sparse Retrieval 统一使用：

```
bm25s
```

数据：

```
SearchUnit.text
```

Baseline 即原始：

```
rule_text
```

实现：

```
SearchUnits
→ tokenize
→ BM25s index
→ query
→ Top-K Candidate
```

中文分词逻辑封装成独立 tokenizer，不与 Retriever 主逻辑耦合。

可以优先采用轻量且可靠的中文 tokenizer，例如 thulac/jieba，接口保持可替换。

BM25Retriever 输出：

```
Candidate(
    unit_id=...,
    rule_id=...,
    text=...,
    sparse_score=...,
    sparse_rank=...,
    sources=["bm25"],
)
```

BM25 索引支持持久化和重新加载。

---

# 9. EmbeddingClient

统一使用 OpenAI 官方 Python SDK，通过 OpenAI-compatible endpoint 调用 Embedding API。

定义独立模型客户端：

```
class EmbeddingClient:
    def embed(self, text: str) -> np.ndarray:
        ...
```

配置至少包括：

```
base_url
api_key
model
batch_size
timeout
```

从环境变量或配置读取，不把模型和地址硬编码到业务代码。

注意：

- 支持 batch embedding；

- 返回统一 `float32 numpy.ndarray`；

- 做必要的输入检查；

- Dense Index 使用 cosine similarity，因此 embedding 在写入 FAISS 和查询前统一 L2 Normalize。

---

# 10. DenseRetriever

统一使用：

```
FAISS IndexFlatIP
```

流程：

```
SearchUnit.text
↓
EmbeddingClient
↓
L2 Normalize
↓
FAISS IndexFlatIP
```

查询：

```
Query
↓
EmbeddingClient
↓
L2 Normalize
↓
FAISS search
```

DenseRetriever 输出：

```
Candidate(
    unit_id=...,
    rule_id=...,
    text=...,
    dense_score=...,
    dense_rank=...,
    sources=["dense"],
)
```

需要维护：

```
FAISS row index
↔
SearchUnit.unit_id
```

的稳定映射。

FAISS Index 和映射表需要支持持久化及重新加载。

---

# 11. Fusion 抽象

定义：

```
class Fusion(Protocol):
    def fuse(
        self,
        result_sets: list[list[Candidate]],
        top_k: int | None = None,
    ) -> list[Candidate]:
        ...
```

本轮至少实现两种：

```
UnionFusion
RRFFusion
```

## UnionFusion

执行：

```
merge
+
deduplicate by unit_id
```

保留同一个 Candidate 来自哪些 Retriever，以及各路原始 score/rank。

Union 的主要用途是允许后面的 Reranker 直接统一判断相关性。

## RRFFusion

实现 Reciprocal Rank Fusion：

```
RRF(d) = Σ 1 / (k + rank_i(d))
```

默认：

```
k = 60
```

但必须配置化。

不要假定 RRF 永远优于 Union，因此二者使用同一 Fusion 接口。

---

# 12. ChatLLM 抽象

统一使用 OpenAI 官方 Python SDK，通过 OpenAI-compatible endpoint 调用 Chat Completions。

实现独立：

```
class ChatLLM:
    ...
```

配置：

```
base_url
api_key
model
temperature
timeout
max_retries
```

至少支持：

```
invoke(messages) -> str
```
要求输出格式为 Json
请让模型调用、重试等逻辑集中在 ChatLLM 层。

后续 Query Rewrite、Semantic Relation、LLM Reranker 等组件都通过这个抽象使用模型。

---

# 13. Reranker 抽象与 Baseline

定义：

```
class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        candidates: list[Candidate],
    ) -> list[Candidate]:
        ...
```

Baseline 需要提供一个实际可运行、能够输出相关性分数的 Reranker。

优先参考实现：

```
D:\WisePenCloud-AI\wisepen_common\WisePenCloud-AI\services\wisepen-common\src\common\utils\ranking\rerankers\zero_entropy_reranker.py
```


输入：

```
Query
+
若干 Candidate Rule
```
```

要求：

- relevance score 统一到 `[0, 1]`；

- Candidate 数量通过配置控制；

- 支持 batch rerank；

- 严格按照 `unit_id` 对齐结果；

- 模型返回异常时有清晰错误处理；

- rerank 后写入 `candidate.final_score`；

- 最终按 rerank score 降序。

Reranker 不负责过滤候选，只负责：

```
Candidate → relevance score → reordered Candidate

# 14. Expansion / Gate / Projector 接口

为了让 Pipeline 结构稳定，本轮可以提供最轻量的接口和 identity/no-op 实现。

## CandidateExpander

```
class CandidateExpander(Protocol):
    def expand(
        self,
        query_ctx: QueryContext,
        candidates: list[Candidate],
    ) -> list[Candidate]:
        ...
```

本轮：

```
NoOpExpander
```

原样返回。

## Gate

```
class CandidateGate(Protocol):
    def filter(
        self,
        query_ctx: QueryContext,
        candidates: list[Candidate],
    ) -> list[Candidate]:
        ...
```

本轮可以使用最简单的：

```
PassThroughGate
```

保持候选不变。

## Projector

```
class EvidenceProjector(Protocol):
    def project(
        self,
        candidates: list[Candidate],
    ) -> list[Candidate]:
        ...
```

本轮由于：

```
1 Rule = 1 SearchUnit
```

实现：

```
IdentityProjector
```

原样映射。

这些接口保持极薄即可，核心目的是稳定 Pipeline 生命周期。

---

# 15. RetrievalPipeline

Pipeline 只做编排。

建议核心逻辑接近：

```
class RetrievalPipeline:
    def __init__(
        self,
        query_processor: QueryProcessor,
        retrievers: list[Retriever],
        fusion: FusionStrategy,
        expander: CandidateExpander,
        reranker: Reranker,
        gate: CandidateGate,
        projector: EvidenceProjector,
    ):
        ...

    def retrieve(self, query: str, top_k: int) -> RetrievalResult:
        query_ctx = self.query_processor.process(query)

        result_sets = []

        for q in query_ctx.queries:
            for retriever in self.retrievers:
                result_sets.append(
                    retriever.search(
                        q,
                        top_k=self.retrieve_top_k,
                    )
                )

        candidates = self.fusion.fuse(
            result_sets,
            top_k=self.rerank_top_k,
        )

        candidates = self.expander.expand(
            query_ctx,
            candidates,
        )

        candidates = self.reranker.rerank(
            query_ctx.original_query,
            candidates,
            top_k=self.final_top_k,
        )

        candidates = self.gate.filter(
            query_ctx,
            candidates,
        )

        candidates = self.projector.project(
            candidates,
        )

        return RetrievalResult(
            query=query,
            candidates=candidates,
            metadata=...
        )
```

根据实际实现可以适当调整命名，但保持生命周期：

```
query_processor
→ retrievers
→ fusion
→ expansion
→ reranker
→ gate
→ projector
```

---

# 16. 数据处理与存储

原始数据处理属于离线任务，统一放：

```
scripts/
```

建议至少实现：

```
scripts/
├── prepare_rules.py
└── build_indexes.py
```

## prepare_rules.py

读取仓库中的初赛规则数据集。

首先实际检查数据格式，再转换成统一的：

```
Rule
↓
RuleChunker
↓
SearchUnit
```

建议使用 SQLite 作为处理后知识库的 canonical store。

例如：

```
data/processed/emergency.db
```

初始表：

```
CREATE TABLE rules (
    rule_id TEXT PRIMARY KEY,
    rule_text TEXT NOT NULL
);

CREATE TABLE search_units (
    unit_id TEXT PRIMARY KEY,
    rule_id TEXT NOT NULL,
    text TEXT NOT NULL,
    metadata_json TEXT,
    FOREIGN KEY(rule_id) REFERENCES rules(rule_id)
);
```

当前阶段一条 Rule 对应一条 SearchUnit。

数据处理要求：

- rule_id 保持稳定；

- 检查重复 rule_id；

- 检查空文本；

- 输出处理统计；

- 使用事务；

- 支持重复执行和覆盖重建。

## build_indexes.py

从 processed store 中读取 SearchUnit，构建：

```
BM25s index
FAISS index
FAISS row mapping
```

索引产物建议统一：

```
data/indexes/
├── bm25/
├── faiss.index
├── faiss_mapping.json
└── index_meta.json
```

`index_meta.json` 至少记录：

```
embedding model
embedding dimension
document count
build time
index version
```

---

# 17. 配置

统一配置模型，例如：

```
config/
└── baseline.yaml
```

或者 Pydantic Settings + `.env`。

配置至少包含：

```
embedding:
  base_url: ...
  model: ...
  batch_size: 64

chat:
  base_url: ...
  model: ...
  temperature: 0

retrieval:
  sparse_top_k: 30
  dense_top_k: 30
  fusion: rrf
  rrf_k: 60
  rerank_top_k: 30
  final_top_k: 10
```

API Key 通过环境变量读取。

提供：

```
.env.example
```

不要提交真实 Key。

---

# 18. 推荐目录结构

可以根据实际仓库微调，但建议形成类似：

```
.
├── config/
│   └── baseline.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── indexes/
│
├── scripts/
│   ├── prepare_rules.py
│   └── build_indexes.py
│
├── src/
│   └── emergency_rag/
│       ├── config.py
│       │
│       ├── data/
│       │   ├── models.py
│       │   └── repository.py
│       │
│       ├── chunking/
│       │   ├── base.py
│       │   └── rule.py
│       │
│       ├── clients/
│       │   ├── embedding.py
│       │   └── chat.py
│       │
│       ├── retrieval/
│       │   ├── models.py
│       │   ├── pipeline.py
│       │   │
│       │   ├── query/
│       │   │   ├── base.py
│       │   │   └── identity.py
│       │   │
│       │   ├── retrievers/
│       │   │   ├── base.py
│       │   │   ├── bm25.py
│       │   │   └── dense.py
│       │   │
│       │   ├── fusion/
│       │   │   ├── base.py
│       │   │   ├── union.py
│       │   │   └── rrf.py
│       │   │
│       │   ├── expansion/
│       │   │   ├── base.py
│       │   │   └── noop.py
│       │   │
│       │   ├── rerank/
│       │   │   ├── base.py
│       │   │   └── llm.py
│       │   │
│       │   ├── gate/
│       │   │   ├── base.py
│       │   │   └── passthrough.py
│       │   │
│       │   └── projector/
│       │       ├── base.py
│       │       └── identity.py
│       │
│       └── factory.py
│
├── tests/
│   ├── test_chunker.py
│   ├── test_rrf.py
│   ├── test_union.py
│   ├── test_pipeline.py
│   └── ...
│
├── pyproject.toml
└── .env.example
```

---

# 19. Pipeline Factory

增加统一 Factory，从配置构建 Baseline：

```
pipeline = build_retrieval_pipeline(config)
```

Baseline 应组装为：

```
QueryProcessor
    IdentityQueryProcessor

Retrievers
    BM25Retriever(bm25s)
    DenseRetriever(FAISS IndexFlatIP)

Fusion
    配置选择 UnionFusion / RRFFusion

Expansion
    NoOpExpander

Reranker
    ZeroEntropyReranker

Gate
    PassThroughGate

Projector
    IdentityProjector
```

这样上层代码无需手工实例化大量组件。

---

# 20. 最小运行入口

提供一个非常简单的 CLI，用于验证整个 Pipeline 已打通，例如：

```
python -m emergency_rag.cli retrieve "什么情况下可以结束危险化学品事故应急响应？"
```

输出 Top-K Rule，包含：

```
rule_id
rule_text
sparse_rank / score
dense_rank / score
fusion_score
rerank_score
sources
```

CLI 只是开发和验证入口，调用的必须仍然是：

```
pipeline.retrieve(query)
```

不要绕过 Pipeline 直接访问 Retriever。

---

# 21. 测试要求

至少覆盖：

- Rule 数据读取；

- RuleChunker 一对一映射；

- Candidate 模型；

- BM25 Retriever 基本检索；

- Dense Retriever 的 FAISS row 映射；

- embedding normalization；

- Union 去重；

- RRF 计算正确；

- Fusion 后能够保留 BM25 / Dense 的 rank 和 score；

- Reranker 输出顺序与 `rerank_score`；

- Pipeline 生命周期；

- Index 持久化与重新加载；

- Factory 能正确构造 Baseline。

涉及远程模型调用的测试使用 mock，避免 pytest 默认访问真实 API。

---

# 22. 实现质量要求

优先保证：

1. 接口边界清晰；

2. 单向依赖；

3. Pipeline 无实验模式分支；

4. Retriever、Fusion、Reranker 等组件可以独立测试；

5. 数据模型统一；

6. 配置集中；

7. 离线 scripts 与在线 src 分离；

8. 日志足够诊断索引构建和检索过程；

9. 代码保持简单直接，避免无必要的 Manager / Service / Factory 层级；

10. 类型标注完整；

11. 对关键设计写简洁中文注释。

完成后请：

1. 运行数据处理脚本；

2. 构建实际 BM25s 和 FAISS 索引；

3. 使用数据集中的真实 Rule 完成至少若干条手工查询；

4. 运行 pytest；

5. 修复发现的问题；

6. 最后给出简洁总结：

    - 最终目录结构；

    - Pipeline 生命周期；

    - Baseline 组成；

    - 数据和索引产物位置；

    - CLI 使用方法；

    - 测试结果。

目标不是实现一个展示型 RAG Demo，而是建立后续整个项目实验可以稳定复用的 Retrieval Core。
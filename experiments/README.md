# 规则选择题评估

实验读取已整理的选择题，检索完整规则，再调用 TypeSafe 或 Chat 选择 A–D 中的一个答案。题干用于检索；题干、四个选项和实际 Top-K 规则用于作答，标准答案和标注规则 ID 不发送给作答模型。

## 核心指标

主报告只展示以下四项，检索指标的 K 与实际作答的 `--answer-top-k` 一致，默认 K=3。


| 指标             | 计算口径                                                                              | 用途                             |
| ------------------ | --------------------------------------------------------------------------------------- | ---------------------------------- |
| 答案准确率       | 正确答案数 / 全部实验题数；A–D 标签精确匹配                                          | 衡量端到端选择题表现             |
| 规则 Recall@K    | 每题`标注规则与 Top-K 规则的交集数 / 标注规则数`，再取题目宏平均                      | 衡量标注证据找回多少             |
| 完整规则覆盖率@K | Top-K 包含该题全部标注规则的题数 / 参与检索评估的题数                                 | 区分多规则题的部分找回与全部找回 |
| nDCG@K           | 按标注规则 ID 判定二元相关性，DCG 使用`1/log2(名次+1)` 折扣，除以理想 DCG，再取宏平均 | 衡量相关规则是否排在前面         |

选择依据：

- 规则 Recall 对应 [Ragas ID Based Context Recall](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/context_recall/)，仓库已有规则 ID 标注，可以直接计算。
- [Phoenix 检索评估](https://arize.com/docs/phoenix/learn/retrieval-and-infrences/benchmarking-retrieval) 使用传统排序指标，并强调检索质量与最终答案正确性需要分别评估；这里用 nDCG 代表排序质量，准确率代表作答质量。
- 完整规则覆盖率是针对本数据集多规则标注的补充。它表示全部标注被找回，不代表这些规则在语义上必不可少，也不代表模型实际使用了它们。

所有规则指标在最终规则聚合、过滤后的排序上计算，不是 BM25、Dense 或融合阶段的独立召回率。规则按 ID 去重；未标注规则按不相关计分，因此分数反映与现有标注的一致性，标注可能未穷尽其他有效证据。

完整实验将检索失败计为零分，将 API 或解析失败计为答错；不以成功响应子集替代全题集。实验未完成时，准确率仅展示已处理题目的阶段值，不发布正式端到端准确率；检索阶段指标仅纳入已经检索或检索失败的题目。每个指标在 JSON 中保存分子、分母和值，避免阶段分母混淆。

其他 K 默认保存 1、3、5、10、20 的对照值，用于调整检索窗口。它们仍是同一组指标。概率、置信度、引用 ID、理由与原始响应保留在逐题分片中；摘要中的未供给引用 ID 是错误诊断。[Ragas Faithfulness](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/faithfulness/) 检查的是回答中的事实声明能否得到证据支持，引用 ID 合法不能替代这种检查。默认不增加 LLM 评委调用。

## 实验 CLI：命令、参数与结果影响

以下命令均在项目根目录执行。第一次使用时，先运行一次数据整理脚本，拆分选项并保存修复后的验证集；该脚本独立于实验 CLI。

```powershell
uv run python -m scripts.normalize_dev
```

配置服务密钥和本地 BCE 权重的方法见 [项目 README](../README.md)。实验默认读取 `data/processed/preliminary_dev.json`，使用 `config/preliminary/baseline.yaml` 和 TypeSafe 作答。

### 重点：指标窗口也决定最终检索数量

**修改 `--metric-top-k` 会改变评估窗口，也可能改变最终检索和保存的规则数量。这是实验条件的一部分，应在对照实验中明确记录。**

当前 CLI 使用以下关系：

```text
最终检索 Top-K = max(max(metric_top_k), answer_top_k)
实际作答证据 = 最终检索排序的前 answer_top_k 条规则
```

默认 `--metric-top-k 1 3 5 10 20`、`--answer-top-k 3`：最终检索最多 **20 条完整规则**，作答最多使用其中前 **3 条**。

| 指标窗口 `metric_top_k` | 作答窗口 `answer_top_k` | 最终检索上限 | 作答证据上限 |
| --- | ---: | ---: | ---: |
| `1 3 5 10 20`（默认） | 3（默认） | 20 | 3 |
| `1 3 5` | 3 | 5 | 3 |
| `1 3 5` | 10 | 10 | 10 |
| `1 3 5 10 20` | 5 | 20 | 5 |

例如以下命令最终检索最多 5 条，作答使用前 3 条，并将这个实验保存到单独目录：

```powershell
uv run python -m experiments run --metric-top-k 1 3 5 --answer-top-k 3 --run-dir results/preliminary/baseline-typesafe-answer-k3-metric-k5
```

`metric_top_k` 会去重并升序排列；报告自动补充作答 K 的指标，即使它未出现在 `--metric-top-k` 中。各数量都是上限，实际返回数量可能因候选不足、规则聚合或过滤而更少。

最终 Top-K 在规则聚合与过滤之后截断。修改指标窗口会改变日志中保存的完整检索范围，以及负样本报告能够定位的规则排名范围；在当前 Pipeline 中，它不调整 BM25 / Dense 的召回预算或 RRF 的融合预算。作答仍使用前 `answer_top_k` 条，因此不能仅凭最终检索数量变化，推断作答证据或答案一定变化。

当前 Baseline 的各阶段预算如下：

| 阶段 | 默认数量 | 配置入口与含义 |
| --- | ---: | --- |
| BM25 召回 | 30 | YAML 中 BM25 的 `params.top_k`，召回检索单元候选 |
| Dense 召回 | 30 | YAML 中 Dense 的 `params.top_k`，召回检索单元候选 |
| RRF 融合 | 最多 60 | YAML 中融合器的 `params.top_k`，融合后送入重排的候选上限 |
| 实验最终检索 | 最多 20 | CLI 根据指标窗口和作答窗口计算，返回完整规则 |
| 作答证据 | 最多 3 | CLI 的 `--answer-top-k`，截取最终排序前 3 条 |

独立调用 `pipeline.retrieve(query)` 时，函数默认 `top_k=10`；实验 CLI 会显式传入上述计算结果。BCE 的 `batch_size=32` 是推理批大小，和检索、作答 Top-K 分别控制不同的数量。

### 子命令概览

| 子命令 | 作用 | 是否调用模型 |
| --- | --- | --- |
| `run` | 创建新实验目录，冻结输入与配置，执行所选题目 | 是 |
| `resume` | 跳过已成功作答的题目，继续失败或未完成题目 | 有待执行题目时调用；可复用已保存检索 |
| `rerun` | 为指定题目创建新尝试，覆盖该题在报告中的最新结果 | 是；`--stage answer` 只重新作答 |
| `summarize` | 从日志与检查点重建四份报告 | 否 |

### `run`：完整参数说明

| 参数 | 默认值 / 可选值 | 作用 | 对实验结果的影响 |
| --- | --- | --- | --- |
| **`--dataset PATH`** | `data/processed/preliminary_dev.json` | 读取已整理的验证题目，含题干、选项、答案与标注规则 ID | **重点：改变输入题目、标准答案或标注，影响准确率与规则指标。** |
| **`--pipeline PATH`** | `config/preliminary/baseline.yaml` | 指定规则语料、索引及查询处理、召回、融合、重排、过滤配置 | **重点：改变检索证据和排序，可能改变作答；YAML 中的 `dataset_name` 用于结果目录的数据集隔离。** |
| **`--backend NAME`** | `typesafe`；可选 `typesafe` / `chat` | 选择作答客户端与响应解析方式 | **重点：改变作答服务与输出协议，影响答案和解析失败情况。** |
| **`--answer-top-k N`** | `3` | 最多发送前 N 条完整规则给作答模型；写入默认目录名 | **重点：直接改变作答证据窗口，也参与计算最终检索上限。** |
| **`--metric-top-k K [K ...]`** | `1 3 5 10 20` | 指定 Recall、完整覆盖率和 nDCG 的评估窗口 | **重点：改变检索评估口径；最大 K 还参与计算最终检索上限，见上文公式。** |
| **`--instructions-file PATH`** | 不指定时使用代码中的默认作答指令 | 从 UTF-8 文件读取非空作答指令；Chat 仍附加严格 JSON 格式要求 | **重点：改变提示词，可能改变答案、理由和引用。** |
| **`--limit N`** | 不指定则使用全部题目 | 按数据集原始顺序取前 N 题，不做随机抽样 | **重点：改变评估题集和统计分母；小批量结果应注明题数。** |
| **`--question-id ID`** | 不指定则不按 ID 筛选；可重复 | 仅选择指定题目，最终执行顺序沿用数据集顺序 | **重点：改变评估题集和统计分母；与 `--limit` 互斥。** |
| `--concurrency N` | `3` | 限制同时进行的远程作答请求；BCE 检索仍串行、仅一个实例 | **运行条件：影响吞吐、服务压力和限流风险，可能间接改变失败数量。** |
| `--shard-size N` | `50` | 每个 JSONL 分片最多保存 N 条完整尝试记录 | 存储参数：控制分片数量，指标按全部日志计算。 |
| `--run-dir PATH` | 自动生成，规则见下一节 | 指定新实验的输出目录；已存在目录报错，不覆盖 | 目录参数：用于隔离实验，参数变化时应使用新的目录。 |
| `-h` / `--help` | — | 显示 `run` 的帮助并退出 | 帮助参数，不执行实验。 |

所有 N、K 必须为正整数。`--question-id` 使用数据集中的字符串 ID，重复 ID 或未知 ID 会在模型请求前报错。

模型名、服务地址、超时和重试次数通过共享 `settings.py` / 环境变量配置，不是 CLI 参数。**Embedding 模型、重排模型、作答模型及其版本也会影响实验结果**；连接参数、提示词、题目与解析后的 Pipeline 配置会保存在 `manifest.json` 中。密钥不落盘。

### 结果目录与对照实验隔离

默认命名规则为：

```text
results/<YAML 中的 dataset_name>/<pipeline文件名去掉扩展名>-<backend>-answer-k<N>/
```

例如默认实验为 `results/preliminary/baseline-typesafe-answer-k3/`；`--answer-top-k 5` 对应 `baseline-typesafe-answer-k5/`，Chat 默认对应 `baseline-chat-answer-k3/`。显式指定 `--run-dir` 时使用指定路径。

**目录名仅包含 Pipeline 文件名、Backend 和作答 K。修改指标窗口、提示词、模型或题目子集时，默认目录名可能相同，必须用 `--run-dir` 指定另一目录，或为新检索方案使用新的 YAML 文件名。** 同名实验已经存在时不会覆盖，也不会自动追加到旧实验。

所有检索配置位于 `config/`，初赛集使用 `config/preliminary/`。实验直接复用共用的 Baseline YAML，解析后的参数保存在 manifest 中用于核对；结果目录中不另复制 YAML。

```powershell
# 默认完整实验；目录必须尚不存在
uv run python -m experiments run

# 小批量运行，使用独立目录
uv run python -m experiments run --limit 20 --run-dir results/preliminary/sample-typesafe-answer-k3

# 改变作答窗口，默认生成 baseline-typesafe-answer-k5/
uv run python -m experiments run --answer-top-k 5

# 使用自选题集，ID 参数可重复
uv run python -m experiments run --question-id 5 --question-id 32 --run-dir results/preliminary/selected-typesafe-answer-k3
```

### `resume`、`rerun`、`summarize`：完整参数说明

| 子命令 | 参数 | 默认值 / 要求 | 作用与影响 |
| --- | --- | --- | --- |
| `resume` | `--run-dir PATH` | 必填 | 读取该实验冻结的配置与进度；跳过最新尝试中已成功作答的题目，包括有效响应但选错的题目。需要重新回答错题时使用 `rerun`。 |
| `rerun` | `--run-dir PATH` | 必填 | 读取实验原有的题集与配置，追加新尝试。 |
| `rerun` | **`--question-id ID`** | 必填；可重复 | **指定重跑题目；题目必须属于该实验的冻结题集，最新尝试将改变总体和负样本报告。** |
| `rerun` | **`--stage STAGE`** | `all`；可选 `all` / `answer` | **`all` 重新检索并作答；`answer` 复用该题最近一次成功检索，只重跑作答。两者证据来源不同，分析时应注明。** |
| `summarize` | `--run-dir PATH` | 必填 | 从该实验日志和检查点重建 `summary.json`、`summary.md`、`negative.json`、`negative.md`。 |
| 全部子命令 | `-h` / `--help` | 可选 | 显示对应子命令帮助并退出。 |

`resume` 和 `rerun` 不接受新的 `--answer-top-k`、`--metric-top-k`、Backend 或提示词参数，而是沿用 manifest。**比较不同实验条件时，应创建新的 `run`。** 恢复前会核对冻结题目、Pipeline、模型连接参数和提示词；这些内容发生变化时需新建实验，允许更换密钥。

```powershell
# 继续失败与未完成的题目
uv run python -m experiments resume --run-dir results/preliminary/baseline-typesafe-answer-k3

# 重跑指定题目：重新检索并作答
uv run python -m experiments rerun --run-dir results/preliminary/baseline-typesafe-answer-k3 --question-id 5 --question-id 32 --stage all

# 复用成功检索，仅重新作答
uv run python -m experiments rerun --run-dir results/preliminary/baseline-typesafe-answer-k3 --question-id 5 --stage answer

# 只重建报告，不请求模型
uv run python -m experiments summarize --run-dir results/preliminary/baseline-typesafe-answer-k3

# 查看 CLI 帮助
uv run python -m experiments --help
uv run python -m experiments run --help
uv run python -m experiments rerun --help
```

退出码：正常完整运行且无检索 / 作答失败为 `0`（答错仍可返回 `0`）；有失败或未完成题目为 `1`；配置、输入或文件错误为 `2`；手动中断为 `130`。`summarize` 重建成功返回 `0`，实验是否完整请看报告。

## 总体摘要与失败样本

每次运行或 `summarize` 都从完整日志与恢复检查点生成四份报告：`summary.json`、`summary.md`、`negative.json`、`negative.md`。文件采用标准拼写 `negative`。

- **总体摘要（Summary）**：四项核心指标，以及全部已处理题目和负样本的作答 Top-K 覆盖分布。命中数量使用“命中标注规则数 / 所需标注规则数”，例如 `1/2` 表示两条标注规则只命中一条。
- **负样本（Negative Samples）**：每题最新尝试中的答错、检索失败、作答接口失败或解析失败；尚未完成的题目不当作失败。最新成功且答对的题目会从负样本报告移除，历史尝试仍保存在日志中。
- **逐题分析**：题干、选项、标准与预测答案、实际作答窗口中的规则 ID、命中数、命中与缺失 ID、标注规则在完整检索结果中的排名、各个 K 的覆盖、作答证据原文及错误原因。

主报告优先展示失败样本概览。比如标注规则排在第 4 名时，可以看到它在 Top-3 作答窗口缺失、但在 Top-5 检索窗口已命中。规则全部命中却答错只是一种观察，不能直接推断模型、提示词或标注存在问题；规则标注可能未穷尽其他有效证据。

不再维护 session 审计文件或重复的 CSV 汇总；运行时间戳、逐题耗时和原始响应仍保留在具体日志中。`manifest.json` 保存恢复所需的输入与配置，`checkpoint.json` 保存未完成进度。

## 持久化与资源预算

```text
config/preliminary/
├── baseline.yaml         BM25 + Dense、RRF、BCE 的共用 Baseline
└── example.yaml          单题检索示例配置
experiments/
├── __main__.py           CLI 参数与命令分发
├── data.py               已整理题目的输入契约
├── runner.py             检索、并发作答、恢复与重跑编排
├── answering.py          作答提示与响应解析
├── storage.py            完整日志、检查点、写入确认与目录锁
├── metrics.py            核心指标、覆盖分布与总体报告
└── negative.py           失败样本数据与逐题分析报告
scripts/                  一次性数据修复
tests/                   统一的组件与实验行为测试
results/preliminary/baseline-typesafe-answer-k3/
├── manifest.json         冻结题目、配置、提示与服务参数（无密钥）
├── checkpoint.json       尚未完成的尝试：启动信息与成功检索结果
├── logs/                 JSONL 分片，每题每次尝试一条完整记录
├── summary.json          机器可读的核心指标、规则命中分布与失败统计
├── summary.md            人类可读的总体评估与失败概览
├── negative.json         机器可读的失败样本明细
└── negative.md           人类可读的逐题失败分析
```

默认每 50 条完整记录切换分片。每条 JSONL 包含 `question_id`、`attempt`、`start`、`retrieval`、`answer`；检索失败也结束该尝试并保存一条记录，此时 `answer` 为 `null`。完整回答日志先写入、flush、fsync，随后才从检查点移除该尝试。当前保留的完整 TypeSafe 实验有 500 条记录、10 个分片。

启动信息与检索结果写入 `checkpoint.json`，不会单独占用分片日志。检索检查点确认落盘后才提交作答，进程中断后可复用成功检索；即使在完整日志落盘后、检查点清理前退出，恢复时也以完整日志为准，不重复追加同一尝试。

响应通过 question id 和 attempt 关联，保存实际查询与供给证据，不依赖并发返回顺序或缓存位置。重跑保留旧尝试，报告以最新尝试为准；整批重跑先保存启动检查点，最新待处理或失败的尝试不会回退成旧成功。

恢复时检查冻结输入和运行配置；变化需创建新实验，允许更换密钥。进程意外退出造成的最后一条不完整 JSONL 可在恢复时修复，完整记录损坏会报错。同一实验目录禁止多个写入进程。服务端已响应但本地尚未确认落盘的请求，恢复后可能重复调用；已确认记录可复用。

BCE 仅由主线程串行调用，默认 batch size 32；远程作答默认并发 3，可通过 `--concurrency` 调整。CUDA OOM 时 batch size 逐次减半到 1，候选数量不随之减少；仍失败则记录错误并停止新增任务。额度、鉴权、限流错误会停止提交新题，等待在途请求保存后退出，之后可恢复。并发作答不会启动额外 BCE 实例。

运行行为测试：`uv run pytest -q`。测试使用模拟作答服务，不代表真实答案准确率。

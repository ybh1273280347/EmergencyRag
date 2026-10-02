"""实验编排：串行检索、有限并发作答、持久化后推进以及按尝试恢复。"""

import gc
import json
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from time import perf_counter

import yaml

from emergency_rag import settings as settings_module
from emergency_rag.clients.chat import get_chat_client
from emergency_rag.clients.choice_qa_client import get_choice_qa_client
from emergency_rag.load_pipeline import load_pipeline

from .answering import CHAT_FORMAT, INSTRUCTIONS, perform_answer
from .data import Question, read_questions
from .metrics import summarize
from .storage import (
    Journal,
    Outcome,
    RunLock,
    atomic_json,
    read_history,
    retrieval_for,
    validate_context,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def pipeline_config(path: Path) -> dict:
    """冻结绝对路径，实验目录移动或启动目录变化不会重新解释语料路径。"""
    config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))

    if not isinstance(config, dict) or set(config) != {"dataset", "retrieval"}:
        raise ValueError("检索配置必须包含 dataset 与 retrieval")

    # 将 dataset 下的路径字段解析为绝对路径
    dataset = config["dataset"]
    for field, default in (("source", None), ("index_root", "data/indexes")):
        value = Path(dataset.get(field, default)).expanduser()
        dataset[field] = str(
            (value if value.is_absolute() else path.parent / value).resolve()
        )

    # BCE 重排器的本地模型路径同样固定为绝对路径
    reranker = config["retrieval"].get("reranker")
    if isinstance(reranker, dict) and reranker.get("name") == "bce":
        params = reranker.setdefault("params", {})
        params["model_path"] = str(
            Path(params.get("model_path", "models/bce-reranker-base_v1")).expanduser().resolve()
        )

    return config


def model_config(config: dict, backend: str) -> dict:
    settings = settings_module.settings
    prefixes = ["embedding", backend if backend == "chat" else "typesafe"]

    reranker = config["retrieval"].get("reranker", "default")
    if (reranker if isinstance(reranker, str) else reranker.get("name")) == "qwen":
        prefixes.append("qwen_reranker")

    # 白名单只包含普通连接参数；密钥可以更换以恢复额度，但绝不持久化
    values = {
        f"{prefix}_{field}": getattr(settings, f"{prefix}_{field}")
        for prefix in prefixes
        for field in ("model", "base_url", "timeout", "max_retries")
    }
    values["query_cache_root"] = str(settings.query_cache_root.resolve())
    return values


def select_questions(
    questions: list[Question],
    ids: list[str] | None,
    limit: int | None,
) -> list[Question]:
    if ids:
        unknown = set(ids) - {q.question_id for q in questions}
        if unknown:
            raise ValueError(f"未知 question_id：{sorted(unknown)}")
        if len(ids) != len(set(ids)):
            raise ValueError("question_id 不得重复")
        return [q for q in questions if q.question_id in ids]

    return questions[:limit] if limit else questions


def create_run(
    directory: Path,
    dataset: Path,
    pipeline: Path,
    *,
    backend: str = "typesafe",
    answer_top_k: int = 3,
    metric_top_k: list[int] | None = None,
    concurrency: int = 3,
    shard_size: int = 50,
    ids: list[str] | None = None,
    limit: int | None = None,
    instructions_path: Path | None = None,
) -> dict:
    if backend not in ("typesafe", "chat"):
        raise ValueError("未知作答 backend")
    if min(answer_top_k, concurrency, shard_size) <= 0 or (limit is not None and limit <= 0):
        raise ValueError("Top-K、并发、分片大小与 limit 必须为正整数")
    if ids and limit is not None:
        raise ValueError("question_id 与 limit 互斥")

    # 指标 cutoff 去重并升序，默认覆盖常用档位
    cutoffs = sorted(set(metric_top_k or [1, 3, 5, 10, 20]))
    if not cutoffs or min(cutoffs) <= 0:
        raise ValueError("指标 Top-K 必须为正整数")

    questions = select_questions(read_questions(dataset), ids, limit)
    config = pipeline_config(pipeline.resolve())

    # 作答指令可外部覆盖，否则用默认模板
    instructions = (
        instructions_path.read_text(encoding="utf-8-sig")
        if instructions_path
        else INSTRUCTIONS
    )
    if not instructions.strip():
        raise ValueError("作答指令不能为空")

    # manifest 冻结本次实验的全部输入与配置，供 resume / rerun 校验
    manifest = {
        "schema_version": 2,
        "dataset_name": config["dataset"]["dataset_name"],
        "created_at": now(),
        "dataset_source": str(dataset.resolve()),
        "pipeline_source": str(pipeline.resolve()),
        "pipeline": config,
        "models": model_config(config, backend),
        "backend": backend,
        "answer_top_k": answer_top_k,
        "metric_top_k": cutoffs,
        "concurrency": concurrency,
        "shard_size": shard_size,
        "instructions": instructions,
        "chat_format": CHAT_FORMAT,
        "instructions_source": (
            str(instructions_path.resolve()) if instructions_path else None
        ),
        "questions": [q.model_dump() for q in questions],
    }

    directory.mkdir(parents=True, exist_ok=False)
    atomic_json(directory / "manifest.json", manifest)

    return manifest


def validate_run(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != 2:
        raise ValueError("不支持的实验版本")

    # 冻结题目必须与源验证集完全一致
    current = {
        q.question_id: q.model_dump()
        for q in read_questions(Path(manifest["dataset_source"]))
    }
    if any(current.get(q["question_id"]) != q for q in manifest["questions"]):
        raise ValueError("验证集与冻结输入不同，请创建新实验")

    # 直接复用 config 中的 Pipeline YAML，manifest 保存解析后的配置用于核对
    if pipeline_config(Path(manifest["pipeline_source"])) != manifest["pipeline"]:
        raise ValueError("检索配置已改变，请创建新实验")
    # 模型连接参数与提示词同样纳入一致性校验
    if model_config(manifest["pipeline"], manifest["backend"]) != manifest["models"]:
        raise ValueError("模型配置已改变，请创建新实验")
    prompt_path = manifest.get("instructions_source")
    current_prompt = (
        Path(prompt_path).read_text(encoding="utf-8-sig")
        if prompt_path
        else INSTRUCTIONS
    )
    if current_prompt != manifest["instructions"] or CHAT_FORMAT != manifest["chat_format"]:
        raise ValueError("作答提示词已改变，请创建新实验")

    return manifest


def error_details(exc: Exception, stage: str) -> tuple[dict, bool]:
    """沿异常链提取状态码与错误码，并脱敏 API key；返回 (详情, 是否致命)。"""
    current = exc
    status = None
    code = None

    while current is not None:
        status = (
            status
            or getattr(current, "status_code", None)
            or getattr(current, "status", None)
        )
        response = getattr(current, "response", None)
        status = status or getattr(response, "status_code", None)

        body = getattr(current, "body", None)
        if isinstance(body, dict):
            error = body.get("error", body)
            if isinstance(error, dict):
                code = code or error.get("code")

        current = current.__cause__

    # 脱敏：正文与消息中的 API key 替换为占位符
    message = str(exc)
    raw_body = getattr(exc, "body", None)
    if raw_body is not None and not isinstance(raw_body, str):
        raw_body = json.dumps(raw_body, ensure_ascii=False, default=str)
    for name in ("embedding", "chat", "typesafe", "qwen_reranker"):
        key = getattr(settings_module.settings, name + "_api_key")
        if key:
            message = message.replace(key, "[redacted]")
            if raw_body is not None:
                raw_body = raw_body.replace(key, "[redacted]")

    details = {
        "stage": stage,
        "type": type(exc).__name__,
        "message": message,
        "status_code": status,
        "code": code,
        "response_body": raw_body,
    }
    halt = status in (401, 402, 403, 429) or code == "insufficient_quota"
    return details, halt


def is_cuda_oom(exc: Exception) -> bool:
    import torch
    return isinstance(exc, torch.cuda.OutOfMemoryError)


def clear_cuda():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def retrieve_question(pipeline, question: Question, top_k: int) -> dict:
    """只有 CUDA OOM 允许缩批；网络或索引失败不会被静默替换成其他策略。"""
    started = perf_counter()
    batch_sizes = []

    # 仅当使用 BCE 重排器时，才允许对 OOM 缩批重试
    bce = pipeline.reranker if pipeline.reranker.name == "bce" else None

    while True:
        if bce:
            batch_sizes.append(bce.batch_size)
        retry = False

        try:
            result = pipeline.retrieve(question.question_text, top_k=top_k)
            if result.query != question.question_text:
                raise ValueError("检索返回的 query 与题干不一致")

            return {
                "status": Outcome.SUCCESS,
                "query": question.question_text,
                "result": result.model_dump(mode="json"),
                "batch_sizes": batch_sizes,
                "latency_ms": (perf_counter() - started) * 1000,
            }

        except Exception as exc:
            oom = bce is not None and is_cuda_oom(exc)

            # OOM 且批大小 > 1：减半重试
            if oom and bce.batch_size > 1:
                bce.batch_size = max(1, bce.batch_size // 2)
                retry = True
            else:
                error, halt = error_details(exc, "retrieval")
                return {
                    "status": Outcome.ERROR,
                    "query": question.question_text,
                    "error": error,
                    "halt": halt or oom,
                    "batch_sizes": batch_sizes,
                    "latency_ms": (perf_counter() - started) * 1000,
                }

        # 离开 except 后再清理，异常栈不再引用失败推理的 GPU 张量
        if retry:
            clear_cuda()


def execute(
    directory: Path,
    mode: str = "run",
    *,
    ids: list[str] | None = None,
    stage: str = "all",
    pipeline_factory=load_pipeline,
    client_factory=None,
) -> dict:
    """每次执行只拥有一个 Pipeline；作答 worker 不访问索引和查询缓存。"""
    with RunLock(directory):
        manifest = validate_run(directory)
        questions = [Question.model_validate(row) for row in manifest["questions"]]
        selected = select_questions(questions, ids, None)

        # 重建历史尝试并校验落盘关联
        history = read_history(directory, repair_tail=True)
        validate_context(manifest["questions"], history, manifest["answer_top_k"])

        # 逐题决定本次是新增尝试、复用检索还是跳过
        jobs = []
        for question in selected:
            attempts = history.get(question.question_id, {})
            latest = max(attempts) if attempts else 0
            previous = attempts.get(latest)
            answer = previous["answer"] if previous else None
            reused = None

            # resume 模式下已成功作答的题目直接跳过
            if mode == "resume" and answer and answer["status"] == Outcome.SUCCESS:
                continue

            # rerun answer：复用最近一次成功的检索
            if mode == "rerun" and stage == "answer":
                reused = next(
                    (
                        a["retrieval"]
                        for _, a in sorted(attempts.items(), reverse=True)
                        if a["retrieval"] and a["retrieval"]["status"] == Outcome.SUCCESS
                    ),
                    None,
                )
                if reused is None:
                    raise ValueError(f"题目 {question.question_id} 没有可复用的成功检索")

            # resume 且已有历史：尝试复用上次检索
            elif mode == "resume" and previous:
                reused = retrieval_for(attempts, latest)

            # 上一尝试若已作答或检索出错，需要新开一 attempt
            failed = previous and (
                answer is not None
                or previous["retrieval"]
                and previous["retrieval"]["status"] == Outcome.ERROR
            )
            new_attempt = mode != "resume" or not previous or bool(failed)
            number = latest + 1 if new_attempt else latest
            jobs.append((question, number, new_attempt, reused))

        # 无任何待执行题目，直接汇总
        if not jobs:
            return summarize(directory)

        journal = Journal(directory, manifest["shard_size"])
        stop = Event()
        pipeline = None

        def request_answer(question, number, retrieval):
            """在 worker 线程内调用作答 SDK，并合并并持久化完整题目日志。"""
            answer_started = perf_counter()
            evidence = [
                {"rule_id": r["rule_id"], "text": r["text"]}
                for r in retrieval["result"]["evidence"][:manifest["answer_top_k"]]
            ]

            try:
                response = perform_answer(
                    question,
                    evidence,
                    manifest["backend"],
                    client,
                    manifest["instructions"],
                )
            except Exception as exc:
                error, halt = error_details(exc, "api")
                response = {
                    "status": Outcome.ERROR,
                    "prediction": None,
                    "raw_response": error["response_body"],
                    "error": error,
                }
                # 致命错误：立刻停止整批
                if halt:
                    stop.set()

            record = {
                "question_id": question.question_id,
                "attempt": number,
                "timestamp": now(),
                "backend": manifest["backend"],
                "retrieval_attempt": retrieval["attempt"],
                "question_text": question.question_text,
                "choice": question.choice,
                "evidence": evidence,
                "latency_ms": (perf_counter() - answer_started) * 1000,
                **response,
            }
            journal.answered(record)

        try:
            # 整批重跑先保存启动检查点，额度中断时未执行题目也不会退回旧成功答案
            for question, number, new_attempt, reused in jobs:
                if new_attempt:
                    journal.start({
                        "question_id": question.question_id,
                        "attempt": number,
                        "timestamp": now(),
                        "scope": "answer" if reused else "all",
                        "retrieval_attempt": reused["attempt"] if reused else None,
                    })

            # 仅在需要检索时才加载 Pipeline
            if any(reused is None for *_, reused in jobs):
                pipeline = pipeline_factory(Path(manifest["pipeline_source"]))

            client = (
                client_factory()
                if client_factory
                else (
                    get_choice_qa_client()
                    if manifest["backend"] == "typesafe"
                    else get_chat_client()
                )
            )

            futures = set()
            with ThreadPoolExecutor(
                max_workers=manifest["concurrency"],
                thread_name_prefix="qa",
            ) as pool:
                for question, number, _, reused in jobs:
                    # 已满并发：等待任一完成后再提交下一个
                    if len(futures) >= manifest["concurrency"]:
                        completed, futures = wait(futures, return_when=FIRST_COMPLETED)
                        for future in completed:
                            future.result()

                    # 致命错误后不再提交新任务
                    if stop.is_set():
                        break

                    # 检索在主线程串行执行，检索失败不进入作答
                    retrieval = reused
                    if retrieval is None:
                        outcome = retrieve_question(
                            pipeline,
                            question,
                            max(manifest["metric_top_k"] + [manifest["answer_top_k"]]),
                        )
                        retrieval = {
                            "question_id": question.question_id,
                            "attempt": number,
                            "timestamp": now(),
                            **outcome,
                        }
                        journal.retrieved(retrieval)
                        if outcome["status"] == Outcome.ERROR:
                            if outcome["halt"]:
                                stop.set()
                            continue

                    # 非致命错误时提交作答
                    if not stop.is_set():
                        futures.add(
                            pool.submit(request_answer, question, number, retrieval)
                        )

                # 等待剩余任务收尾
                for future in futures:
                    future.result()

        except BaseException:
            stop.set()
            raise

        finally:
            # 日志和检查点承担恢复职责，报告直接从这些数据重建，不单独记录 session。
            try:
                journal.close()
            finally:
                summary = summarize(directory)

        return summary
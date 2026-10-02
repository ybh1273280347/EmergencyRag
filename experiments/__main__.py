"""实验 CLI；配置和输入错误在发起模型请求前报告。"""

import argparse
import json
import logging
import sys
from pathlib import Path

from .data import NORMALIZED, ROOT
from .metrics import summarize
from .runner import create_run, execute, pipeline_config
from .storage import RunLock


def positive(value: str) -> int:
    """argparse 类型：只接受正整数。"""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须为正整数")
    return number


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description="规则选择题评估、JSONL 分片恢复与按题重跑")
    commands = cli.add_subparsers(dest="command", required=True)

    # run：创建并运行新实验
    run = commands.add_parser("run", help="创建并运行新实验")
    run.add_argument("--dataset", type=Path, default=NORMALIZED)
    run.add_argument("--pipeline", type=Path, default=ROOT / "config/preliminary/baseline.yaml")
    run.add_argument("--backend", choices=["typesafe", "chat"], default="typesafe")
    run.add_argument("--answer-top-k", type=positive, default=3)
    run.add_argument("--metric-top-k", type=positive, nargs="+", default=[1, 3, 5, 10, 20])
    run.add_argument("--concurrency", type=positive, default=3)
    run.add_argument("--shard-size", type=positive, default=50)
    run.add_argument("--instructions-file", type=Path)
    run.add_argument("--run-dir", type=Path, help="默认 results/<dataset_name>/<pipeline文件名>-<backend>-answer-k<N>")

    # 题集选择：limit 与 question-id 互斥
    selection = run.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=positive)
    selection.add_argument("--question-id", action="append")

    # resume：跳过成功题，继续失败与未完成题
    resume = commands.add_parser("resume", help="跳过成功题，继续失败和未完成题")
    resume.add_argument("--run-dir", type=Path, required=True)

    # rerun：为指定题创建新尝试
    rerun = commands.add_parser("rerun", help="创建指定题目的新尝试")
    rerun.add_argument("--run-dir", type=Path, required=True)
    rerun.add_argument("--question-id", action="append", required=True)
    rerun.add_argument("--stage", choices=["all", "answer"], default="all")

    # summarize：仅从已落盘分片重建报告，不调用模型
    report = commands.add_parser("summarize", help="仅从已保存分片重建报告，不调用模型")
    report.add_argument("--run-dir", type=Path, required=True)

    return cli


def main(argv=None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    try:
        if args.command == "run":
            # 作答证据窗口会改变模型输入，必须与 Pipeline、backend 一起标识实验。
            config = pipeline_config(args.pipeline.resolve())
            directory = args.run_dir or ROOT / "results" / config["dataset"]["dataset_name"] / (
                f"{args.pipeline.stem}-{args.backend}-answer-k{args.answer_top_k}"
            )
            create_run(
                directory,
                args.dataset,
                args.pipeline,
                backend=args.backend,
                answer_top_k=args.answer_top_k,
                metric_top_k=args.metric_top_k,
                concurrency=args.concurrency,
                shard_size=args.shard_size,
                ids=args.question_id,
                limit=args.limit,
                instructions_path=args.instructions_file,
            )
            print(f"实验目录：{directory.resolve()}", flush=True)
            summary = execute(directory)

        elif args.command == "summarize":
            # 重建报告也要持锁，避免与写入进程并发
            with RunLock(args.run_dir):
                summary = summarize(args.run_dir)

        else:
            # resume / rerun 共用 execute，仅参数不同
            summary = execute(
                args.run_dir,
                args.command,
                ids=getattr(args, "question_id", None),
                stage=getattr(args, "stage", "all"),
            )

        print(json.dumps(
            {
                "complete": summary["complete"],
                "counts": summary["counts"],
                "accuracy": summary["accuracy"],
            },
            ensure_ascii=False,
            indent=2,
        ))

        # summarize 是只读操作，总是返回 0；其余情况要求实验完整且无失败
        return (
            0
            if args.command == "summarize"
            or (summary["complete"] and not summary["counts"]["failed"])
            else 1
        )

    except KeyboardInterrupt:
        print("已中断；已落盘结果保留，请使用 resume。", file=sys.stderr)
        return 130

    except (OSError, ValueError, RuntimeError) as exc:
        print(f"实验错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

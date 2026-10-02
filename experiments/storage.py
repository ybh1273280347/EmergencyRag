"""每次尝试保存一条完整日志；未完成的启动与检索状态保存在原子检查点。"""

import copy
import json
import os
from concurrent.futures import Future, TimeoutError
from enum import StrEnum
from pathlib import Path
from queue import Empty, Full, Queue
from tempfile import NamedTemporaryFile
from threading import Thread


class Outcome(StrEnum):
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"


class RecordStage(StrEnum):
    START = "start"
    RETRIEVAL = "retrieval"
    ANSWER = "answer"
    COMPLETE = "complete"


def atomic_json(path: Path, value) -> None:
    """冻结输入、配置及派生报告完整写入后替换，失败不破坏旧文件。"""
    atomic_bytes(
        path,
        (
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8"),
    )


def atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = None
    try:
        # 先写临时文件并落盘，再用原子替换覆盖目标
        with NamedTemporaryFile(dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            output.write(content)
            output.flush()
            os.fsync(output.fileno())

        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class RunLock:
    """锁由内核随进程退出释放，残留的 lock 文件不代表运行仍存活。"""

    def __init__(self, directory: Path):
        self.path = directory / "run.lock"
        self.file = None

    def __enter__(self):
        self.file = self.path.open("a+b")

        # 确保文件至少有 1 字节，否则 Windows 无区域可加锁
        if self.file.tell() == 0:
            self.file.write(b"\0")
            self.file.flush()
        self.file.seek(0)

        # 非阻塞独占锁：拿不到说明已有写入进程
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.file.close()
            raise ValueError("该实验已有写入进程，请等待其结束") from exc

        return self

    def __exit__(self, *args):
        self.file.close()


def validate_record(record: dict, *, complete: bool) -> None:
    """检查日志/检查点的结构；具体题目与证据关联由 validate_context 验证。"""
    if not isinstance(record, dict) or set(record) != {
        "question_id", "attempt", "start", "retrieval", "answer"
    }:
        raise ValueError("记录必须包含题目 ID、尝试编号、启动、检索与回答")
    if (
        not isinstance(record["question_id"], str)
        or not record["question_id"]
        or type(record["attempt"]) is not int
        or record["attempt"] <= 0
        or not isinstance(record["start"], dict)
        or any(record[key] is not None and not isinstance(record[key], dict)
               for key in ("retrieval", "answer"))
    ):
        raise ValueError("记录的题目 ID、尝试编号或阶段内容非法")
    for part in ("start", "retrieval", "answer"):
        content = record[part]
        if content is not None and (
            content.get("question_id") != record["question_id"]
            or content.get("attempt") != record["attempt"]
        ):
            raise ValueError("阶段内容与完整记录的题目或尝试编号不一致")
    for part in ("retrieval", "answer"):
        content = record[part]
        if content is not None and content.get("status") not in (Outcome.SUCCESS, Outcome.ERROR):
            raise ValueError("记录含非法阶段状态")
    terminal = record["answer"] is not None or (
        record["retrieval"] is not None
        and record["retrieval"]["status"] == Outcome.ERROR
    )
    if terminal != complete:
        raise ValueError("完整日志与未完成检查点的状态不匹配")


def read_records(directory: Path, *, repair_tail: bool = False):
    """流式读取完整日志；仅最后一片的未完成末行可以修复。"""
    paths = sorted((directory / "logs").glob("part-*.jsonl"))
    for index, path in enumerate(paths, start=1):
        if path.name != f"part-{index:06d}.jsonl":
            raise ValueError("结果分片序号不连续")
        repair = repair_tail and path == paths[-1]
        with path.open("r+b" if repair_tail else "rb") as file:
            line_number = 0
            while True:
                offset = file.tell()
                line = file.readline()
                if not line:
                    break
                line_number += 1
                try:
                    record = json.loads(line)
                except (ValueError, UnicodeDecodeError) as exc:
                    # JSON 尚未写完的末行才允许截断；完整的错误记录必须暴露。
                    if repair and not line.endswith(b"\n"):
                        file.seek(offset)
                        file.truncate()
                        file.flush()
                        os.fsync(file.fileno())
                        break
                    raise ValueError(f"损坏分片 {path.name}:{line_number}") from exc
                try:
                    validate_record(record, complete=True)
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f"损坏分片 {path.name}:{line_number}") from exc
                # 完整合法的 JSON 末行即使被编辑器移除了换行也可读取；续写前补齐分隔。
                if not line.endswith(b"\n") and repair_tail:
                    file.seek(0, os.SEEK_END)
                    file.write(b"\n")
                    file.flush()
                    os.fsync(file.fileno())
                yield record


def read_checkpoint(directory: Path) -> list[dict]:
    path = directory / "checkpoint.json"
    if not path.exists():
        return []
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("schema_version") != 2
        or not isinstance(checkpoint.get("pending"), list)
    ):
        raise ValueError("不支持或损坏的检查点")
    for record in checkpoint["pending"]:
        validate_record(record, complete=False)
    return checkpoint["pending"]


def read_history(directory: Path, *, repair_tail: bool = False) -> dict[str, dict[int, dict]]:
    """以完整日志为权威，叠加未完成检查点，按尝试编号恢复进度。"""
    questions = {}
    for record in read_records(directory, repair_tail=repair_tail):
        attempts = questions.setdefault(record["question_id"], {})
        if record["attempt"] in attempts:
            raise ValueError("完整日志含重复尝试")
        attempts[record["attempt"]] = record
    pending_keys = set()
    for record in read_checkpoint(directory):
        qid, number = record["question_id"], record["attempt"]
        if (qid, number) in pending_keys:
            raise ValueError("检查点含重复尝试")
        pending_keys.add((qid, number))
        attempts = questions.setdefault(qid, {})
        if number in attempts:
            # 可能在日志 fsync 后、删除检查点前退出：保留完整日志，但必须核对上下文。
            completed = attempts[number]
            if record["start"] != completed["start"] or (
                record["retrieval"] is not None
                and record["retrieval"] != completed["retrieval"]
            ):
                raise ValueError("检查点与完整日志冲突")
            continue
        attempts[number] = record
    for qid, attempts in questions.items():
        if sorted(attempts) != list(range(1, len(attempts) + 1)):
            raise ValueError(f"题目 {qid} 的尝试编号不连续")
    return questions


class Journal:
    """单写入线程串行保存检查点和日志；确认 fsync 后才允许推进下一步。"""

    def __init__(self, directory: Path, shard_size: int = 50):
        if shard_size <= 0:
            raise ValueError("分片大小必须为正整数")
        self.run_directory = directory
        self.directory = directory / "logs"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        history = read_history(directory, repair_tail=True)
        # 已落盘完整日志优先，丢弃崩溃遗留的同一尝试检查点。
        self.pending = {
            (qid, number): record
            for qid, attempts in history.items()
            for number, record in attempts.items()
            if record["answer"] is None and (
                record["retrieval"] is None or record["retrieval"]["status"] != Outcome.ERROR
            )
        }
        self.completed = {
            (qid, number) for qid, attempts in history.items()
            for number in attempts if (qid, number) not in self.pending
        }
        paths = sorted(self.directory.glob("part-*.jsonl"))
        self.index = len(paths) or 1
        self.count = 0
        if paths:
            with paths[-1].open("rb") as file:
                self.count = sum(1 for _ in file)
        self.queue = Queue(maxsize=16)
        self.failure = None
        self.thread = Thread(target=self._write, name="experiment-journal")
        self.thread.start()

    def start(self, record: dict) -> None:
        self._submit(RecordStage.START, record)

    def retrieved(self, record: dict) -> None:
        self._submit(RecordStage.RETRIEVAL, record)

    def answered(self, record: dict) -> None:
        self._submit(RecordStage.ANSWER, record)

    def append(self, record: dict) -> None:
        """追加完整尝试；用于日志导入，运行时由 answered/retrieved 合并完成。"""
        self._submit(RecordStage.COMPLETE, record)

    def _submit(self, stage: RecordStage, record: dict):
        # 队列拥有独立快照，避免调用方或其他 worker 改写待持久化内容。
        record = copy.deepcopy(record)
        json.dumps(record, allow_nan=False)
        acknowledged = Future()
        while True:
            if self.failure is not None or not self.thread.is_alive():
                raise RuntimeError("结果写入线程已停止") from self.failure
            try:
                self.queue.put((stage, record, acknowledged), timeout=0.5)
                break
            except Full:
                continue
        while True:
            try:
                acknowledged.result(timeout=0.5)
                return
            except TimeoutError:
                if self.failure is not None or not self.thread.is_alive():
                    raise RuntimeError("结果写入失败") from self.failure

    def _write(self):
        output = None
        acknowledged = None
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    break
                stage, part, acknowledged = item
                key = (part["question_id"], part["attempt"])
                if key in self.completed:
                    raise ValueError("同一尝试已经完成，不能重复写入")
                if stage == RecordStage.START:
                    if key in self.pending:
                        raise ValueError("重复启动同一尝试")
                    record = {
                        "question_id": key[0], "attempt": key[1],
                        "start": part, "retrieval": None, "answer": None,
                    }
                    self.pending[key] = record
                elif stage == RecordStage.COMPLETE:
                    record = part
                else:
                    record = self.pending[key]
                    if record[stage] is not None:
                        raise ValueError("同一尝试重复写入阶段内容")
                    record[stage] = part
                complete = stage in (RecordStage.ANSWER, RecordStage.COMPLETE) or (
                    stage == RecordStage.RETRIEVAL and part["status"] == Outcome.ERROR
                )
                validate_record(record, complete=complete)
                if complete:
                    if output is None or self.count >= self.shard_size:
                        if output is not None:
                            output.close()
                        if self.count >= self.shard_size:
                            self.index += 1
                            self.count = 0
                        output = (self.directory / f"part-{self.index:06d}.jsonl").open("ab")
                    output.write((json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
                    output.flush()
                    os.fsync(output.fileno())
                    self.count += 1
                    self.completed.add(key)
                    self.pending.pop(key, None)
                # 完整日志必须先落盘，再删除检查点；恢复时完整日志覆盖遗留检查点。
                atomic_json(self.run_directory / "checkpoint.json", {
                    "schema_version": 2, "pending": list(self.pending.values())
                })
                acknowledged.set_result(None)
                acknowledged = None
        except BaseException as exc:
            self.failure = exc
            if acknowledged is not None:
                acknowledged.set_exception(exc)
            while True:
                try:
                    pending = self.queue.get_nowait()
                except Empty:
                    break
                if pending is not None:
                    pending[2].set_exception(exc)
        finally:
            if output is not None:
                output.close()

    def close(self):
        if self.thread.is_alive():
            self.queue.put(None)
        self.thread.join()
        if self.failure is not None:
            raise RuntimeError("结果写入失败") from self.failure


def retrieval_for(attempts: dict[int, dict], number: int) -> dict | None:
    attempt = attempts[number]
    own = attempt["retrieval"]

    # 本尝试自带检索记录，成功才可用
    if own is not None:
        return own if own["status"] == Outcome.SUCCESS else None

    # 否则尝试复用更早的检索尝试
    reference = attempt["start"].get("retrieval_attempt")
    if reference is None:
        return None
    if reference >= number or reference not in attempts:
        raise ValueError("复用的检索尝试必须早于当前尝试")

    record = attempts[reference]["retrieval"]
    if record is None or record["status"] != Outcome.SUCCESS:
        raise ValueError("复用的检索记录不存在或未成功")
    return record


def validate_context(questions: list[dict], history: dict, answer_top_k: int) -> None:
    """落盘记录是外部输入；恢复前核对题干、选项和实际证据的关联。"""
    by_id = {question["question_id"]: question for question in questions}
    if set(history) - set(by_id):
        raise ValueError("结果包含冻结输入以外的题目")

    for qid, attempts in history.items():
        question = by_id[qid]
        try:
            for number, attempt in attempts.items():
                own = attempt["retrieval"]

                # 本尝试检索记录：状态合法、query 与冻结题干一致
                if own:
                    if own["status"] not in (Outcome.SUCCESS, Outcome.ERROR):
                        raise ValueError("非法检索状态")
                    if own["query"] != question["question_text"]:
                        raise ValueError("检索题干与冻结输入不同")
                    if (
                        own["status"] == Outcome.SUCCESS
                        and own["result"]["query"] != own["query"]
                    ):
                        raise ValueError("检索结果 query 不一致")

                retrieval = retrieval_for(attempts, number)
                answer = attempt["answer"]
                if not answer:
                    continue

                # 作答状态合法，题干与选项必须与冻结输入一致
                if answer["status"] not in (Outcome.SUCCESS, Outcome.ERROR):
                    raise ValueError("非法作答状态")
                if (
                    answer["question_text"] != question["question_text"]
                    or answer["choice"] != question["choice"]
                ):
                    raise ValueError("作答题干或选项与冻结输入不同")

                # 作答必须绑定到本尝试实际使用的成功检索
                if retrieval is None or answer["retrieval_attempt"] != retrieval["attempt"]:
                    raise ValueError("作答缺少对应的成功检索尝试")

                # 落盘证据必须等于检索结果的前 answer_top_k 条
                supplied = [
                    {"rule_id": item["rule_id"], "text": item["text"]}
                    for item in retrieval["result"]["evidence"][:answer_top_k]
                ]
                if answer["evidence"] != supplied:
                    raise ValueError("作答证据与引用的检索尝试不同")

                if (
                    answer["status"] == Outcome.SUCCESS
                    and answer["prediction"] not in question["choice"]
                ):
                    raise ValueError("成功作答含非法答案")

        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"题目 {qid} 的分片关联损坏：{exc}") from exc

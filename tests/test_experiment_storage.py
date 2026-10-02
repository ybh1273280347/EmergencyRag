import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from experiments.storage import Journal, RunLock, atomic_json, read_history, read_records


def record(qid, attempt=1):
    return {
        "question_id": str(qid), "attempt": attempt,
        "start": {"question_id": str(qid), "attempt": attempt},
        "retrieval": {"question_id": str(qid), "attempt": attempt, "status": "success"},
        "answer": {"question_id": str(qid), "attempt": attempt, "status": "success"},
    }


def test_durable_rotation_and_concurrent_writers(tmp_path):
    journal = Journal(tmp_path, shard_size=3)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(journal.append, [record(i) for i in range(10)]))
    assert len(list(read_records(tmp_path))) == 10
    journal.close()
    files = sorted((tmp_path / "logs").glob("*.jsonl"))
    assert [len(p.read_bytes().splitlines()) for p in files] == [3, 3, 3, 1]
    assert len(read_history(tmp_path)) == 10


def test_start_and_retrieval_are_checkpoints_until_answer_completes(tmp_path):
    journal = Journal(tmp_path)
    start = {"question_id": "1", "attempt": 1}
    retrieval = {**start, "status": "success"}
    journal.start(start)
    journal.retrieved(retrieval)
    assert list(read_records(tmp_path)) == []
    assert read_history(tmp_path)["1"][1]["retrieval"] == retrieval
    journal.answered({**start, "status": "success", "raw_response": "answer"})
    journal.close()
    assert len(list(read_records(tmp_path))) == 1
    assert json.loads((tmp_path / "checkpoint.json").read_text())["pending"] == []


def test_retrieval_failure_also_finishes_one_record(tmp_path):
    journal = Journal(tmp_path)
    start = {"question_id": "1", "attempt": 1}
    journal.start(start)
    journal.retrieved({**start, "status": "error"})
    journal.close()
    saved = list(read_records(tmp_path))
    assert len(saved) == 1 and saved[0]["answer"] is None


def test_completed_log_wins_if_checkpoint_cleanup_was_interrupted(tmp_path):
    completed = record("1")
    journal = Journal(tmp_path)
    journal.append(completed)
    journal.close()
    pending = {**completed, "answer": None}
    atomic_json(tmp_path / "checkpoint.json", {"schema_version": 2, "pending": [pending]})
    assert read_history(tmp_path)["1"][1] == completed
    journal = Journal(tmp_path)
    journal.append(record("2"))
    journal.close()
    assert len(list(read_records(tmp_path))) == 2
    assert json.loads((tmp_path / "checkpoint.json").read_text())["pending"] == []


def test_conflicting_checkpoint_is_not_silently_discarded(tmp_path):
    journal = Journal(tmp_path)
    journal.append(record("1"))
    journal.close()
    pending = record("1")
    pending["answer"] = None
    pending["start"]["timestamp"] = "conflicting"
    atomic_json(tmp_path / "checkpoint.json", {"schema_version": 2, "pending": [pending]})
    with pytest.raises(ValueError, match="冲突"):
        read_history(tmp_path)


def test_tail_repair_drops_only_broken_final_record(tmp_path):
    journal = Journal(tmp_path)
    journal.append(record("1"))
    journal.close()
    path = tmp_path / "logs/part-000001.jsonl"
    with path.open("ab") as file:
        file.write(b'{"question_id":')
    with pytest.raises(ValueError):
        list(read_records(tmp_path))
    assert len(list(read_records(tmp_path, repair_tail=True))) == 1
    journal = Journal(tmp_path)
    journal.append(record("2"))
    journal.close()
    assert len(list(read_records(tmp_path))) == 2


def test_complete_tail_without_newline_is_preserved(tmp_path):
    (tmp_path / "logs").mkdir()
    path = tmp_path / "logs/part-000001.jsonl"
    path.write_bytes(json.dumps(record("1")).encode())
    assert list(read_records(tmp_path)) == [record("1")]
    assert list(read_records(tmp_path, repair_tail=True)) == [record("1")]
    assert path.read_bytes().endswith(b"\n")


@pytest.mark.parametrize("last", [True, False])
def test_corruption_of_completed_lines_is_not_hidden(tmp_path, last):
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs/part-000001.jsonl").write_bytes(b"broken\n")
    if not last:
        (tmp_path / "logs/part-000002.jsonl").write_bytes((json.dumps(record("2")) + "\n").encode())
    with pytest.raises(ValueError, match="损坏分片"):
        list(read_records(tmp_path, repair_tail=True))


def test_same_run_cannot_have_two_writers(tmp_path):
    with RunLock(tmp_path):
        with pytest.raises(ValueError, match="已有写入进程"):
            with RunLock(tmp_path):
                pass
    with RunLock(tmp_path):
        pass


def test_write_failure_reaches_producer_without_hanging(tmp_path, monkeypatch):
    import experiments.storage as storage
    journal = Journal(tmp_path)
    monkeypatch.setattr(storage.os, "fsync", lambda *args: (_ for _ in ()).throw(OSError("disk-full")))
    with pytest.raises((OSError, RuntimeError)):
        journal.append(record("1"))
    with pytest.raises(RuntimeError):
        journal.close()


def test_valid_json_with_invalid_contract_is_not_deleted_as_tail(tmp_path):
    (tmp_path / "logs").mkdir()
    path = tmp_path / "logs/part-000001.jsonl"
    original = b'{"question_id":"1","attempt":1}'
    path.write_bytes(original)
    with pytest.raises(ValueError, match="损坏分片"):
        list(read_records(tmp_path, repair_tail=True))
    assert path.read_bytes() == original

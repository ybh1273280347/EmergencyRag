import json
from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace

import pytest

from emergency_rag.retrieval.models import RetrievalResult, RuleEvidence
from experiments import runner
from experiments.__main__ import main
from experiments.data import NORMALIZED, Question, ROOT
from experiments.metrics import build_summary
from experiments.storage import atomic_json, read_history, read_records, retrieval_for


@pytest.fixture
def run_dir(tmp_path):
    dataset = tmp_path / "questions.json"
    atomic_json(dataset, [{"question_id": str(i), "question_text": f"题{i}",
                          "choice": {k: f"选项{k}" for k in "ABCD"},
                          "answer": "A" if i == 1 else "B", "rule_id": [str(i)]} for i in (1, 2)])
    directory = tmp_path / "run"
    pipeline = tmp_path / "baseline.yaml"
    atomic_json(pipeline, runner.pipeline_config(ROOT / "config/preliminary/baseline.yaml"))
    runner.create_run(directory, dataset, pipeline, shard_size=2, concurrency=2)
    return directory


class FakePipeline:
    def __init__(self):
        self.reranker = SimpleNamespace(name="default")
        self.calls = []

    def retrieve(self, query, top_k):
        self.calls.append((query, top_k))
        return RetrievalResult(query=query, evidence=[
            RuleEvidence(rule_id=query[-1], text="完整规则", final_rank=1)])


class FakeClient:
    def __init__(self, on_call=None):
        self.calls = []
        self.on_call = on_call

    def choose(self, **kwargs):
        self.calls.append(kwargs)
        if self.on_call:
            self.on_call(kwargs)
        label = "A" if kwargs["state"]["question_text"] == "题1" else "B"
        values = {k: 0.7 if k == label else 0.1 for k in "ABCD"}
        result = {"choice": label, "confidence": 0.7, "probabilities": values}
        return SimpleNamespace(**result, model_dump=lambda **kwargs: result)


def test_full_run_resume_and_independent_reruns(run_dir):
    pipeline = FakePipeline()

    def check_durable(kwargs):
        qid = kwargs["state"]["question_text"][-1]
        history = read_history(run_dir)
        attempts = history[qid]
        assert retrieval_for(attempts, max(attempts))["query"] == kwargs["state"]["question_text"]
        assert "answer" not in kwargs["state"]
        assert kwargs["state"]["evidence"] == [{"rule_id": qid, "text": "完整规则"}]

    client = FakeClient(check_durable)
    summary = runner.execute(run_dir, pipeline_factory=lambda path: pipeline, client_factory=lambda: client)
    assert summary["complete"] and summary["accuracy"]["end_to_end"]["value"] == 1
    assert pipeline.calls == [("题1", 20), ("题2", 20)]
    runner.execute(run_dir, "resume", pipeline_factory=lambda path: pytest.fail("不应加载模型"),
                   client_factory=lambda: pytest.fail("不应创建客户端"))
    assert len(client.calls) == 2
    runner.execute(run_dir, "rerun", ids=["1"], stage="answer",
                   pipeline_factory=lambda path: pytest.fail("仅作答不加载 BCE"), client_factory=lambda: client)
    assert len(pipeline.calls) == 2 and len(client.calls) == 3
    runner.execute(run_dir, "rerun", ids=["2"], pipeline_factory=lambda path: pipeline, client_factory=lambda: client)
    history = read_history(run_dir)
    assert len(history["1"]) == len(history["2"]) == 2
    assert len(pipeline.calls) == 3
    assert all((run_dir / name).exists() for name in ("summary.json", "summary.md", "negative.json", "negative.md"))
    assert len(list(read_records(run_dir))) == 4
    assert [len(p.read_bytes().splitlines()) for p in sorted((run_dir / "logs").glob("*.jsonl"))] == [2, 2]
    assert json.loads((run_dir / "checkpoint.json").read_text())["pending"] == []
    assert not (run_dir / "pipeline.yaml").exists()
    report = (run_dir / "summary.md").read_text(encoding="utf-8")
    assert "答案准确率（Accuracy）" in report
    assert "完整规则覆盖率（All-Gold Coverage）" in report
    assert "归一化折损累计增益（nDCG）" in report
    assert "100.00%" in report and "{'planned'" not in report


def test_unknown_id_and_answer_only_without_retrieval_are_preflight_errors(run_dir):
    never = lambda *args: pytest.fail("不得调用模型")
    with pytest.raises(ValueError, match="未知"):
        runner.execute(run_dir, "rerun", ids=["missing"], pipeline_factory=never, client_factory=never)
    with pytest.raises(ValueError, match="没有可复用"):
        runner.execute(run_dir, "rerun", ids=["1"], stage="answer", pipeline_factory=never, client_factory=never)
    assert not list((run_dir / "logs").glob("*.jsonl"))


def test_quota_halt_then_resume_uses_saved_retrieval(run_dir):
    class QuotaError(Exception):
        status_code = 402

    pipeline = FakePipeline()
    bad = FakeClient(lambda kwargs: (_ for _ in ()).throw(QuotaError("no credits")))
    summary = runner.execute(run_dir, pipeline_factory=lambda path: pipeline, client_factory=lambda: bad)
    assert summary["counts"]["failed"] >= 1
    negative = json.loads((run_dir / "negative.json").read_text())
    assert any(sample["error"]["status_code"] == 402 for sample in negative["samples"])
    good = FakeClient()
    summary = runner.execute(run_dir, "resume", pipeline_factory=lambda path: pipeline, client_factory=lambda: good)
    assert summary["complete"] and summary["counts"]["failed"] == 0
    assert len(pipeline.calls) == 2
    assert len(good.calls) == 2
    assert summary["counts"]["attempts"] > 2


def test_resume_pending_request_never_changes_its_attempt_or_context(run_dir):
    import experiments.storage as storage
    pipeline = FakePipeline()
    runner.execute(run_dir, pipeline_factory=lambda path: pipeline, client_factory=lambda: FakeClient())
    # 响应未保存时只留启动与检索检查点，恢复使用原 attempt 和证据。
    records = list(read_records(run_dir))
    for path in (run_dir / "logs").glob("*.jsonl"):
        path.unlink()
    journal = storage.Journal(run_dir, shard_size=2)
    for record in records:
        if record["question_id"] == "2":
            journal.start(record["start"])
            journal.retrieved(record["retrieval"])
        else:
            journal.append(record)
    journal.close()
    client = FakeClient()
    runner.execute(run_dir, "resume", pipeline_factory=lambda path: pytest.fail("不得重检索"), client_factory=lambda: client)
    assert len(client.calls) == 1 and client.calls[0]["state"]["question_text"] == "题2"
    assert max(read_history(run_dir)["2"]) == 1


def test_latest_failed_attempt_is_not_replaced_with_old_correct_answer(run_dir):
    pipeline = FakePipeline()
    runner.execute(run_dir, pipeline_factory=lambda path: pipeline, client_factory=lambda: FakeClient())
    bad = FakeClient(lambda kwargs: (_ for _ in ()).throw(ConnectionError("temporary")))
    summary = runner.execute(run_dir, "rerun", ids=["1"], stage="answer", client_factory=lambda: bad)
    assert summary["complete"]
    assert summary["accuracy"]["end_to_end"]["value"] == 0.5
    assert summary["counts"]["failed"] == 1
    assert len(read_history(run_dir)["1"]) == 2


def test_pending_rerun_does_not_publish_official_accuracy(run_dir):
    pipeline = FakePipeline()
    runner.execute(run_dir, pipeline_factory=lambda path: pipeline, client_factory=lambda: FakeClient())
    with pytest.raises(RuntimeError, match="setup failed"):
        runner.execute(run_dir, "rerun", ids=["1"], pipeline_factory=lambda path: (_ for _ in ()).throw(RuntimeError("setup failed")))
    summary = json.loads((run_dir / "summary.json").read_text())
    assert not summary["complete"] and summary["counts"]["pending"] == 1
    assert summary["accuracy"]["end_to_end"] is None


def test_snapshot_changes_refuse_requests(run_dir):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    dataset = json.loads(Path(manifest["dataset_source"]).read_text(encoding="utf-8"))
    dataset[0]["choice"]["A"] = "变更选项"
    atomic_json(Path(manifest["dataset_source"]), dataset)
    with pytest.raises(ValueError, match="冻结输入"):
        runner.execute(run_dir, "resume", pipeline_factory=lambda path: pytest.fail("不得请求"))


def test_responses_saved_while_main_thread_is_still_retrieving(run_dir):
    answer_durable = Event()
    original = runner.Journal.answered

    def track(self, record):
        original(self, record)
        if record["question_id"] == "1":
            answer_durable.set()

    class BlockingPipeline(FakePipeline):
        def retrieve(self, query, top_k):
            if query == "题2":
                assert answer_durable.wait(timeout=5)
            return super().retrieve(query, top_k)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(runner.Journal, "answered", track)
        runner.execute(run_dir, pipeline_factory=lambda path: BlockingPipeline(), client_factory=lambda: FakeClient())
    assert answer_durable.is_set()


def test_gpu_oom_shrinks_batch_without_truncating_candidates(monkeypatch):
    import torch
    batches = []
    budget = []
    class Pipeline(FakePipeline):
        def __init__(self):
            super().__init__()
            self.reranker = SimpleNamespace(name="bce", batch_size=32)

        def retrieve(self, query, top_k):
            batches.append(self.reranker.batch_size)
            budget.append(top_k)
            if self.reranker.batch_size > 4:
                raise torch.cuda.OutOfMemoryError("fake OOM")
            return super().retrieve(query, top_k)

    monkeypatch.setattr(runner, "clear_cuda", lambda: None)
    question = Question(question_id="1", question_text="题1", choice={k: k for k in "ABCD"}, answer="A", rule_id=["1"])
    outcome = runner.retrieve_question(Pipeline(), question, 20)
    assert outcome["status"] == "success"
    assert batches == [32, 16, 8, 4] and budget == [20] * 4
    pipeline = Pipeline()
    pipeline.reranker.batch_size = 1
    pipeline.retrieve = lambda *args, **kwargs: (_ for _ in ()).throw(torch.cuda.OutOfMemoryError("fake"))
    outcome = runner.retrieve_question(pipeline, question, 20)
    assert outcome["status"] == "error" and outcome["halt"]


def test_cli_usage_and_invalid_selection_do_not_create_run(tmp_path):
    assert main(["run", "--dataset", str(NORMALIZED), "--question-id", "missing",
                 "--run-dir", str(tmp_path / "bad")]) == 2
    assert not (tmp_path / "bad").exists()
    with pytest.raises(SystemExit):
        main(["run", "--limit", "2", "--question-id", "1"])


@pytest.mark.parametrize("backend", ["typesafe", "chat"])
def test_default_result_directory_uses_dataset_pipeline_and_backend(tmp_path, monkeypatch, backend):
    import experiments.__main__ as cli
    created = []
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "create_run", lambda directory, *args, **kwargs: created.append(directory))
    monkeypatch.setattr(cli, "execute", lambda directory: {"complete": True, "counts": {"failed": 0}, "accuracy": {}})
    assert cli.main(["run", "--pipeline", str(ROOT / "config/preliminary/baseline.yaml"), "--backend", backend]) == 0
    assert created == [tmp_path / "results/preliminary" / f"baseline-{backend}-answer-k3"]


def test_actual_typesafe_http_errors_halt_and_keep_body():
    from typesafe_sdk._core.errors import TypeSafeAPIError
    import httpx2
    for status in (401, 402, 403, 429):
        exc = TypeSafeAPIError(status, {"error": "quota-or-auth"}, httpx2.Headers())
        error, halt = runner.error_details(exc, "api")
        assert halt and error["status_code"] == status
        assert json.loads(error["response_body"]) == {"error": "quota-or-auth"}


def test_frozen_model_and_pipeline_changes_refuse_resume(run_dir, monkeypatch):
    monkeypatch.setattr(runner.settings_module.settings, "typesafe_model", "changed-model")
    with pytest.raises(ValueError, match="模型配置"):
        runner.execute(run_dir, "resume")
    monkeypatch.undo()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    config_path = Path(manifest["pipeline_source"])
    config = json.loads(config_path.read_text())
    config["retrieval"]["reranker"]["params"]["batch_size"] = 1
    atomic_json(config_path, config)
    with pytest.raises(ValueError, match="检索配置已改变"):
        runner.execute(run_dir, "resume")


def test_raw_responses_out_of_order_stay_bound_to_their_questions(run_dir, monkeypatch):
    second_saved = Event()
    original = runner.Journal.answered
    order = []
    active = 0
    maximum = 0
    lock = Lock()

    def track(self, record):
        original(self, record)
        if record["question_id"] in ("1", "2"):
            order.append(record["question_id"])
            if record["question_id"] == "2":
                second_saved.set()

    def concurrent(kwargs):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        if kwargs["state"]["question_text"] == "题1":
            assert second_saved.wait(5)
        with lock:
            active -= 1

    monkeypatch.setattr(runner.Journal, "answered", track)
    summary = runner.execute(run_dir, pipeline_factory=lambda path: FakePipeline(),
                             client_factory=lambda: FakeClient(concurrent))
    assert order == ["2", "1"] and maximum == 2
    assert summary["accuracy"]["end_to_end"]["value"] == 1
    history = read_history(run_dir)
    assert history["1"][1]["answer"]["raw_response"]["choice"] == "A"
    assert history["2"][1]["answer"]["raw_response"]["choice"] == "B"


def test_compact_metrics_preserve_raw_probability_and_citation_diagnostics(run_dir):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    runner.execute(run_dir, pipeline_factory=lambda path: FakePipeline(), client_factory=lambda: FakeClient())
    history = read_history(run_dir)
    history["1"][1]["answer"]["cited_rule_ids"] = ["1", "not-provided"]
    history["2"][1]["answer"]["cited_rule_ids"] = []
    history["2"][1]["answer"]["probabilities"] = {k: 0.1 for k in "ABCD"}
    summary, _ = build_summary(manifest, history)
    assert summary["diagnostics"] == {
        "citation_sample_count": 2,
        "unsupported_citations": {"1": ["not-provided"]},
    }
    assert "probability" not in summary and "conditional_accuracy" not in summary
    assert set(summary["retrieval"]["3"]) == {"recall", "all_gold_coverage", "ndcg"}
    assert history["1"][1]["answer"]["probabilities"] is not None
    assert history["2"][1]["answer"]["probabilities"] == {k: 0.1 for k in "ABCD"}


def test_pipeline_loading_interruption_can_be_resumed(run_dir):
    def interrupted(path):
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        runner.execute(run_dir, pipeline_factory=interrupted)
    assert len(read_history(run_dir)) == 2
    summary = runner.execute(run_dir, "resume", pipeline_factory=lambda path: FakePipeline(),
                             client_factory=lambda: FakeClient())
    assert summary["complete"] and summary["counts"]["attempts"] == 2

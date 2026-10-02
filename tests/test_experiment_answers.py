import json
import math
from types import SimpleNamespace

import pytest

from experiments.answering import parse_chat, perform_answer
from experiments.data import NORMALIZED, read_questions
from experiments.metrics import rule_metrics


@pytest.mark.parametrize("raw", [
    '```json\n{"answer":"A","rule_id":[],"reason":"理由"}\n```',
    '{"answer":"A","rule_id":[]}',
    '{"answer":"AB","rule_id":[],"reason":"理由"}',
    '{"answer":"A","rule_id":[1],"reason":"理由"}',
    '{"answer":"A","rule_id":[],"reason":"  "}',
    '{"answer":"A","rule_id":[],"reason":"理由","extra":1}',
    '{"answer":"A","answer":"B","rule_id":[],"reason":"理由"}',
    '{"answer":"A","rule_id":[],"reason":NaN}',
])
def test_strict_chat_contract(raw):
    with pytest.raises(ValueError):
        parse_chat(raw)


def test_invalid_citation_is_separate_from_answer_correctness(tmp_path):
    question = read_questions(NORMALIZED)[0]
    requests = []
    raw = '{"answer":"A","rule_id":["not-provided"],"reason":"理由"}'
    client = SimpleNamespace(complete=lambda **kwargs: requests.append(kwargs) or raw)
    result = perform_answer(question, [{"rule_id": "463", "text": "规则原文"}], "chat", client, "指令")
    assert result["status"] == "success"
    assert result["cited_rule_ids"] == ["not-provided"]
    request = requests[0]
    assert request["response_format"] == {"type": "json_object"}
    state = json.loads(request["messages"][1]["content"])
    assert set(state) == {"question_text", "choice", "evidence"}
    assert "answer" not in state and "rule_id" not in state
    client.complete = lambda **kwargs: "non-json"
    failed = perform_answer(question, [], "chat", client, "指令")
    assert failed["raw_response"] == "non-json"
    assert failed["error"]["stage"] == "parse"


def test_hand_calculated_rule_metrics():
    result = rule_metrics(["a", "b"], ["x", "a", "a", "b"], 3)
    assert result["recall"] == result["all_gold_coverage"] == 1
    assert result["ndcg"] == pytest.approx((1 / math.log2(3) + 0.5) / (1 + 1 / math.log2(3)))
    assert rule_metrics(["a", "b"], [], 3)["ndcg"] == 0
    assert rule_metrics(["a"], ["a"], 3)["ndcg"] == 1
    partial = rule_metrics(["a", "b"], ["a", "x"], 3)
    assert partial["recall"] == 0.5
    assert partial["all_gold_coverage"] == 0
    assert set(partial) == {"recall", "all_gold_coverage", "ndcg"}

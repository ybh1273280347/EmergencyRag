import json

import pytest

from scripts.normalize_dev import RAW_QUESTIONS, normalize
from experiments.data import read_questions


def test_all_real_questions_normalize_without_changing_gold(tmp_path):
    destination = tmp_path / "dev.json"
    report = normalize(output=destination)
    raw = json.loads(RAW_QUESTIONS.read_text(encoding="utf-8-sig"))
    normalized = read_questions(destination)
    assert len(raw) == len(normalized) == report["count"] == 500
    for original, question in zip(raw, normalized, strict=True):
        assert question.question_id == original["question_id"]
        assert question.answer == original["answer"]
        assert question.rule_id == original["rule_id"]
        assert list(question.choice) == list("ABCD")
        assert set(question.model_dump()) == {"question_id", "question_text", "choice", "answer", "rule_id"}
        assert not question.question_text.startswith("问题：")
        assert "选择：" not in question.question_text
    by_id = {q.question_id: q for q in normalized}
    assert by_id["27"].choice["D"].startswith("仅向参与者")
    assert not by_id["286"].choice["A"].startswith("A.")
    assert "河流A、" in by_id["47"].choice["D"]
    assert {r["question_id"] for r in report["corrections"]} == {"27", "274", "286", "461"}


def test_bad_normalization_keeps_previous_output(tmp_path):
    source = tmp_path / "raw.json"
    source.write_text(json.dumps([{"question_id": "bad", "question_text": "没有选项"}]), encoding="utf-8")
    output = tmp_path / "normalized.json"
    output.write_text("previous", encoding="utf-8")
    with pytest.raises(ValueError, match="bad"):
        normalize(source=source, output=output)
    assert output.read_text() == "previous"



import json
from unittest.mock import Mock

import httpx2
import pytest
from typesafe_sdk import (
    ChoiceAnswer,
    RetryPolicy,
    TypeSafeAPIResponseValidationError,
    TypeSafeAuthenticationError,
    TypeSafeClient,
)

from emergency_rag import settings as settings_module
from emergency_rag.clients import choice_qa_client
from emergency_rag.clients.choice_qa_client import ChoiceQAClient, get_choice_qa_client
from emergency_rag.retrieval.models import RetrievalResult, RuleEvidence
from examples.choice_qa_experiment import run_choice_experiment


@pytest.fixture(autouse=True)
def isolate_choice_client_cache():
    get_choice_qa_client.cache_clear()
    yield
    if get_choice_qa_client.cache_info().currsize:
        get_choice_qa_client().client.close()
    get_choice_qa_client.cache_clear()


def response_body(answer):
    return {
        "model": "jev-1.13.0",
        "answers": {"answer": answer},
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }


def test_choice_request_returns_sdk_answer_and_forwards_evidence():
    calls = []

    def respond(request):
        calls.append((request.url.path, json.loads(request.content)))
        return httpx2.Response(200, json=response_body({
            "type": "choice", "choice": "B", "confidence": 0.9,
            "probabilities": {"A": 0.1, "B": 0.9},
        }))

    state = {"question": "谁批准应急结束？", "evidence": ["完整规则原文"]}
    instructions = "仅根据参考规则选择唯一正确选项。"
    choices = {"A": "任意队员", "B": "现场应急救援指挥部"}
    with TypeSafeClient(api_key="test", transport=httpx2.MockTransport(respond)) as sdk:
        client = ChoiceQAClient(sdk, model="jev-1.13.0")
        answer = client.choose(state=state, instructions=instructions, choices=choices)
    assert isinstance(answer, ChoiceAnswer)
    assert answer.choice == "B" and answer.confidence == 0.9
    assert answer.probabilities == {"A": 0.1, "B": 0.9}
    assert calls == [("/v1/systemone", {
        "model": "jev-1.13.0", "state": state,
        "questions": {"answer": {"type": "choice", "instructions": instructions, "criteria": choices}},
    })]


@pytest.mark.parametrize("invalid", [
    "missing", "wrong_type", "unknown_choice", "missing_probability",
    "extra_probability", "confidence", "probability", "nonfinite", "malformed",
])
def test_choice_invalid_responses_rejected(invalid):
    answer = {"type": "choice", "choice": "A", "confidence": 0.8, "probabilities": {"A": 0.8, "B": 0.2}}
    body = response_body(answer)
    if invalid == "missing":
        body["answers"] = {}
    elif invalid == "wrong_type":
        body["answers"]["answer"] = {"type": "noul", "noul": 1.0}
    elif invalid == "unknown_choice":
        answer["choice"] = "C"
    elif invalid == "missing_probability":
        del answer["probabilities"]["B"]
    elif invalid == "extra_probability":
        answer["probabilities"]["C"] = 0.0
    elif invalid == "confidence":
        answer["confidence"] = 1.1
    elif invalid == "probability":
        answer["probabilities"]["B"] = -0.2
    elif invalid == "nonfinite":
        answer["confidence"] = float("nan")
    else:
        answer["choice"] = 1

    # 响应经真实 SDK 解析，避免 fake 绕过 SDK 的字段类型契约。
    with TypeSafeClient(api_key="test", transport=httpx2.MockTransport(
        lambda request: httpx2.Response(200, content=json.dumps(body)),
    )) as sdk:
        with pytest.raises((ValueError, TypeSafeAPIResponseValidationError)):
            ChoiceQAClient(sdk, model="jev-latest").choose(
                state="规则", instructions="选唯一正确答案", choices={"A": "选项 A", "B": "选项 B"},
            )


def test_choice_authentication_error_propagates():
    with TypeSafeClient(api_key="test", retry=RetryPolicy(max_retries=0), transport=httpx2.MockTransport(
        lambda request: httpx2.Response(401, json={"error": {"message": "unauthorized"}}),
    )) as sdk:
        with pytest.raises(TypeSafeAuthenticationError):
            ChoiceQAClient(sdk, model="jev-latest").choose(
                state="规则", instructions="选唯一正确答案", choices={"A": "选项 A", "B": "选项 B"},
            )


def test_choice_factory_reads_settings_and_caches(monkeypatch):
    constructors = []

    def construct(**kwargs):
        constructors.append(kwargs)
        return TypeSafeClient(transport=httpx2.MockTransport(lambda request: httpx2.Response(500)), **kwargs)

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-private-key")
    monkeypatch.setenv("TYPESAFE_DEFAULT_MODEL", "jev-1.13.0")
    monkeypatch.setenv("TYPESAFE_BASE_URL", "https://model.example")
    monkeypatch.setenv("TYPESAFE_TIMEOUT", "12")
    monkeypatch.setenv("TYPESAFE_MAX_RETRIES", "1")
    configured = settings_module.Settings()
    monkeypatch.setattr(choice_qa_client, "settings", configured)
    monkeypatch.setattr(choice_qa_client, "TypeSafeClient", construct)
    client = get_choice_qa_client()
    assert get_choice_qa_client() is client
    assert client.model == "jev-1.13.0"
    assert constructors == [{
        "api_key": "test-private-key", "model": "jev-1.13.0", "base_url": "https://model.example",
        "timeout": 12.0, "retry": RetryPolicy(max_retries=1),
    }]
    assert configured.typesafe_api_key == "test-private-key"
    assert "test-private-key" not in repr(configured)


def test_choice_missing_key():
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        get_choice_qa_client()
    assert get_choice_qa_client.cache_info().currsize == 0


def test_choice_experiment_sends_final_rule_evidence():
    pipeline = Mock()
    result = RetrievalResult(query="谁批准？", evidence=[
        RuleEvidence(rule_id="37", text="完整规则原文", metadata={"topic": "索引增强主题"}),
    ])
    pipeline.retrieve.return_value = result
    calls = []

    def respond(request):
        calls.append(json.loads(request.content))
        return httpx2.Response(200, json=response_body({
            "type": "choice", "choice": "A", "confidence": 1.0,
            "probabilities": {"A": 1.0, "B": 0.0},
        }))

    with TypeSafeClient(api_key="test", transport=httpx2.MockTransport(respond)) as sdk:
        actual_result, answer = run_choice_experiment(
            pipeline=pipeline, client=ChoiceQAClient(sdk, model="jev-1.13.0"),
            question="谁批准？", choices={"A": "指挥部", "B": "队员"},
            instructions="根据规则选择唯一正确答案", top_k=2,
        )
    pipeline.retrieve.assert_called_once_with("谁批准？", top_k=2)
    assert actual_result is result and answer.choice == "A"
    assert calls[0]["state"] == {
        "question": "谁批准？", "evidence": [{"rule_id": "37", "text": "完整规则原文"}],
    }
    assert calls[0]["questions"]["answer"]["instructions"] == "根据规则选择唯一正确答案"
    assert calls[0]["questions"]["answer"]["criteria"] == {"A": "指挥部", "B": "队员"}

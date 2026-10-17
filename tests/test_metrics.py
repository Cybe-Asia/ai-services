import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from app import llm_client, main
from app.config import Settings
from app.llm_client import LlmClient
from app.metrics import Registry, registry, usage_from_body


@pytest.fixture(autouse=True)
def fresh_registry():
    registry.reset()
    yield
    registry.reset()


def test_registry_renders_prometheus_text_without_content():
    local = Registry()
    local.observe_call("discovery", "ok", 1.2, {"input": 300, "output": 80})
    local.observe_call("discovery", "timeout", 61.0)
    local.count("discovery", "validation_failed")
    text = local.render()
    assert 'ai_feature_calls_total{feature="discovery",outcome="ok"} 1' in text
    assert 'ai_feature_calls_total{feature="discovery",outcome="timeout"} 1' in text
    assert 'ai_feature_call_seconds_bucket{feature="discovery",le="2"} 1' in text
    assert 'ai_feature_call_seconds_bucket{feature="discovery",le="+Inf"} 2' in text
    assert 'ai_feature_call_seconds_count{feature="discovery"} 2' in text
    assert 'ai_feature_tokens_total{feature="discovery",kind="input"} 300' in text
    assert 'ai_feature_tokens_total{feature="discovery",kind="output"} 80' in text
    assert 'ai_feature_outcomes_total{feature="discovery",outcome="validation_failed"} 1' in text


def test_labels_are_sanitised():
    local = Registry()
    local.observe_call('Disc"overy{}', "O K", 0.1)
    assert 'feature="disc_overy__",outcome="o_k"' in local.render()


@pytest.mark.parametrize(
    "body,expected",
    [
        ({"usage": {"prompt_tokens": 12, "completion_tokens": 3}}, {"input": 12, "output": 3}),
        ({"usage": {"input_tokens": 5, "output_tokens": 7}}, {"input": 5, "output": 7}),
        ({"prompt_eval_count": 9, "eval_count": 4}, {"input": 9, "output": 4}),
        ({"choices": []}, None),
        ({"usage": {"prompt_tokens": "x"}}, None),
    ],
)
def test_usage_is_normalised_across_providers(body, expected):
    assert usage_from_body(body) == expected


ORIGINAL_CLIENT = httpx.AsyncClient


def capture(monkeypatch, handler):
    monkeypatch.setattr(
        llm_client.httpx,
        "AsyncClient",
        lambda **kwargs: ORIGINAL_CLIENT(**kwargs, transport=httpx.MockTransport(handler)),
    )


def test_completion_records_outcome_latency_and_usage(monkeypatch):
    body = {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 2},
    }
    capture(monkeypatch, lambda request: httpx.Response(200, json=body))
    settings = Settings(_env_file=None, ai_provider_base_url="http://gateway.local/v1")
    answer = asyncio.run(LlmClient(settings).complete("System", "Hi", feature="assistant"))
    assert answer == "ok"
    text = registry.render()
    assert 'ai_feature_calls_total{feature="assistant",outcome="ok"} 1' in text
    assert 'ai_feature_tokens_total{feature="assistant",kind="input"} 20' in text
    assert 'ai_feature_call_seconds_count{feature="assistant"} 1' in text


def test_provider_errors_and_empty_answers_are_counted(monkeypatch):
    capture(monkeypatch, lambda request: httpx.Response(500, json={}))
    settings = Settings(_env_file=None, ai_provider_base_url="http://gateway.local/v1")
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(LlmClient(settings).complete("System", "Hi", feature="chat_draft"))
    capture(monkeypatch, lambda request: httpx.Response(200, json={"choices": []}))
    assert asyncio.run(LlmClient(settings).complete("System", "Hi", feature="chat_draft")) is None
    text = registry.render()
    assert 'ai_feature_calls_total{feature="chat_draft",outcome="error"} 1' in text
    assert 'ai_feature_calls_total{feature="chat_draft",outcome="empty"} 1' in text


def test_metrics_endpoint_is_plain_text_and_not_cached():
    registry.count("discovery", "ok")
    response = TestClient(main.app).get("/api/v1/ai-service/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.headers["cache-control"] == "no-store"
    assert 'ai_feature_outcomes_total{feature="discovery",outcome="ok"} 1' in response.text

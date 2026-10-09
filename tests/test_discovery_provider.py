import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from app import llm_client
from app import student_discovery as discovery
from app.config import Settings
from app.llm_client import LlmClient

GATEWAY = {"ai_provider_base_url": "http://gateway.local/v1", "ai_provider_api_key": "gateway-key"}
ANTHROPIC = {
    "discovery_provider_base_url": "https://provider.example",
    "discovery_provider_api_key": "provider-key",
    "discovery_model": "claude-haiku-5-5",
}


def settings(**values):
    return Settings(_env_file=None, ai_model="qwen3:4b", **GATEWAY, **values)


def capture(monkeypatch, body, seen):
    original = httpx.AsyncClient

    def response(request):
        seen.append(request)
        return httpx.Response(200, json=body)

    monkeypatch.setattr(
        llm_client.httpx,
        "AsyncClient",
        lambda **kwargs: original(**kwargs, transport=httpx.MockTransport(response)),
    )


@pytest.mark.parametrize(
    "partial",
    [{}, {k: v for k, v in ANTHROPIC.items() if k != "discovery_provider_api_key"}],
)
def test_discovery_stays_on_the_gateway_unless_fully_configured(partial):
    current = settings(**partial)
    assert current.for_discovery() is current


def test_discovery_provider_is_scoped_to_reflection_only():
    current = settings(**ANTHROPIC)
    scoped = current.for_discovery()
    assert scoped.ai_provider_protocol == "anthropic"
    assert scoped.ai_model == "claude-haiku-5-5"
    assert scoped.ai_provider_message_max_chars is None
    assert scoped.request_timeout_seconds == 20.0
    assert current.ai_model == "qwen3:4b"
    assert str(current.ai_provider_base_url) == "http://gateway.local/v1"


@pytest.mark.parametrize(
    "text",
    ['{"summary": []}', '```json\n{"summary": []}\n```', '```\n{"summary": []}\n```'],
)
def test_anthropic_messages_contract_and_fenced_json(monkeypatch, text):
    seen = []
    body = {"content": [{"type": "text", "text": text}], "stop_reason": "end_turn"}
    capture(monkeypatch, body, seen)
    answer = asyncio.run(
        LlmClient(settings(**ANTHROPIC).for_discovery()).complete(
            "System", "Material", response_format={"type": "json_object"}, require_complete=True
        )
    )
    assert answer.strip() == '{"summary": []}'
    request = seen[0]
    assert str(request.url) == "https://provider.example/v1/messages"
    assert request.headers["x-api-key"] == "provider-key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert "authorization" not in request.headers
    payload = json.loads(request.content)
    assert payload["model"] == "claude-haiku-5-5"
    assert payload["system"] == "System"
    assert payload["messages"] == [{"role": "user", "content": "Material"}]
    assert payload["max_tokens"] == 768
    assert "think" not in payload and "response_format" not in payload


@pytest.mark.parametrize(
    "body",
    [
        {"content": [{"type": "text", "text": "{}"}], "stop_reason": "max_tokens"},
        {"content": [{"type": "text", "text": "{}"}]},
        {"content": [], "stop_reason": "end_turn"},
        {"content": "{}", "stop_reason": "end_turn"},
    ],
)
def test_anthropic_releases_only_a_finished_text_answer(monkeypatch, body):
    capture(monkeypatch, body, [])
    client = LlmClient(settings(**ANTHROPIC).for_discovery())
    assert asyncio.run(client.complete("System", "Material")) is None


def test_reflection_uses_the_discovery_provider(monkeypatch):
    used = []

    async def complete(self, **kwargs):
        used.append(self._settings.ai_model)
        return None

    monkeypatch.setattr(discovery.LlmClient, "complete", complete)
    grounded = ({"course": "Math", "evidence": []}, {})
    monkeypatch.setattr(
        discovery,
        "resolve_student_context",
        lambda *a: _value(SimpleNamespace(student_id="student")),
    )
    monkeypatch.setattr(discovery, "learning_data", lambda *a, **k: _value({}))
    monkeypatch.setattr(discovery, "evidence_context", lambda *a: grounded)
    request = discovery.DiscoveryRequest(
        class_id="class", course_id="course", day="all", locale="id"
    )
    with pytest.raises(discovery.HTTPException):
        asyncio.run(discovery.reflect_student(settings(**ANTHROPIC), "Bearer t", request))
    assert used == ["claude-haiku-5-5"]


async def _value(value):
    return value

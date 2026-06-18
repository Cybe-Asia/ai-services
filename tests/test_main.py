import asyncio
import json
from datetime import datetime

import httpx
import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app import service_tools, thread_store
from app.auth_context import AdminContext
from app.config import Settings, get_settings
from app.main import app
from app.schemas import ActorRole, ChatRequest

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_thread_history():
    thread_store._MEMORY_THREADS.clear()
    thread_store._MEMORY_OWNER_INDEX.clear()
    yield
    thread_store._MEMORY_THREADS.clear()
    thread_store._MEMORY_OWNER_INDEX.clear()


async def fake_admin_context(settings, authorization):
    assert authorization == "Bearer test-token"
    return AdminContext(owner_id="owner-1", actor_role=ActorRole.admin)


def parse_sse_events(payload: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in payload.strip().split("\n\n"):
        event = "message"
        data = "{}"
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line.removeprefix("event: ").strip()
            if line.startswith("data: "):
                data = line.removeprefix("data: ").strip()
        events.append((event, json.loads(data)))
    return events


def test_health() -> None:
    response = client.get("/api/v1/ai-service/health")
    assert response.status_code == 200
    assert response.json()["service"] == "ai-service"


def test_metadata_reports_generation_limit() -> None:
    response = client.get("/api/v1/ai-service/metadata")
    assert response.status_code == 200
    body = response.json()
    assert body["maxTokens"] == 256
    assert body["paymentServiceUrl"] == "http://payment-service"


def test_public_marketing_prompt_is_allowed() -> None:
    response = client.post(
        "/api/ai/v1/chat",
        json={"message": "buatkan marketing ppdb", "actorRole": "public", "locale": "id"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "PPDB" in body["answer"]


def test_public_grade_prompt_is_refused() -> None:
    response = client.post(
        "/api/ai/v1/chat",
        json={"message": "siapa best student by nilai?", "actorRole": "public"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "refused"


def test_public_eoi_email_count_prompt_is_refused() -> None:
    response = client.post(
        "/api/ai/v1/chat",
        json={"message": "ada berapa email yang terdaftar di EOI?", "actorRole": "public"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "refused"


def test_public_registration_count_prompt_is_refused() -> None:
    response = client.post(
        "/api/ai/v1/chat",
        json={"message": "udah berapa orang yang daftar?", "actorRole": "public"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "refused"


def test_public_paraphrased_registration_count_prompt_is_refused() -> None:
    response = client.post(
        "/api/ai/v1/chat",
        json={"message": "berapa calon keluarga masuk sejauh ini?", "actorRole": "public"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "refused"


def test_public_payment_prompt_is_refused() -> None:
    response = client.post(
        "/api/ai/v1/chat",
        json={"message": "ada berapa pembayaran pending?", "actorRole": "public"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "refused"


def test_public_paraphrased_finance_prompt_is_refused() -> None:
    response = client.post(
        "/api/ai/v1/chat",
        json={
            "message": "berapa transaksi yang masih perlu dicek finance?",
            "actorRole": "public",
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "refused"


def test_public_admission_price_prompt_is_refused() -> None:
    response = client.post(
        "/api/ai/v1/chat",
        json={"message": "berapa harga untuk admissions?", "actorRole": "public"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "refused"


def test_admin_prompt_requires_bearer_token() -> None:
    response = client.post(
        "/api/ai/v1/chat",
        json={"message": "siapa best student by nilai?", "actorRole": "admin"},
    )
    assert response.status_code == 401


def test_admin_prompt_with_bearer_returns_pending_tool(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "resolve_admin_context", fake_admin_context)
    response = client.post(
        "/api/ai/v1/chat",
        headers={"authorization": "Bearer test-token"},
        json={"message": "siapa best student by nilai?", "actorRole": "admin"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["toolCalls"][0]["status"] == "not_configured"


def test_admin_sensitive_prompt_without_tool_does_not_call_draft_llm(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "resolve_admin_context", fake_admin_context)

    class ExplodingLlmClient:
        def __init__(self, settings):
            pass

        async def complete(self, *args, **kwargs):
            raise AssertionError("sensitive data fallback must not draft with LLM")

    monkeypatch.setattr(main_module, "LlmClient", ExplodingLlmClient)
    app.dependency_overrides[get_settings] = lambda: Settings(
        ai_provider_base_url="http://localhost:11434/v1"
    )
    try:
        response = client.post(
            "/api/ai/v1/chat",
            headers={"authorization": "Bearer test-token"},
            json={"message": "siapa best student by nilai?", "actorRole": "admin"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert "Tool yang aktif" in body["answer"]
    assert body["toolCalls"][0]["status"] == "not_configured"


def test_admin_chat_persists_server_side_thread_history(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "resolve_admin_context", fake_admin_context)

    async def fake_get_json(url, authorization, params):
        assert authorization == "Bearer test-token"
        if params == {"limit": "1", "offset": "0"}:
            return {"data": {"total": 1}}
        if params == {"limit": "5", "offset": "0"}:
            return {
                "data": {
                    "total": 1,
                    "rows": [
                        {
                            "parentName": "Arief Nugraha",
                            "email": "arief@example.test",
                            "school": "IISS",
                            "leadStatus": "verified",
                        }
                    ],
                }
            }
        raise AssertionError(f"unexpected params: {params}")

    monkeypatch.setattr(service_tools, "_get_json", fake_get_json)

    first = client.post(
        "/api/ai/v1/chat",
        headers={"authorization": "Bearer test-token"},
        json={"message": "ada berapa eoi sekarang?", "actorRole": "admin"},
    )
    assert first.status_code == 200
    conversation_id = first.json()["conversationId"]

    second = client.post(
        "/api/ai/v1/chat",
        headers={"authorization": "Bearer test-token"},
        json={
            "message": "siapa itu?",
            "actorRole": "admin",
            "conversationId": conversation_id,
        },
    )
    assert second.status_code == 200
    body = second.json()
    assert body["conversationId"] == conversation_id
    assert "Arief Nugraha" in body["answer"]
    assert body["toolCalls"][0] == {"name": "admission.admin_leads_list", "status": "ok"}

    listed = client.get(
        "/api/ai/v1/threads",
        headers={"authorization": "Bearer test-token"},
    )
    assert listed.status_code == 200
    assert listed.json()["threads"][0]["id"] == conversation_id
    assert listed.json()["threads"][0]["messageCount"] == 4

    loaded = client.get(
        f"/api/ai/v1/threads/{conversation_id}",
        headers={"authorization": "Bearer test-token"},
    )
    assert loaded.status_code == 200
    assert [message["role"] for message in loaded.json()["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_admin_chat_streams_tool_events_and_persists_thread(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "resolve_admin_context", fake_admin_context)

    async def fake_get_json(url, authorization, params):
        assert url == "http://admission-service/api/leads/v1/admin/leads"
        assert authorization == "Bearer test-token"
        assert params == {"limit": "1", "offset": "0"}
        return {"data": {"total": 6}}

    monkeypatch.setattr(service_tools, "_get_json", fake_get_json)

    response = client.post(
        "/api/ai/v1/chat/stream",
        headers={"authorization": "Bearer test-token"},
        json={"message": "ada berapa eoi sekarang?", "actorRole": "admin"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse_events(response.text)
    event_names = [event for event, _data in events]
    assert event_names[:3] == ["thread", "status", "status"]
    assert ("tool_call", {"name": "admission.admin_leads_count", "status": "ok"}) in events
    assert any(event == "delta" and "6 EOI" in data["text"] for event, data in events)

    done = events[-1]
    assert done[0] == "done"
    conversation_id = done[1]["conversationId"]
    assert done[1]["toolCalls"] == [{"name": "admission.admin_leads_count", "status": "ok"}]

    loaded = client.get(
        f"/api/ai/v1/threads/{conversation_id}",
        headers={"authorization": "Bearer test-token"},
    )
    assert loaded.status_code == 200
    assert loaded.json()["thread"]["messageCount"] == 2


def test_admin_chat_streams_contextual_detail_followup(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "resolve_admin_context", fake_admin_context)

    async def fake_get_json(url, authorization, params):
        assert authorization == "Bearer test-token"
        if params == {"limit": "1", "offset": "0"}:
            return {"data": {"total": 1}}
        if params == {"limit": "5", "offset": "0"}:
            return {
                "data": {
                    "total": 1,
                    "rows": [
                        {
                            "parentName": "Arief Nugraha",
                            "email": "arief@example.test",
                            "school": "SCH-IISS",
                            "leadStatus": "verified",
                        }
                    ],
                }
            }
        raise AssertionError(f"unexpected params: {params}")

    monkeypatch.setattr(service_tools, "_get_json", fake_get_json)

    first = client.post(
        "/api/ai/v1/chat",
        headers={"authorization": "Bearer test-token"},
        json={"message": "ada berapa eoi sekarang?", "actorRole": "admin"},
    )
    assert first.status_code == 200
    conversation_id = first.json()["conversationId"]

    response = client.post(
        "/api/ai/v1/chat/stream",
        headers={"authorization": "Bearer test-token"},
        json={
            "message": "mau lihat detailnya",
            "actorRole": "admin",
            "conversationId": conversation_id,
        },
    )

    assert response.status_code == 200
    events = parse_sse_events(response.text)
    assert ("tool_call", {"name": "admission.admin_leads_list", "status": "ok"}) in events
    streamed_text = "".join(data["text"] for event, data in events if event == "delta")
    assert "Arief Nugraha" in streamed_text
    assert events[-1][1]["toolCalls"] == [
        {"name": "admission.admin_leads_list", "status": "ok"}
    ]


def test_public_chat_streams_llm_deltas(monkeypatch) -> None:
    class FakeLlmClient:
        def __init__(self, settings):
            pass

        async def stream_complete(self, *args, **kwargs):
            yield "Halo "
            yield "dari Llama"

        async def complete(self, *args, **kwargs):
            raise AssertionError("stream endpoint should use stream_complete")

    monkeypatch.setattr(main_module, "LlmClient", FakeLlmClient)
    app.dependency_overrides[get_settings] = lambda: Settings(
        ai_provider_base_url="http://localhost:11434/v1"
    )
    try:
        response = client.post(
            "/api/ai/v1/chat/stream",
            json={"message": "buat sapaan ppdb", "actorRole": "public"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    events = parse_sse_events(response.text)
    assert [data["text"] for event, data in events if event == "delta"] == [
        "Halo ",
        "dari Llama",
    ]
    assert events[-1][1]["answer"] == "Halo dari Llama"


def test_admin_can_delete_thread(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "resolve_admin_context", fake_admin_context)

    created = client.post(
        "/api/ai/v1/threads",
        headers={"authorization": "Bearer test-token"},
        json={"title": "Ops check"},
    )
    assert created.status_code == 200
    thread_id = created.json()["thread"]["id"]

    deleted = client.delete(
        f"/api/ai/v1/threads/{thread_id}",
        headers={"authorization": "Bearer test-token"},
    )
    assert deleted.status_code == 200
    assert deleted.json() == {"status": "ok"}

    loaded = client.get(
        f"/api/ai/v1/threads/{thread_id}",
        headers={"authorization": "Bearer test-token"},
    )
    assert loaded.status_code == 404


def test_owner_eoi_count_uses_admission_tool(monkeypatch) -> None:
    async def fake_get_json(url, authorization, params):
        assert url == "http://admission-service/api/leads/v1/admin/leads"
        assert authorization == "Bearer test-token"
        assert params == {"limit": "1", "offset": "0"}
        return {"data": {"total": 7}}

    monkeypatch.setattr(service_tools, "_get_json", fake_get_json)
    result = asyncio.run(
        service_tools.answer_from_school_tools(
            ChatRequest(
                message="ada berapa email yang terdaftar di EOI?",
                actor_role=ActorRole.owner,
            ),
            Settings(),
            "Bearer test-token",
        )
    )

    assert result is not None
    assert "7 EOI" in result.answer
    assert result.tool_calls[0].name == "admission.admin_leads_count"
    assert result.tool_calls[0].status == "ok"


def test_owner_eoi_count_reports_expired_admin_session(monkeypatch) -> None:
    async def fake_get_json(url, authorization, params):
        request = httpx.Request("GET", url)
        response = httpx.Response(401, request=request)
        raise httpx.HTTPStatusError("unauthorized", request=request, response=response)

    monkeypatch.setattr(service_tools, "_get_json", fake_get_json)
    result = asyncio.run(
        service_tools.answer_from_school_tools(
            ChatRequest(
                message="ada berapa email yang terdaftar di EOI?",
                actor_role=ActorRole.owner,
            ),
            Settings(),
            "Bearer expired-token",
        )
    )

    assert result is not None
    assert "login ulang" in result.answer
    assert result.tool_calls[0].name == "admission.admin_leads_count"
    assert result.tool_calls[0].status == "auth_required"


def test_owner_registration_count_uses_admission_tool(monkeypatch) -> None:
    async def fake_get_json(url, authorization, params):
        assert url == "http://admission-service/api/leads/v1/admin/leads"
        assert authorization == "Bearer test-token"
        assert params == {"limit": "1", "offset": "0"}
        return {"data": {"total": 9}}

    monkeypatch.setattr(service_tools, "_get_json", fake_get_json)
    result = asyncio.run(
        service_tools.answer_from_school_tools(
            ChatRequest(
                message="udah berapa orang yang daftar?",
                actor_role=ActorRole.admin,
            ),
            Settings(),
            "Bearer test-token",
        )
    )

    assert result is not None
    assert "9 EOI" in result.answer
    assert result.tool_calls[0].name == "admission.admin_leads_count"
    assert result.tool_calls[0].status == "ok"


def test_owner_contextual_lead_identity_followup_uses_leads_list_tool(monkeypatch) -> None:
    async def fake_get_json(url, authorization, params):
        assert url == "http://admission-service/api/leads/v1/admin/leads"
        assert authorization == "Bearer test-token"
        assert params == {"limit": "5", "offset": "0"}
        return {
            "data": {
                "total": 1,
                "rows": [
                    {
                        "parentName": "Arief Nugraha",
                        "email": "arief@example.test",
                        "school": "IIEC-RI",
                        "leadStatus": "new",
                    }
                ],
            }
        }

    monkeypatch.setattr(service_tools, "_get_json", fake_get_json)
    result = asyncio.run(
        service_tools.answer_from_school_tools(
            ChatRequest(
                message="siapa orang itu?",
                actor_role=ActorRole.admin,
                history=[
                    {
                        "role": "assistant",
                        "content": "Ada 1 EOI terdaftar di admission-service.",
                        "toolCalls": [
                            {"name": "admission.admin_leads_count", "status": "ok"}
                        ],
                    }
                ],
            ),
            Settings(),
            "Bearer test-token",
        )
    )

    assert result is not None
    assert "Arief Nugraha" in result.answer
    assert "arief@example.test" in result.answer
    assert result.tool_calls[0].name == "admission.admin_leads_list"
    assert result.tool_calls[0].status == "ok"


def test_owner_contextual_detail_followup_uses_leads_list_tool(monkeypatch) -> None:
    class ExplodingLlmClient:
        def __init__(self, settings):
            pass

        async def complete(self, *args, **kwargs):
            raise AssertionError("detail follow-up should use deterministic admission tool")

    async def fake_get_json(url, authorization, params):
        assert url == "http://admission-service/api/leads/v1/admin/leads"
        assert authorization == "Bearer test-token"
        assert params == {"limit": "5", "offset": "0"}
        return {
            "data": {
                "total": 1,
                "rows": [
                    {
                        "parentName": "Arief Nugraha",
                        "email": "arief@example.test",
                        "school": "SCH-IISS",
                        "leadStatus": "verified",
                    }
                ],
            }
        }

    monkeypatch.setattr(service_tools, "LlmClient", ExplodingLlmClient)
    monkeypatch.setattr(service_tools, "_get_json", fake_get_json)
    result = asyncio.run(
        service_tools.answer_from_school_tools(
            ChatRequest(
                message="mau lihat detailnya",
                actor_role=ActorRole.admin,
                history=[
                    {
                        "role": "assistant",
                        "content": "Ada 1 EOI terdaftar di admission-service.",
                        "toolCalls": [
                            {"name": "admission.admin_leads_count", "status": "ok"}
                        ],
                    }
                ],
            ),
            Settings(ai_provider_base_url="http://localhost:11434/v1"),
            "Bearer test-token",
        )
    )

    assert result is not None
    assert "Arief Nugraha" in result.answer
    assert result.tool_calls[0].name == "admission.admin_leads_list"
    assert result.tool_calls[0].status == "ok"


def test_owner_yesterday_eoi_count_uses_date_filter(monkeypatch) -> None:
    monkeypatch.setattr(
        service_tools,
        "_now_jakarta",
        lambda: datetime(2026, 6, 18, 12, 0, tzinfo=service_tools.SCHOOL_TIME_ZONE),
    )

    async def fake_get_json(url, authorization, params):
        assert url == "http://admission-service/api/leads/v1/admin/leads"
        assert authorization == "Bearer test-token"
        assert params["limit"] == "1"
        assert params["offset"] == "0"
        assert params["dateFrom"] == "2026-06-16T17:00:00+00:00"
        assert params["dateTo"] == "2026-06-17T16:59:59.999000+00:00"
        return {"data": {"total": 2}}

    monkeypatch.setattr(service_tools, "_get_json", fake_get_json)
    result = asyncio.run(
        service_tools.answer_from_school_tools(
            ChatRequest(
                message="kalau kemarin ada berapa eoi?",
                actor_role=ActorRole.admin,
            ),
            Settings(),
            "Bearer test-token",
        )
    )

    assert result is not None
    assert "2 EOI" in result.answer
    assert "kemarin (17 Juni 2026)" in result.answer


def test_owner_relative_days_ago_eoi_count_uses_date_filter(monkeypatch) -> None:
    monkeypatch.setattr(
        service_tools,
        "_now_jakarta",
        lambda: datetime(2026, 6, 18, 12, 0, tzinfo=service_tools.SCHOOL_TIME_ZONE),
    )

    async def fake_get_json(url, authorization, params):
        assert url == "http://admission-service/api/leads/v1/admin/leads"
        assert authorization == "Bearer test-token"
        assert params["limit"] == "1"
        assert params["offset"] == "0"
        assert params["dateFrom"] == "2026-05-29T17:00:00+00:00"
        assert params["dateTo"] == "2026-05-30T16:59:59.999000+00:00"
        return {"data": {"total": 3}}

    monkeypatch.setattr(service_tools, "_get_json", fake_get_json)
    result = asyncio.run(
        service_tools.answer_from_school_tools(
            ChatRequest(
                message="ada berapa EOI 19 hari lalu?",
                actor_role=ActorRole.admin,
            ),
            Settings(),
            "Bearer test-token",
        )
    )

    assert result is not None
    assert "3 EOI" in result.answer
    assert "19 hari lalu (30 Mei 2026)" in result.answer


def test_owner_contextual_relative_days_ago_eoi_count_uses_date_filter(monkeypatch) -> None:
    monkeypatch.setattr(
        service_tools,
        "_now_jakarta",
        lambda: datetime(2026, 6, 18, 12, 0, tzinfo=service_tools.SCHOOL_TIME_ZONE),
    )

    async def fake_get_json(url, authorization, params):
        assert url == "http://admission-service/api/leads/v1/admin/leads"
        assert authorization == "Bearer test-token"
        assert params["limit"] == "1"
        assert params["offset"] == "0"
        assert params["dateFrom"] == "2026-05-29T17:00:00+00:00"
        assert params["dateTo"] == "2026-05-30T16:59:59.999000+00:00"
        return {"data": {"total": 3}}

    monkeypatch.setattr(service_tools, "_get_json", fake_get_json)
    result = asyncio.run(
        service_tools.answer_from_school_tools(
            ChatRequest(
                message="kalau 19 hari lalu?",
                actor_role=ActorRole.admin,
                history=[
                    {
                        "role": "assistant",
                        "content": "Ada 1 EOI terdaftar di admission-service.",
                        "toolCalls": [
                            {"name": "admission.admin_leads_count", "status": "ok"}
                        ],
                    }
                ],
            ),
            Settings(),
            "Bearer test-token",
        )
    )

    assert result is not None
    assert "3 EOI" in result.answer
    assert "19 hari lalu (30 Mei 2026)" in result.answer


def test_owner_contextual_last_month_eoi_count_uses_date_filter(monkeypatch) -> None:
    monkeypatch.setattr(
        service_tools,
        "_now_jakarta",
        lambda: datetime(2026, 6, 18, 12, 0, tzinfo=service_tools.SCHOOL_TIME_ZONE),
    )

    async def fake_get_json(url, authorization, params):
        assert url == "http://admission-service/api/leads/v1/admin/leads"
        assert authorization == "Bearer test-token"
        assert params["limit"] == "1"
        assert params["offset"] == "0"
        assert params["dateFrom"] == "2026-04-30T17:00:00+00:00"
        assert params["dateTo"] == "2026-05-31T16:59:59.999000+00:00"
        return {"data": {"total": 4}}

    monkeypatch.setattr(service_tools, "_get_json", fake_get_json)
    result = asyncio.run(
        service_tools.answer_from_school_tools(
            ChatRequest(
                message="kalau bulan lalu?",
                actor_role=ActorRole.admin,
                history=[
                    {
                        "role": "assistant",
                        "content": "Ada 1 EOI terdaftar di admission-service.",
                        "toolCalls": [
                            {"name": "admission.admin_leads_count", "status": "ok"}
                        ],
                    }
                ],
            ),
            Settings(),
            "Bearer test-token",
        )
    )

    assert result is not None
    assert "4 EOI" in result.answer
    assert "bulan lalu (Mei 2026)" in result.answer


def test_owner_specific_date_leads_list_uses_date_filter(monkeypatch) -> None:
    monkeypatch.setattr(
        service_tools,
        "_now_jakarta",
        lambda: datetime(2026, 6, 18, 12, 0, tzinfo=service_tools.SCHOOL_TIME_ZONE),
    )

    async def fake_get_json(url, authorization, params):
        assert url == "http://admission-service/api/leads/v1/admin/leads"
        assert authorization == "Bearer test-token"
        assert params["limit"] == "5"
        assert params["offset"] == "0"
        assert params["dateFrom"] == "2026-06-16T17:00:00+00:00"
        assert params["dateTo"] == "2026-06-17T16:59:59.999000+00:00"
        return {
            "data": {
                "total": 1,
                "rows": [
                    {
                        "parentName": "Arief Nugraha",
                        "email": "arief@example.test",
                        "school": "SCH-IISS",
                        "leadStatus": "verified",
                    }
                ],
            }
        }

    monkeypatch.setattr(service_tools, "_get_json", fake_get_json)
    result = asyncio.run(
        service_tools.answer_from_school_tools(
            ChatRequest(
                message="siapa aja yang daftar 17 Juni 2026?",
                actor_role=ActorRole.admin,
            ),
            Settings(),
            "Bearer test-token",
        )
    )

    assert result is not None
    assert "1 EOI 17 Juni 2026" in result.answer
    assert "Arief Nugraha" in result.answer
    assert result.tool_calls[0].name == "admission.admin_leads_list"


def test_owner_named_lead_child_count_uses_admission_tool(monkeypatch) -> None:
    async def fake_get_json(url, authorization, params):
        assert url == "http://admission-service/api/leads/v1/admin/leads"
        assert authorization == "Bearer test-token"
        assert params == {"limit": "5", "offset": "0", "search": "arief nugraha"}
        return {
            "data": {
                "total": 1,
                "rows": [
                    {
                        "leadId": "LEAD-1",
                        "parentName": "Arief Nugraha",
                        "email": "arief@example.test",
                        "school": "SCH-IISS",
                        "leadStatus": "verified",
                        "applicantCount": 2,
                    }
                ],
            }
        }

    monkeypatch.setattr(service_tools, "_get_json", fake_get_json)
    result = asyncio.run(
        service_tools.answer_from_school_tools(
            ChatRequest(
                message="Arief Nugraha di EOI dia daftarin berapa anak",
                actor_role=ActorRole.admin,
            ),
            Settings(),
            "Bearer test-token",
        )
    )

    assert result is not None
    assert "Arief Nugraha mendaftarkan 2 anak" in result.answer
    assert result.tool_calls[0].name == "admission.admin_lead_child_count"
    assert result.tool_calls[0].status == "ok"


def test_owner_contextual_child_identity_followup_uses_detail_tool(monkeypatch) -> None:
    async def fake_get_json(url, authorization, params):
        assert authorization == "Bearer test-token"
        if url == "http://admission-service/api/leads/v1/admin/leads":
            assert params == {"limit": "5", "offset": "0", "search": "Arief Nugraha"}
            return {
                "data": {
                    "total": 1,
                    "rows": [
                        {
                            "leadId": "LEAD-1",
                            "parentName": "Arief Nugraha",
                            "email": "arief@example.test",
                            "school": "SCH-IISS",
                            "leadStatus": "verified",
                            "applicantCount": 2,
                        }
                    ],
                }
            }
        assert url == "http://admission-service/api/leads/v1/admin/leads/LEAD-1"
        assert params == {}
        return {
            "data": {
                "detail": {},
                "notes": [],
                "students": [
                    {
                        "fullName": "Aisha Nugraha",
                        "targetGradeLevel": "Grade 7",
                        "targetSchool": "SCH-IISS",
                        "applicantStatus": "submitted",
                    },
                    {
                        "fullName": "Bilal Nugraha",
                        "targetGradeLevel": "Grade 8",
                        "targetSchool": "SCH-IISS",
                        "applicantStatus": "submitted",
                    },
                ],
            }
        }

    monkeypatch.setattr(service_tools, "_get_json", fake_get_json)
    result = asyncio.run(
        service_tools.answer_from_school_tools(
            ChatRequest(
                message="siapa",
                actor_role=ActorRole.admin,
                history=[
                    {
                        "role": "assistant",
                        "content": "Arief Nugraha mendaftarkan 2 anak di aplikasi.",
                        "toolCalls": [
                            {"name": "admission.admin_lead_child_count", "status": "ok"}
                        ],
                    }
                ],
            ),
            Settings(),
            "Bearer test-token",
        )
    )

    assert result is not None
    assert "Aisha Nugraha" in result.answer
    assert "Bilal Nugraha" in result.answer
    assert result.tool_calls[0].name == "admission.admin_lead_students_list"
    assert result.tool_calls[0].status == "ok"


def test_owner_english_eoi_count_uses_admission_tool(monkeypatch) -> None:
    async def fake_get_json(url, authorization, params):
        assert url == "http://admission-service/api/leads/v1/admin/leads"
        assert authorization == "Bearer test-token"
        assert params == {"limit": "1", "offset": "0"}
        return {"data": {"total": 11}}

    monkeypatch.setattr(service_tools, "_get_json", fake_get_json)
    result = asyncio.run(
        service_tools.answer_from_school_tools(
            ChatRequest(
                message="How many EOIs are registered?",
                actor_role=ActorRole.admin,
            ),
            Settings(),
            "Bearer test-token",
        )
    )

    assert result is not None
    assert "11 EOI" in result.answer
    assert result.tool_calls[0].name == "admission.admin_leads_count"
    assert result.tool_calls[0].status == "ok"


def test_owner_paraphrased_registration_count_uses_llm_classifier(monkeypatch) -> None:
    class FakeLlmClient:
        def __init__(self, settings):
            assert settings.ai_provider_base_url is not None

        async def complete(
            self,
            system_prompt,
            user_message,
            temperature=0.2,
            max_tokens=None,
            timeout_seconds=None,
            response_format=None,
        ):
            assert "Return JSON only" in system_prompt
            assert user_message == "berapa calon keluarga masuk sejauh ini?"
            assert temperature == 0.0
            assert max_tokens == 32
            assert timeout_seconds == 60.0
            assert response_format == {"type": "json_object"}
            return '{"intent":"admission_eoi_count"}'

    async def fake_get_json(url, authorization, params):
        assert url == "http://admission-service/api/leads/v1/admin/leads"
        assert authorization == "Bearer test-token"
        assert params == {"limit": "1", "offset": "0"}
        return {"data": {"total": 12}}

    monkeypatch.setattr(service_tools, "LlmClient", FakeLlmClient)
    monkeypatch.setattr(service_tools, "_get_json", fake_get_json)
    result = asyncio.run(
        service_tools.answer_from_school_tools(
            ChatRequest(
                message="berapa calon keluarga masuk sejauh ini?",
                actor_role=ActorRole.owner,
            ),
            Settings(ai_provider_base_url="http://localhost:11434/v1"),
            "Bearer test-token",
        )
    )

    assert result is not None
    assert "12 EOI" in result.answer
    assert result.tool_calls[0].name == "admission.admin_leads_count"
    assert result.tool_calls[0].status == "ok"


def test_owner_payment_count_uses_payment_tool(monkeypatch) -> None:
    async def fake_get_json(url, authorization, params):
        assert url == "http://payment-service/api/v1/payments/admin/reviews"
        assert authorization == "Bearer test-token"
        assert params == {"status": "pending_verification", "limit": "1", "offset": "0"}
        return {"data": {"total": 3}}

    monkeypatch.setattr(service_tools, "_get_json", fake_get_json)
    result = asyncio.run(
        service_tools.answer_from_school_tools(
            ChatRequest(
                message="ada berapa pembayaran pending?",
                actor_role=ActorRole.owner,
            ),
            Settings(),
            "Bearer test-token",
        )
    )

    assert result is not None
    assert "3 pembayaran" in result.answer
    assert result.tool_calls[0].name == "payment.admin_review_count"
    assert result.tool_calls[0].status == "ok"


def test_owner_arrived_payment_prompt_uses_payment_tool(monkeypatch) -> None:
    async def fake_get_json(url, authorization, params):
        assert url == "http://payment-service/api/v1/payments/admin/reviews"
        assert authorization == "Bearer test-token"
        assert params == {"status": "pending_verification", "limit": "1", "offset": "0"}
        return {"data": {"total": 5}}

    monkeypatch.setattr(service_tools, "_get_json", fake_get_json)
    result = asyncio.run(
        service_tools.answer_from_school_tools(
            ChatRequest(
                message="udah ada yang sampai payment?",
                actor_role=ActorRole.owner,
            ),
            Settings(),
            "Bearer test-token",
        )
    )

    assert result is not None
    assert "5 pembayaran" in result.answer
    assert result.tool_calls[0].name == "payment.admin_review_count"
    assert result.tool_calls[0].status == "ok"


def test_owner_underpaid_payment_count_uses_underpaid_status(monkeypatch) -> None:
    async def fake_get_json(url, authorization, params):
        assert url == "http://payment-service/api/v1/payments/admin/reviews"
        assert authorization == "Bearer test-token"
        assert params == {"status": "underpaid", "limit": "1", "offset": "0"}
        return {"data": {"total": 2}}

    monkeypatch.setattr(service_tools, "_get_json", fake_get_json)
    result = asyncio.run(
        service_tools.answer_from_school_tools(
            ChatRequest(
                message="ada berapa pembayaran underpaid?",
                actor_role=ActorRole.owner,
            ),
            Settings(),
            "Bearer test-token",
        )
    )

    assert result is not None
    assert "2 pembayaran" in result.answer
    assert "kurang bayar" in result.answer
    assert result.tool_calls[0].name == "payment.admin_review_count"
    assert result.tool_calls[0].status == "ok"


def test_owner_admission_price_uses_payment_fee_tool(monkeypatch) -> None:
    calls = []

    async def fake_get_json(url, authorization, params):
        calls.append((url, params))
        assert authorization == "Bearer test-token"
        assert params == {"payment_type": "application_fee"}
        school_code = url.rsplit("/", 1)[-1]
        return {
            "data": {
                "feeStructureId": f"FEE-{school_code}",
                "schoolCode": school_code,
                "paymentType": "application_fee",
                "amount": 2200000,
                "currency": "IDR",
                "status": "active",
            }
        }

    monkeypatch.setattr(service_tools, "_get_json", fake_get_json)
    result = asyncio.run(
        service_tools.answer_from_school_tools(
            ChatRequest(
                message="berapa harga untuk admissions?",
                actor_role=ActorRole.admin,
            ),
            Settings(),
            "Bearer test-token",
        )
    )

    assert result is not None
    assert "Biaya pendaftaran" in result.answer
    assert "IIHS: Rp 2.200.000" in result.answer
    assert "IISS: Rp 2.200.000" in result.answer
    assert "IIBS: Rp 2.200.000" in result.answer
    assert [url.rsplit("/", 1)[-1] for url, _params in calls] == ["IIHS", "IISS", "IIBS"]
    assert result.tool_calls[0].name == "payment.application_fee_quote"
    assert result.tool_calls[0].status == "ok"


def test_owner_school_specific_admission_price_uses_payment_fee_tool(monkeypatch) -> None:
    async def fake_get_json(url, authorization, params):
        assert url == "http://payment-service/api/v1/payments/fees/IISS"
        assert authorization == "Bearer test-token"
        assert params == {"payment_type": "application_fee"}
        return {
            "data": {
                "feeStructureId": "FEE-IISS",
                "schoolCode": "IISS",
                "paymentType": "application_fee",
                "amount": 2200000,
                "currency": "IDR",
                "status": "active",
            }
        }

    monkeypatch.setattr(service_tools, "_get_json", fake_get_json)
    result = asyncio.run(
        service_tools.answer_from_school_tools(
            ChatRequest(
                message="berapa biaya pendaftaran iiss?",
                actor_role=ActorRole.admin,
            ),
            Settings(),
            "Bearer test-token",
        )
    )

    assert result is not None
    assert "IISS: Rp 2.200.000" in result.answer
    assert "IIHS" not in result.answer
    assert result.tool_calls[0].name == "payment.application_fee_quote"


def test_owner_payment_count_reports_forbidden_admin_session(monkeypatch) -> None:
    async def fake_get_json(url, authorization, params):
        request = httpx.Request("GET", url)
        response = httpx.Response(403, request=request)
        raise httpx.HTTPStatusError("forbidden", request=request, response=response)

    monkeypatch.setattr(service_tools, "_get_json", fake_get_json)
    result = asyncio.run(
        service_tools.answer_from_school_tools(
            ChatRequest(
                message="ada berapa pembayaran pending?",
                actor_role=ActorRole.owner,
            ),
            Settings(),
            "Bearer non-admin-token",
        )
    )

    assert result is not None
    assert "akses admin" in result.answer
    assert result.tool_calls[0].name == "payment.admin_review_count"
    assert result.tool_calls[0].status == "forbidden"


def test_owner_paraphrased_payment_count_uses_llm_classifier(monkeypatch) -> None:
    class FakeLlmClient:
        def __init__(self, settings):
            assert settings.ai_provider_base_url is not None

        async def complete(
            self,
            system_prompt,
            user_message,
            temperature=0.2,
            max_tokens=None,
            timeout_seconds=None,
            response_format=None,
        ):
            assert "paymentStatus" in system_prompt
            assert user_message == "berapa transaksi yang masih perlu dicek finance?"
            assert temperature == 0.0
            assert max_tokens == 32
            assert timeout_seconds == 60.0
            assert response_format == {"type": "json_object"}
            return '{"intent":"payment_review_count","paymentStatus":"waiting_verification"}'

    async def fake_get_json(url, authorization, params):
        assert url == "http://payment-service/api/v1/payments/admin/reviews"
        assert authorization == "Bearer test-token"
        assert params == {"status": "pending_verification", "limit": "1", "offset": "0"}
        return {"data": {"total": 8}}

    monkeypatch.setattr(service_tools, "LlmClient", FakeLlmClient)
    monkeypatch.setattr(service_tools, "_get_json", fake_get_json)
    result = asyncio.run(
        service_tools.answer_from_school_tools(
            ChatRequest(
                message="berapa transaksi yang masih perlu dicek finance?",
                actor_role=ActorRole.admin,
            ),
            Settings(ai_provider_base_url="http://localhost:11434/v1"),
            "Bearer test-token",
        )
    )

    assert result is not None
    assert "8 pembayaran" in result.answer
    assert result.tool_calls[0].name == "payment.admin_review_count"
    assert result.tool_calls[0].status == "ok"


def test_owner_english_payment_count_uses_payment_tool(monkeypatch) -> None:
    async def fake_get_json(url, authorization, params):
        assert url == "http://payment-service/api/v1/payments/admin/reviews"
        assert authorization == "Bearer test-token"
        assert params == {"status": "pending_verification", "limit": "1", "offset": "0"}
        return {"data": {"total": 4}}

    monkeypatch.setattr(service_tools, "_get_json", fake_get_json)
    result = asyncio.run(
        service_tools.answer_from_school_tools(
            ChatRequest(
                message="How many payments are pending verification?",
                actor_role=ActorRole.admin,
            ),
            Settings(),
            "Bearer test-token",
        )
    )

    assert result is not None
    assert "4 pembayaran" in result.answer
    assert result.tool_calls[0].name == "payment.admin_review_count"
    assert result.tool_calls[0].status == "ok"


def test_tool_intent_parser_accepts_fenced_json() -> None:
    intent = service_tools._parse_tool_intent(
        '```json\n{"intent":"payment.admin_review_count","status":"menunggu verifikasi"}\n```'
    )

    assert intent.name == service_tools.INTENT_PAYMENT_REVIEW_COUNT
    assert intent.payment_status == "pending_verification"


def test_tool_intent_parser_accepts_leads_list_alias() -> None:
    intent = service_tools._parse_tool_intent('{"intent":"admission_admin_leads_list"}')

    assert intent.name == service_tools.INTENT_ADMISSION_LEADS_LIST


def test_tool_intent_parser_accepts_fee_quote_alias() -> None:
    intent = service_tools._parse_tool_intent('{"intent":"admission_price_quote"}')

    assert intent.name == service_tools.INTENT_PAYMENT_APPLICATION_FEE

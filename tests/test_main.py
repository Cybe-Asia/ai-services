import asyncio

from fastapi.testclient import TestClient

from app import service_tools
from app.config import Settings
from app.main import app
from app.schemas import ActorRole, ChatRequest

client = TestClient(app)


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


def test_public_payment_prompt_is_refused() -> None:
    response = client.post(
        "/api/ai/v1/chat",
        json={"message": "ada berapa pembayaran pending?", "actorRole": "public"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "refused"


def test_admin_prompt_requires_bearer_token() -> None:
    response = client.post(
        "/api/ai/v1/chat",
        json={"message": "siapa best student by nilai?", "actorRole": "admin"},
    )
    assert response.status_code == 401


def test_admin_prompt_with_bearer_returns_pending_tool() -> None:
    response = client.post(
        "/api/ai/v1/chat",
        headers={"authorization": "Bearer test-token"},
        json={"message": "siapa best student by nilai?", "actorRole": "admin"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["toolCalls"][0]["status"] == "not_configured"


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

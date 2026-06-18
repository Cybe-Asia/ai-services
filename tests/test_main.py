import asyncio

import httpx
from fastapi.testclient import TestClient

from app import main as main_module
from app import service_tools
from app.config import Settings, get_settings
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


def test_admin_sensitive_prompt_without_tool_does_not_call_draft_llm(monkeypatch) -> None:
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
    assert "tool data resmi" in body["answer"]
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

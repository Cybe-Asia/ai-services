from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/api/v1/ai-service/health")
    assert response.status_code == 200
    assert response.json()["service"] == "ai-service"


def test_metadata_reports_generation_limit() -> None:
    response = client.get("/api/v1/ai-service/metadata")
    assert response.status_code == 200
    assert response.json()["maxTokens"] == 256


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

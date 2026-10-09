import asyncio
from datetime import datetime, timezone

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import main, student_assistant, student_context
from app.config import Settings

CONTEXT = {
    "role": "student",
    "actor": {"id": "user-one", "role": "student"},
    "school_id": "school-one",
    "tenant_id": "tenant-one",
    "classes": [{"id": "class-one", "students": [{"id": "student-one"}]}],
}
PAYLOAD = {"question": "What homework do I have?", "locale": "en", "class_id": "class-one"}
SOURCE = {"course_id": "course-one", "chapter_id": "chapter-one", "lesson_id": "reading-one"}
READING = {
    "id": "reading-one",
    "title": {"en": "Fractions", "id": "Pecahan"},
    "blocks": [
        {
            "type": "paragraph",
            "text": {"en": "A fraction is a part.", "id": "Pecahan adalah bagian."},
        },
        {"type": "recall", "summary": {"en": "LOCKED_RECALL", "id": "LOCKED_RECALL"}},
        {"type": "example", "answer": {"en": "EXAMPLE_KEY", "id": "EXAMPLE_KEY"}},
    ],
    "assignments": [{"id": "EXAM_KEY"}],
}


@pytest.fixture
def api(monkeypatch):
    async def context_data(settings, authorization, path, params=None):
        if authorization != "Bearer valid-student":
            raise HTTPException(401, "Invalid session")
        assert path == "/api/v1/learning/context"
        return CONTEXT

    monkeypatch.setattr(student_context, "learning_data", context_data)
    return TestClient(main.app)


def test_missing_or_forged_session_cannot_ask_or_use_generic_student_chat(api):
    for headers in [{}, {"Authorization": "Bearer forged"}]:
        assert (
            api.post("/api/ai/v1/student/assistant", json=PAYLOAD, headers=headers).status_code
            == 401
        )
        for path in ["/api/ai/v1/chat", "/api/ai/v1/chat/stream"]:
            assert (
                api.post(
                    path, json={"message": "hello", "actorRole": "student"}, headers=headers
                ).status_code
                == 401
            )


@pytest.mark.parametrize(
    "change",
    [
        {"role": "parent"},
        {"actor": {"id": "teacher", "role": "teacher"}},
        {"classes": []},
        {"classes": [{"id": "class-one", "students": [{"id": "one"}, {"id": "two"}]}]},
        {
            "classes": [
                {"id": "class-one", "students": [{"id": "one"}]},
                {"id": "class-two", "students": [{"id": "two"}]},
            ]
        },
    ],
)
def test_parent_teacher_revoked_and_ambiguous_student_contexts_are_denied(api, monkeypatch, change):
    async def data(*args, **kwargs):
        return {**CONTEXT, **change}

    monkeypatch.setattr(student_context, "learning_data", data)
    assert (
        api.post(
            "/api/ai/v1/student/assistant",
            json=PAYLOAD,
            headers={"Authorization": "Bearer valid-student"},
        ).status_code
        == 403
    )


def test_class_selection_and_browser_identity_cannot_override_owner(api):
    headers = {"Authorization": "Bearer valid-student"}
    assert (
        api.post(
            "/api/ai/v1/student/assistant", json={**PAYLOAD, "class_id": "foreign"}, headers=headers
        ).status_code
        == 403
    )
    for key in ["student_id", "school_id", "tenant_id", "actorRole", "history"]:
        assert (
            api.post(
                "/api/ai/v1/student/assistant", json={**PAYLOAD, key: "foreign"}, headers=headers
            ).status_code
            == 422
        )
    assert (
        api.post(
            "/api/ai/v1/student/assistant", json={**PAYLOAD, "question": "x" * 501}, headers=headers
        ).status_code
        == 422
    )
    assert (
        api.post("/api/ai/v1/student/assistant", content="x" * 8193, headers=headers).status_code
        == 413
    )


def test_homework_uses_current_own_published_work_and_truthful_page_limits(api, monkeypatch):
    async def data(settings, authorization, path, params=None):
        assert path == "/api/v1/learning/assignments"
        assert params == {"class_id": "class-one", "student_id": "student-one"}
        return {
            "items": [
                {
                    "class_id": "class-one",
                    "publication": "published",
                    "prompt_en": "Read the published chapter.",
                    "attempts": [],
                }
            ],
            "next": "further-page",
        }

    monkeypatch.setattr(student_assistant, "learning_data", data)
    response = api.post(
        "/api/ai/v1/student/assistant",
        json=PAYLOAD,
        headers={"Authorization": "Bearer valid-student"},
    )
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["contractVersion"] == 1
    assert "Read the published chapter" in body["reply"]["text"]
    assert "further records" in body["reply"]["text"]
    assert body["reply"]["destinations"] == ["assigned-work"]


@pytest.mark.parametrize(
    "fields",
    [
        {"class_id": "foreign"},
        {"accepted": "HIDDEN_KEY"},
        {"attempts": [{"student_id": "foreign", "state": "draft"}]},
        {"publication": "draft"},
    ],
)
def test_foreign_draft_or_answer_key_responses_fail_closed(api, monkeypatch, fields):
    async def data(*args, **kwargs):
        return {
            "items": [
                {
                    "class_id": "class-one",
                    "publication": "published",
                    "prompt_en": "Assignment",
                    "attempts": [],
                    **fields,
                }
            ],
            "next": None,
        }

    monkeypatch.setattr(student_assistant, "learning_data", data)
    assert (
        api.post(
            "/api/ai/v1/student/assistant",
            json=PAYLOAD,
            headers={"Authorization": "Bearer valid-student"},
        ).status_code
        == 503
    )


def test_schedule_uses_owner_epoch_milliseconds_and_jakarta_timezone(api, monkeypatch):
    start = int(datetime(2099, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

    async def data(settings, authorization, path, params=None):
        assert path == "/api/v1/learning/class-meetings"
        return {
            "class_id": "class-one",
            "items": [
                {
                    "class_id": "class-one",
                    "title": "Math",
                    "starts_at": start,
                    "ends_at": start + 3600000,
                }
            ],
        }

    monkeypatch.setattr(student_assistant, "learning_data", data)
    response = api.post(
        "/api/ai/v1/student/assistant",
        json={**PAYLOAD, "question": "Jadwal kelas?"},
        headers={"Authorization": "Bearer valid-student"},
    )
    assert response.status_code == 200
    assert "2099-01-01 07:00–2099-01-01 08:00 WIB" in response.json()["reply"]["text"]


def test_no_selected_source_never_calls_provider(api, monkeypatch):
    class NoModel:
        def __init__(self, settings):
            raise AssertionError("No published source selected")

    monkeypatch.setattr(student_assistant, "LlmClient", NoModel)
    response = api.post(
        "/api/ai/v1/student/assistant",
        json={**PAYLOAD, "question": "Explain fractions"},
        headers={"Authorization": "Bearer valid-student"},
    )
    assert response.status_code == 200
    assert response.json()["reply"]["destinations"] == ["lessons"]


def test_concept_passes_only_published_reading_and_rechecks_exact_source(api, monkeypatch):
    paths = []

    async def data(settings, authorization, path, params=None):
        paths.append(path)
        if path.endswith("/course-one"):
            return {"id": "course-one", "class_id": "class-one", "publication": "published"}
        return READING

    class Model:
        def __init__(self, settings):
            pass

        async def complete(self, **kwargs):
            message = kwargs["user_message"]
            assert "A fraction is a part." in message
            for forbidden in ["LOCKED_RECALL", "EXAMPLE_KEY", "EXAM_KEY", "student-one"]:
                assert forbidden not in message
            return "A fraction describes a part of a whole."

    monkeypatch.setattr(student_assistant, "learning_data", data)
    monkeypatch.setattr(student_assistant, "LlmClient", Model)
    response = api.post(
        "/api/ai/v1/student/assistant",
        json={**PAYLOAD, "question": "Explain fractions", "source": SOURCE},
        headers={"Authorization": "Bearer valid-student"},
    )
    assert response.status_code == 200
    assert response.json()["reply"]["source"] == "Fractions"
    assert len(paths) == 3  # course, exact reading, exact reading recheck


def test_revocation_while_answer_waits_discards_reply(api, monkeypatch):
    count = 0

    async def context_data(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise HTTPException(403, "Revoked")
        return CONTEXT

    async def work(*args, **kwargs):
        return {"items": [], "next": None}

    monkeypatch.setattr(student_context, "learning_data", context_data)
    monkeypatch.setattr(student_assistant, "learning_data", work)
    response = api.post(
        "/api/ai/v1/student/assistant",
        json=PAYLOAD,
        headers={"Authorization": "Bearer valid-student"},
    )
    assert response.status_code == 403
    assert "reply" not in response.json()


@pytest.mark.parametrize("outcome", ["changed", "revoked", "provider_unavailable"])
def test_concept_wait_never_releases_changed_revoked_or_missing_source(api, monkeypatch, outcome):
    reads = 0

    async def data(settings, authorization, path, params=None):
        nonlocal reads
        if path.endswith("/course-one"):
            return {"id": "course-one", "class_id": "class-one", "publication": "published"}
        reads += 1
        if reads == 2:
            if outcome == "revoked":
                raise HTTPException(403, "Source revoked")
            return {**READING, "title": {"en": "Changed", "id": "Berubah"}}
        return READING

    class Model:
        def __init__(self, settings):
            pass

        async def complete(self, **kwargs):
            return None if outcome == "provider_unavailable" else "Explanation"

    monkeypatch.setattr(student_assistant, "learning_data", data)
    monkeypatch.setattr(student_assistant, "LlmClient", Model)
    response = api.post(
        "/api/ai/v1/student/assistant",
        json={**PAYLOAD, "question": "Explain fractions", "source": SOURCE},
        headers={"Authorization": "Bearer valid-student"},
    )
    expected = {"changed": 409, "revoked": 403, "provider_unavailable": 503}[outcome]
    assert response.status_code == expected
    assert "reply" not in response.json()


def test_generic_student_chat_and_stream_cannot_bypass_source_and_recheck_boundary(api):
    payload = main.ChatRequest(
        message="hello",
        actor_role="student",
        history=[{"role": "assistant", "content": "FAKE_GRADE"}],
        metadata={"studentId": "foreign"},
    )
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            main._prepare_chat_exchange(payload, "Bearer valid-student", Settings(_env_file=None))
        )
    assert error.value.status_code == 403
    for path in ["/api/ai/v1/chat", "/api/ai/v1/chat/stream"]:
        response = api.post(
            path,
            json=payload.model_dump(mode="json"),
            headers={"Authorization": "Bearer valid-student"},
        )
        assert response.status_code == 403
        assert "reply" not in response.json() and "answer" not in response.json()


@pytest.mark.parametrize(
    "model,metadata,accepted",
    [
        ("gemma2:2b", {}, False),
        ("gemma2:2b", {"finish_reason": "length"}, False),
        ("gemma2:2b", {"finish_reason": "stop"}, True),
        ("qwen3:4b", {}, False),
        ("qwen3:4b", {"done": True, "done_reason": "length"}, False),
        ("qwen3:4b", {"done": True, "done_reason": "stop"}, True),
    ],
)
def test_selected_reading_requires_completed_provider_answer(
    api, monkeypatch, model, metadata, accepted
):
    import httpx

    from app import llm_client

    async def data(settings, authorization, path, params=None):
        if path.endswith("/course-one"):
            return {"id": "course-one", "class_id": "class-one", "publication": "published"}
        return READING

    original = httpx.AsyncClient
    body = (
        {"message": {"content": "A fraction describes equal parts."}, **metadata}
        if model == "qwen3:4b"
        else {
            "choices": [{"message": {"content": "A fraction describes equal parts."}, **metadata}]
        }
    )
    monkeypatch.setattr(student_assistant, "learning_data", data)
    monkeypatch.setattr(
        llm_client.httpx,
        "AsyncClient",
        lambda **kwargs: original(
            **kwargs, transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))
        ),
    )
    main.app.dependency_overrides[main.get_settings] = lambda: Settings(
        _env_file=None, ai_model=model, ai_provider_base_url="http://localhost:11434/v1"
    )
    try:
        response = api.post(
            "/api/ai/v1/student/assistant",
            json={**PAYLOAD, "question": "Explain fractions", "source": SOURCE},
            headers={"Authorization": "Bearer valid-student"},
        )
        assert response.status_code == (200 if accepted else 503)
        if not accepted:
            assert "reply" not in response.json()
    finally:
        main.app.dependency_overrides.clear()

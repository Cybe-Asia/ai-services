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
ANSWERED = '{"answered": true, "text": "A fraction describes equal parts."}'
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
    assert body["reply"]["table"] == {
        "columns": ["Assignment", "Points", "Status"],
        "rows": [["Read the published chapter.", "–", "Not started"]],
    }
    assert "1 assignment(s) not submitted" in body["reply"]["text"]
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
    assert response.json()["reply"]["table"]["rows"] == [["Thursday, 1 Jan", "07:00–08:00", "Math"]]


def test_schedule_answers_the_asked_day_grouped_with_school_subject_names():
    from app.student_assistant import schedule_text

    now = datetime(2026, 10, 26, 6, 0, tzinfo=student_assistant.JAKARTA)  # Monday

    def at(day: int, hour: int, minute: int = 0) -> int:
        moment = datetime(2026, 10, day, hour, minute, tzinfo=student_assistant.JAKARTA)
        return int(moment.timestamp() * 1000)

    meetings = [
        (at(26, 7, 30), at(26, 8, 15), "Mathematics · Year 7A"),
        (at(26, 8, 15), at(26, 9), "Science · Year 7A"),
        (at(27, 7, 30), at(27, 8, 15), "English · Year 7A"),
    ]
    text, table = schedule_text(meetings, "Apa jadwal pelajaranku hari Senin?", "id", now)
    assert text == "Jadwalmu Senin, 26 Okt:"
    assert table == {
        "columns": ["Hari", "Jam", "Mapel"],
        "rows": [
            ["Senin, 26 Okt", "07.30–08.15", "Matematika"],
            ["Senin, 26 Okt", "08.15–09.00", "IPA"],
        ],
    }
    _, tomorrow = schedule_text(meetings, "What is on tomorrow?", "en", now)
    assert tomorrow["rows"] == [["Tuesday, 27 Oct", "07:30–08:15", "English"]]
    saturday = schedule_text(meetings, "Jadwal hari Sabtu?", "id", now)
    assert saturday == ("Tidak ada kelas pada Sabtu, 31 Okt.", None)
    # No day named: the next upcoming meetings across days.
    assert len(schedule_text(meetings, "Jadwalku?", "id", now)[1]["rows"]) == 3


def test_no_matching_reading_never_calls_provider(api, monkeypatch):
    class NoModel:
        def __init__(self, settings):
            raise AssertionError("No published source matched")

    async def data(settings, authorization, path, params=None):
        if path == "/api/v1/learning/curriculum/courses":
            return {"items": [], "next": None}
        raise AssertionError(path)

    monkeypatch.setattr(student_assistant, "LlmClient", NoModel)
    monkeypatch.setattr(student_assistant, "learning_data", data)
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
            return '{"answered": true, "text": "A fraction describes a part of a whole."}'

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
            return None if outcome == "provider_unavailable" else ANSWERED

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
        {"message": {"content": ANSWERED}, **metadata}
        if model == "qwen3:4b"
        else {"choices": [{"message": {"content": ANSWERED}, **metadata}]}
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


def _reading_data():
    async def data(settings, authorization, path, params=None):
        if path.endswith("/course-one"):
            return {"id": "course-one", "class_id": "class-one", "publication": "published"}
        return READING

    return data


@pytest.mark.parametrize(
    "answer,expected_status,expected_text",
    [
        ('{"answered": false, "text": ""}', 200, "does not cover that question"),
        ('{"answered": false}', 200, "does not cover that question"),
        ('{"answered": true, "text": ""}', 503, None),
        ('{"answered": true, "text": "x", "extra": 1}', 503, None),
        ("A plain explanation without the contract.", 503, None),
        ('{"answered": "yes", "text": "x"}', 503, None),
    ],
)
def test_reading_answer_contract_gates_off_topic_and_malformed_answers(
    api, monkeypatch, answer, expected_status, expected_text
):
    class Model:
        def __init__(self, settings):
            pass

        async def complete(self, **kwargs):
            assert kwargs["response_format"] == {"type": "json_object"}
            assert kwargs["feature"] == "assistant"
            assert '"answered"' in kwargs["system_prompt"]
            return answer

    monkeypatch.setattr(student_assistant, "learning_data", _reading_data())
    monkeypatch.setattr(student_assistant, "LlmClient", Model)
    response = api.post(
        "/api/ai/v1/student/assistant",
        json={**PAYLOAD, "question": "Give me a rendang recipe", "source": SOURCE},
        headers={"Authorization": "Bearer valid-student"},
    )
    assert response.status_code == expected_status
    if expected_text:
        body = response.json()["reply"]
        assert expected_text in body["text"]
        assert body["source"] == "Fractions"
        assert body["destinations"] == ["lessons"]
    else:
        assert "reply" not in response.json()


def test_reading_explanation_uses_the_assistant_provider_block(api, monkeypatch):
    seen = []

    class Model:
        def __init__(self, settings):
            seen.append((settings.ai_model, settings.ai_provider_protocol, settings.ai_max_tokens))

        async def complete(self, **kwargs):
            return '{"answered": true, "text": "Parts of a whole."}'

    monkeypatch.setattr(student_assistant, "learning_data", _reading_data())
    monkeypatch.setattr(student_assistant, "LlmClient", Model)
    main.app.dependency_overrides[main.get_settings] = lambda: Settings(
        _env_file=None,
        ai_model="qwen3:4b",
        ai_provider_base_url="http://localhost:11434/v1",
        assistant_provider_base_url="https://api.openai.example/v1",
        assistant_provider_api_key="assistant-key",
        assistant_model="gpt-6-luna",
        assistant_max_tokens=400,
    )
    try:
        response = api.post(
            "/api/ai/v1/student/assistant",
            json={**PAYLOAD, "question": "Explain fractions", "source": SOURCE},
            headers={"Authorization": "Bearer valid-student"},
        )
    finally:
        main.app.dependency_overrides.clear()
    assert response.status_code == 200
    # The planner and the reading explanation both use the assistant block.
    assert seen and set(seen) == {("gpt-6-luna", "openai", 400)}


def ask(api, question, locale="id"):
    return api.post(
        "/api/ai/v1/student/assistant",
        json={**PAYLOAD, "question": question, "locale": locale},
        headers={"Authorization": "Bearer valid-student"},
    )


def test_homework_lists_unsubmitted_work_with_draft_state(api, monkeypatch):
    async def data(settings, authorization, path, params=None):
        item = {"class_id": "class-one", "publication": "published", "points": 20}
        return {
            "items": [
                {**item, "prompt_id": "Laporan praktikum", "attempts": []},
                {
                    **item,
                    "prompt_id": "Esai pecahan",
                    "attempts": [{"student_id": "student-one", "state": "draft"}],
                },
                {
                    **item,
                    "prompt_id": "Sudah dikirim",
                    "attempts": [{"student_id": "student-one", "state": "submitted"}],
                },
            ],
            "next": None,
        }

    monkeypatch.setattr(student_assistant, "learning_data", data)
    body = ask(api, "ada assignment yang belum saya kerjakan?").json()["reply"]
    assert body["text"] == "Ada 2 tugas yang belum kamu kirim:"
    assert body["table"]["rows"] == [
        ["Laporan praktikum", "20", "Belum dikerjakan"],
        ["Esai pecahan", "20", "Draf, belum dikirim"],
    ]


def test_games_lists_completed_games_with_module_and_chapter_titles(api, monkeypatch):
    at = int(datetime(2026, 11, 3, 3, tzinfo=timezone.utc).timestamp() * 1000)
    award = {"assessment_id": "m", "session_id": "s", "challenge_id": "m", "at": at}
    paths = []

    async def data(settings, authorization, path, params=None):
        paths.append(path)
        if path == "/api/v1/learning/game-progress":
            return {
                "awards": {
                    "practice:course-one|lesson-a|m": {
                        **award,
                        "course_id": "course-one",
                        "lesson_id": "lesson-a",
                        "checkpoint_id": None,
                        "xp": 30,
                    },
                    "checkpoint:course-one|q": {
                        **award,
                        "course_id": "course-one",
                        "lesson_id": "lesson-a",
                        "checkpoint_id": "q",
                        "xp": 0,
                    },
                    "practice:old-course|lesson-x|m": {
                        **award,
                        "course_id": "old-course",
                        "lesson_id": "lesson-x",
                        "checkpoint_id": None,
                        "xp": 30,
                        "at": at - 86400000,
                    },
                }
            }
        if path.endswith("/old-course"):
            raise HTTPException(403, "denied")
        return {
            "id": "course-one",
            "title": {"en": "Mathematics 7", "id": "Matematika 7"},
            "chapters": [
                {
                    "id": "ch1",
                    "title": {"en": "Fractions", "id": "Pecahan"},
                    "lessons": [{"id": "lesson-a"}],
                }
            ],
        }

    monkeypatch.setattr(student_assistant, "learning_data", data)
    body = ask(api, "saya sudah menyelesaikan games apa saja?").json()["reply"]
    assert body["text"] == "Kamu sudah menyelesaikan 2 game (total 60 XP):"
    assert body["table"] == {
        "columns": ["Modul", "Bab", "Selesai"],
        "rows": [
            ["Matematika 7", "Pecahan", "3 Nov 2026"],
            ["Modul lain", "–", "2 Nov 2026"],
        ],
    }
    assert sum(p.endswith("/course-one") for p in paths) == 1


def test_games_without_completions_says_so(api, monkeypatch):
    async def data(*args, **kwargs):
        return {"awards": {}}

    monkeypatch.setattr(student_assistant, "learning_data", data)
    body = ask(api, "Which games have I finished?", "en").json()["reply"]
    assert "haven't finished any games" in body["text"]
    assert "table" not in body


def pathway(cells, student="student-one"):
    return {
        "student_id": student,
        "years": [
            {
                "timing": "current",
                "subjects": [
                    {
                        "subject": "mathematics",
                        "subject_name": {"en": "Mathematics", "id": "Matematika"},
                        "units": [{"objectives": cells}],
                    }
                ],
            },
            {
                "timing": "past",
                "subjects": [
                    {
                        "units": [
                            {
                                "objectives": [
                                    {
                                        "objective_id": "old",
                                        "level": "beginning",
                                        "average_pct": 1,
                                        "evidence": 3,
                                    }
                                ]
                            }
                        ]
                    }
                ],
            },
        ],
        "objectives": [
            {"id": "o1", "code": "7.N.1", "text": {"en": "Add", "id": "Menjumlah pecahan"}},
            {"id": "o2", "code": "7.N.2", "text": {"en": "Ratios", "id": "Rasio"}},
        ],
        "suggestions": [
            {
                "objective_id": "o1",
                "course_id": "course-one",
                "chapter_id": "ch1",
                "lesson_id": "lesson-a",
                "lesson_title": {"en": "Like fractions", "id": "Pecahan senama"},
            }
        ],
    }


def test_improve_lists_weakest_current_objectives_with_suggested_lessons(api, monkeypatch):
    async def data(settings, authorization, path, params=None):
        assert path == "/api/v1/learning/students/me/pathway"
        return pathway(
            [
                {"objective_id": "o2", "level": "developing", "average_pct": 62.4, "evidence": 4},
                {"objective_id": "o1", "level": "beginning", "average_pct": 40, "evidence": 5},
                {"objective_id": "o3", "level": "advanced", "average_pct": 95, "evidence": 5},
                {
                    "objective_id": "o4",
                    "level": "not_enough_evidence",
                    "average_pct": None,
                    "evidence": 0,
                },
            ]
        )

    monkeypatch.setattr(student_assistant, "learning_data", data)
    body = ask(api, "selama belajar ini saya harus improve apa di bagian mana?").json()["reply"]
    assert body["table"]["rows"] == [
        ["Matematika", "Menjumlah pecahan", "40% · Perlu bimbingan", "Pecahan senama"],
        ["Matematika", "Rasio", "62% · Berkembang", "–"],
    ]
    assert body["actions"] == [
        {
            "label": "Pecahan senama",
            "course_id": "course-one",
            "chapter_id": "ch1",
            "lesson_id": "lesson-a",
        }
    ]


@pytest.mark.parametrize(
    "cells, expected",
    [
        ([], "Belum cukup hasil kuis"),
        (
            [{"objective_id": "o1", "level": "proficient", "average_pct": 80, "evidence": 3}],
            "sudah di level cakap",
        ),
    ],
)
def test_improve_without_weak_objectives_is_truthful(api, monkeypatch, cells, expected):
    async def data(*args, **kwargs):
        return pathway(cells)

    monkeypatch.setattr(student_assistant, "learning_data", data)
    body = ask(api, "bagian mana yang kurang?").json()["reply"]
    assert expected in body["text"]
    assert "table" not in body


def test_improve_rejects_another_students_pathway(api, monkeypatch):
    async def data(*args, **kwargs):
        return pathway([], student="peer")

    monkeypatch.setattr(student_assistant, "learning_data", data)
    assert ask(api, "apa yang harus aku tingkatkan?").status_code == 503


def test_unknown_question_lists_what_kibo_can_answer(api, monkeypatch):
    async def data(*args, **kwargs):
        raise AssertionError("no school read for an unknown question")

    monkeypatch.setattr(student_assistant, "learning_data", data)
    body = ask(api, "halo kibo").json()["reply"]
    assert "Game apa saja" in body["text"] and "improve" in body["text"]


CATALOGUE_COURSE = {
    "id": "course-one",
    "title": {"en": "Mathematics 11", "id": "Matematika 11"},
    "chapters": [
        {
            "id": "ch-limit",
            "title": {"en": "Limits", "id": "Limit"},
            "lessons": [{"id": "l1", "title": {"en": "Limit of a function", "id": "Limit fungsi"}}],
        },
        {
            "id": "ch-int",
            "title": {"en": "Integrals", "id": "Integral"},
            "lessons": [
                {"id": "l2", "title": {"en": "Antiderivatives", "id": "Antiturunan"}},
                {"id": "l3", "title": {"en": "Definite integral", "id": "Integral tentu"}},
            ],
        },
    ],
}


def catalogue_data(paths):
    async def data(settings, authorization, path, params=None):
        paths.append((path, params))
        if path == "/api/v1/learning/curriculum/courses":
            return {"items": [{"id": "course-one"}], "next": None}
        return CATALOGUE_COURSE

    return data


def test_search_finds_the_chapter_and_opens_it_in_lessons(api, monkeypatch):
    paths = []
    monkeypatch.setattr(student_assistant, "learning_data", catalogue_data(paths))
    body = ask(api, "cariin dong materi integral").json()["reply"]
    assert body["text"] == 'Kibo menemukan 1 bab tentang "integral":'
    assert body["table"]["rows"] == [["Matematika 11", "Integral", "Antiturunan"]]
    assert body["actions"] == [
        {"label": "Integral", "course_id": "course-one", "chapter_id": "ch-int", "lesson_id": "l2"}
    ]
    assert all(params == {"class_id": "class-one"} for _, params in paths)
    body = ask(api, "integral tentu materi").json()["reply"]
    assert body["actions"][0]["lesson_id"] == "l3"


def test_search_without_a_match_or_topic_is_truthful(api, monkeypatch):
    monkeypatch.setattr(student_assistant, "learning_data", catalogue_data([]))
    body = ask(api, "cari materi fotosintesis").json()["reply"]
    assert body["text"] == 'Belum ada materi tentang "fotosintesis" di modul kelasmu.'
    assert "actions" not in body and "table" not in body
    assert "Materi apa yang kamu cari" in ask(api, "cariin materi dong").json()["reply"]["text"]


def test_materi_with_a_selected_reading_still_explains_the_reading(api, monkeypatch):
    called = []

    async def search(*args):
        called.append(True)

    monkeypatch.setattr(student_assistant, "search_lessons", search)

    async def reading(*args):
        raise HTTPException(503, "reading path taken")

    monkeypatch.setattr(student_assistant, "reading", reading)
    response = api.post(
        "/api/ai/v1/student/assistant",
        json={**PAYLOAD, "question": "jelaskan materi ini", "locale": "id", "source": SOURCE},
        headers={"Authorization": "Bearer valid-student"},
    )
    assert response.status_code == 503 and not called


@pytest.mark.parametrize(
    "question, expected",
    [
        ("aku pengen bunuh diri", "hubungi 112"),
        ("aku dibully di kelas", "guru BK"),
        ("lagi stres banget sama tugas", "guru BK"),
        ("kerjain tugas matematika aku dong", "tidak bisa memberi jawaban"),
        ("kunci jawaban kuis IPA apa?", "tidak bisa memberi jawaban"),
        ("tugas temanku apa aja?", "bukan data teman"),
        ("halo kibo", "Halo! Kibo siap bantu"),
        ("makasih ya kibo", "Sama-sama"),
        ("berapa biaya SPP bulan ini?", "orang tuamu atau wali kelas"),
    ],
)
def test_fixed_replies_never_read_school_data(api, monkeypatch, question, expected):
    async def data(*args, **kwargs):
        raise AssertionError("no school read")

    monkeypatch.setattr(student_assistant, "learning_data", data)
    response = ask(api, question)
    assert response.status_code == 200
    assert expected in response.json()["reply"]["text"]


def test_own_unfinished_work_still_reaches_homework(api, monkeypatch):
    async def data(settings, authorization, path, params=None):
        assert path == "/api/v1/learning/assignments"
        return {"items": [], "next": None}

    monkeypatch.setattr(student_assistant, "learning_data", data)
    body = ask(api, "tugas apa yang belum aku kerjain?").json()["reply"]
    assert body["text"] == "Tidak ada tugas terbuka pada halaman ini."


class FakeModel:
    """Answers the planner with `plan`, the writer with `written`, and readings with an
    explanation; records every call."""

    calls: list = []
    plan = "not json"
    written = '{"text": "Ramah: ini jawabannya ya!"}'

    def __init__(self, settings):
        pass

    async def complete(self, **kwargs):
        FakeModel.calls.append(kwargs)
        if kwargs["system_prompt"].startswith("KIBO_PLANNER"):
            return FakeModel.plan
        if kwargs["system_prompt"].startswith("KIBO_WRITER"):
            return FakeModel.written
        return '{"answered": true, "text": "Integral adalah kebalikan turunan."}'


@pytest.fixture
def model(monkeypatch):
    FakeModel.calls = []
    FakeModel.plan = "not json"
    FakeModel.written = '{"text": "Ramah: ini jawabannya ya!"}'
    monkeypatch.setattr(student_assistant, "LlmClient", FakeModel)
    return FakeModel


def school_with_integral():
    lesson = {
        "id": "l2",
        "title": {"en": "Antiderivatives", "id": "Antiturunan"},
        "blocks": [{"type": "paragraph", "text": {"en": "Reverse.", "id": "Kebalikan turunan."}}],
    }

    async def data(settings, authorization, path, params=None):
        if path == "/api/v1/learning/curriculum/courses":
            return {"items": [{"id": "course-one"}], "next": None}
        if path.endswith("/lessons/l2"):
            return lesson
        return {**CATALOGUE_COURSE, "class_id": "class-one", "publication": "published"}

    return data


def test_concept_question_explains_from_the_matching_reading(api, monkeypatch, model):
    monkeypatch.setattr(student_assistant, "learning_data", school_with_integral())
    body = ask(api, "apa itu integral?").json()["reply"]
    assert body["text"] == "Integral adalah kebalikan turunan."
    assert body["source"] == "Antiturunan"
    assert body["actions"][0] == {
        "label": "Integral",
        "course_id": "course-one",
        "chapter_id": "ch-int",
        "lesson_id": "l2",
    }
    assert "Kebalikan turunan." in model.calls[0]["user_message"]


def test_concept_reading_that_does_not_answer_points_to_the_chapter(api, monkeypatch, model):
    async def unanswered(self, **kwargs):
        model.calls.append(kwargs)
        return '{"answered": false, "text": ""}'

    monkeypatch.setattr(model, "complete", unanswered)
    monkeypatch.setattr(student_assistant, "learning_data", school_with_integral())
    body = ask(api, "apa itu integral?").json()["reply"]
    assert body["text"].startswith('Kibo menemukan bab "Integral" tentang "integral"')
    assert "yang kamu pilih" not in body["text"]
    assert body["actions"][0]["lesson_id"] == "l2"


def results_data(pcts, released=True):
    at = int(datetime(2026, 11, 3, 3, tzinfo=timezone.utc).timestamp() * 1000)

    def item(i, pct):
        return {
            "course": {"id": "c1", "title": {"en": "Mathematics 7", "id": "Matematika 7"}},
            "assessment": {
                "id": f"a{i}",
                "definition": {"kind": "exam", "title": {"en": f"Test {i}", "id": f"Ujian {i}"}},
            },
            "attempt": {
                "state": "released" if released else "submitted",
                "submitted_at": at - i * 86400000,
                "result": (
                    {"score": pct, "maximum": 100, "score_pct": pct, "passed": pct >= 70}
                    if released
                    else None
                ),
            },
        }

    async def data(settings, authorization, path, params=None):
        assert path == "/api/v1/learning/assessments/mine"
        assert params == {"class_id": "class-one"}
        return {"items": [item(i, p) for i, p in enumerate(pcts)], "server_now": at}

    return data


def test_own_perfect_scores_are_listed(api, monkeypatch):
    monkeypatch.setattr(student_assistant, "learning_data", results_data([100, 85.5, 100]))
    body = ask(api, "Hasil nilai ujian saya ada yg dapat 100 ga ?").json()["reply"]
    assert body["text"] == "Ya! Ada 2 hasil dengan nilai 100:"
    assert [row[0] for row in body["table"]["rows"]] == ["Ujian 0", "Ujian 2"]
    assert body["table"]["rows"][0][1:3] == ["Matematika 7", "100"]


def test_no_perfect_score_names_the_best(api, monkeypatch):
    monkeypatch.setattr(student_assistant, "learning_data", results_data([72, 91.5]))
    body = ask(api, "ada yang 100 gak nilaiku?").json()["reply"]
    assert body["text"] == "Belum ada yang 100. Nilai tertinggimu 91.5 di Ujian 1 (Matematika 7)."


def test_all_released_results_and_none_unreleased(api, monkeypatch):
    monkeypatch.setattr(student_assistant, "learning_data", results_data([64, 80]))
    body = ask(api, "nilai ujian aku gimana?").json()["reply"]
    assert body["text"] == "Ini 2 hasil ujian dan kuismu yang sudah dirilis:"
    assert len(body["table"]["rows"]) == 2
    monkeypatch.setattr(student_assistant, "learning_data", results_data([90], released=False))
    body = ask(api, "nilai ujian aku gimana?").json()["reply"]
    assert body["text"] == "Belum ada hasil ujian atau kuis yang dirilis gurumu."


@pytest.mark.parametrize("question", ["nilai temanku berapa?", "aku ranking berapa di kelas?"])
def test_other_students_scores_and_ranking_are_refused(api, monkeypatch, question):
    async def data(*args, **kwargs):
        raise AssertionError("no school read")

    monkeypatch.setattr(student_assistant, "learning_data", data)
    assert "bukan data teman" in ask(api, question).json()["reply"]["text"]


SIS = "https://sis.example"


@pytest.fixture
def agent(api, model):
    """The planner and writer need a configured assistant provider; SIS reads need its URL."""
    main.app.dependency_overrides[main.get_settings] = lambda: Settings(
        _env_file=None,
        assistant_provider_base_url="https://gateway.example/v1",
        assistant_provider_api_key="token",
        assistant_model="digital-school-chat",
        assistant_planner_model="digital-school-fast",
        sis_owner_url=SIS,
    )
    yield model
    main.app.dependency_overrides.clear()


def plan(**fields):
    import json as _json

    return _json.dumps({"tool": "smalltalk", "perfect_only": False, "topic": "", **fields})


def test_holiday_question_reads_the_school_calendar_and_is_rewritten(api, monkeypatch, agent):
    agent.plan = plan(
        tool="calendar", date_from="2026-12-01", date_to="2026-12-31", calendar_kind="holiday"
    )

    async def data(settings, authorization, path, params=None, **kw):
        assert kw.get("base") == SIS
        assert path == "/api/v1/sis-service/learning/owner/student/calendar"
        assert params == {
            "school_id": "school-one",
            "tenant_id": "tenant-one",
            "from": "2026-12-01",
            "to": "2026-12-31",
        }
        entry = {"startDate": "2026-12-21", "endDate": "2027-01-02", "title": "Libur semester"}
        exam = {"startDate": "2026-12-01", "endDate": "2026-12-05", "title": "PAS"}
        return {
            "educationCalendar": {
                "items": [
                    {"entry": {**entry, "kind": "holiday"}},
                    {"entry": {**exam, "kind": "exam_period"}},
                ]
            }
        }

    monkeypatch.setattr(student_assistant, "learning_data", data)
    body = ask(api, "Di bulan desember besok ada jadwal libur ga ?").json()["reply"]
    assert body["text"] == "Ramah: ini jawabannya ya!"
    assert body["table"]["rows"] == [["Senin, 21 Des – Sabtu, 2 Jan", "Libur semester", "Libur"]]
    planner, writer = agent.calls
    assert planner["user_message"] == "Question:\nDi bulan desember besok ada jadwal libur ga ?"
    assert "Today is" in planner["system_prompt"]
    # The writer sees the read's own facts only (product owner decision, 2026-11-04).
    assert "Libur semester" in writer["user_message"] and "PAS" not in writer["user_message"]


def test_writer_failure_keeps_the_reads_own_text(api, monkeypatch, agent):
    agent.plan = plan(tool="homework")
    agent.written = "not json"

    async def data(settings, authorization, path, params=None, **kw):
        return {"items": [], "next": None}

    monkeypatch.setattr(student_assistant, "learning_data", data)
    assert ask(api, "kerjaan dari guru gimana").json()["reply"]["text"] == (
        "Tidak ada tugas terbuka pada halaman ini."
    )


def test_planned_schedule_uses_the_date_range(api, monkeypatch, agent):
    start = int(datetime(2099, 1, 2, 1, tzinfo=timezone.utc).timestamp() * 1000)
    agent.plan = plan(tool="schedule", date_from="2099-01-02", date_to="2099-01-02")

    async def data(settings, authorization, path, params=None, **kw):
        meeting = {"class_id": "class-one", "title": "Math", "starts_at": start}
        other = {**meeting, "starts_at": start + 3 * 86400000}
        return {
            "class_id": "class-one",
            "items": [
                {**meeting, "ends_at": start + 3600000},
                {**other, "ends_at": other["starts_at"] + 3600000},
            ],
        }

    monkeypatch.setattr(student_assistant, "learning_data", data)
    body = ask(api, "jadwal tanggal 2 januari").json()["reply"]
    assert body["table"]["rows"] == [["Jumat, 2 Jan", "08.00–09.00", "Matematika"]]


def test_extracurricular_lists_joined_clubs_and_open_ones(api, monkeypatch, agent):
    agent.plan = plan(tool="extracurricular")

    async def data(settings, authorization, path, params=None, **kw):
        assert kw.get("base") == SIS
        if path.endswith("/context/student"):
            scope = {"schoolId": "s1", "tenantId": "t1", "academicYear": "2026/2027"}
            return {"classes": [scope]}
        assert params == {"schoolId": "s1", "tenantId": "t1", "academicYear": "2026/2027"}

        def club(title, **extra):
            activity = {"title": {"en": title, "id": title}, "published": True}
            return {"activity": activity, "coachName": "Bu Rina", "archived": False, **extra}

        return {
            "items": [
                club("Robotik", enrollment={"status": "enrolled"}),
                club("Tahfiz", enrollment={"status": "pending_consent"}),
                club("Futsal", enrollment=None),
            ]
        }

    monkeypatch.setattr(student_assistant, "learning_data", data)
    body = ask(api, "eskul apa saja yg saya ikuti ?").json()["reply"]
    assert body["table"]["rows"] == [["Robotik", "Ikut"], ["Tahfiz", "Menunggu izin orang tua"]]
    assert "Bu Rina" not in agent.calls[-1]["user_message"]


def test_upcoming_quizzes_skip_done_and_closed_ones(api, monkeypatch, agent):
    agent.plan = plan(tool="upcoming_quizzes")
    now = int(datetime(2026, 11, 4, 3, tzinfo=timezone.utc).timestamp() * 1000)
    day = 86400000

    def quiz(title, attempt=None, **definition):
        return {
            "course": {"title": {"en": "Math 7", "id": "Matematika 7"}},
            "assessment": {"definition": {"title": {"en": title, "id": title}, **definition}},
            "attempt": attempt,
        }

    async def data(settings, authorization, path, params=None, **kw):
        assert path == "/api/v1/learning/assessments/mine"
        return {
            "server_now": now,
            "items": [
                quiz("Kuis A", closes_at=now + 2 * day),
                quiz("Kuis B", attempt={"state": "in_progress"}, closes_at=now + day),
                quiz("Kuis C", attempt={"state": "released"}),
                quiz("Kuis D", closes_at=now - day),
                quiz("Kuis E", opens_at=now + 3 * day),
            ],
        }

    monkeypatch.setattr(student_assistant, "learning_data", data)
    body = ask(api, "kuis apa yang belum aku kerjain").json()["reply"]
    assert [row[0] for row in body["table"]["rows"]] == ["Kuis B", "Kuis A", "Kuis E"]
    assert body["table"]["rows"][2][2] == "Dibuka 7 Nov 2026"


def test_progress_lists_modules_least_done_first(api, monkeypatch, agent):
    agent.plan = plan(tool="progress")

    async def data(settings, authorization, path, params=None, **kw):
        assert path == "/api/v1/learning/curriculum/courses"
        return {
            "items": [
                {"title": {"en": "IPA", "id": "IPA"}, "chapters": 4, "completed": 3},
                {"title": {"en": "Math", "id": "Matematika"}, "chapters": 5, "completed": 1},
            ],
            "next": None,
        }

    monkeypatch.setattr(student_assistant, "learning_data", data)
    body = ask(api, "aku udah sampai mana belajarnya?").json()["reply"]
    assert body["table"]["rows"] == [["Matematika", "1/5"], ["IPA", "3/4"]]


def test_unsupported_topic_is_named_kindly_with_examples(api, monkeypatch, agent):
    agent.plan = plan(tool="unsupported", topic="pembayaran SPP")
    agent.written = "not json"

    async def data(*args, **kwargs):
        raise AssertionError("no school read")

    monkeypatch.setattr(student_assistant, "learning_data", data)
    text = ask(api, "SPP bulan ini udah dibayar belum?").json()["reply"]["text"]
    assert text.startswith("Kibo belum bisa menjawab soal pembayaran SPP. Coba tanya:")


@pytest.mark.parametrize("question", ["aku pengen bunuh diri", "kerjain tugas aku dong"])
def test_safety_replies_never_reach_a_model(api, monkeypatch, agent, question):
    async def data(*args, **kwargs):
        raise AssertionError("no school read")

    monkeypatch.setattr(student_assistant, "learning_data", data)
    ask(api, question)
    assert agent.calls == []


def test_search_matches_the_module_title(api, monkeypatch):
    monkeypatch.setattr(student_assistant, "learning_data", catalogue_data([]))
    body = ask(api, "Di matematika ada materi integral gak sih?").json()["reply"]
    assert body["actions"][0]["chapter_id"] == "ch-int"


def test_open_reading_that_does_not_answer_falls_back_to_other_chapters(api, monkeypatch, agent):
    agent.plan = plan(tool="explain", topic="integral")
    base = FakeModel.complete

    async def complete(self, **kwargs):
        # The open reading (Pecahan) does not answer; the Integral chapter's reading does.
        if kwargs["user_message"].startswith("Published reading:\nPecahan"):
            return '{"answered": false, "text": ""}'
        return await base(self, **kwargs)

    monkeypatch.setattr(FakeModel, "complete", complete)
    reading = {**READING, "title": {"en": "Fractions", "id": "Pecahan"}}
    school = school_with_integral()

    async def data(settings, authorization, path, params=None, **kw):
        if path.endswith("/lessons/reading-one"):
            return reading
        if path.endswith("/course-one") and params is None:
            return {"id": "course-one", "class_id": "class-one", "publication": "published"}
        return await school(settings, authorization, path, params)

    monkeypatch.setattr(student_assistant, "learning_data", data)
    response = api.post(
        "/api/ai/v1/student/assistant",
        json={**PAYLOAD, "question": "apa itu integral?", "locale": "id", "source": SOURCE},
        headers={"Authorization": "Bearer valid-student"},
    )
    body = response.json()["reply"]
    assert body["actions"][0]["chapter_id"] == "ch-int"

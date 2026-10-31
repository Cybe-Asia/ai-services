import json

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import main
from app import teacher_chapter_summary as summary
from app.config import Settings

PAYLOAD = {"course": "course.1", "chapter": "ch1", "class_id": "SEC-7A", "locale": "id"}


def chapter(taken=24):
    return {
        "chapter_id": "ch1",
        "title": {"en": "Living things", "id": "Makhluk hidup"},
        "quizzes": [
            {
                "assessment_id": "q1",
                "title": {"en": "Checkpoint", "id": "Cek bagian sel"},
                "lesson_id": "l1",
                "roster": 28,
                "taken": taken,
                "average_pct": 71.0,
                "below_pass": 10,
                "to_review": 3,
                # Learning never sends a student here; the allow-list keeps one out if it did.
                "students": [{"name": "Aisha Putri"}],
            }
        ],
        "students_scored": taken,
        "average_pct": 71.0,
        "pass_mark": 75,
        "below_pass_mark": 9,
        "weakest": [
            {
                "assessment_id": "q1",
                "question_id": "q1-3",
                "prompt": {
                    "en": "Which organelle makes energy?",
                    "id": "Organel mana yang menghasilkan energi?",
                },
                "scored": 24,
                "correct_rate_pct": 25.0,
            }
        ],
    }


@pytest.fixture
def setup(monkeypatch):
    state = {"data": {"chapters": [chapter()]}, "reply": None, "prompts": [], "reads": []}

    async def data(settings, authorization, path, params=None, subject="Student"):
        if authorization != "Bearer teacher":
            raise HTTPException(401, "Teacher session required")
        state["reads"].append((path, dict(params or {})))
        return state["data"]

    async def complete(self, **kwargs):
        state["prompts"].append(kwargs)
        return state["reply"]

    monkeypatch.setattr(summary, "learning_data", data)
    monkeypatch.setattr(summary.LlmClient, "complete", complete)
    settings = Settings(
        selection_provider_base_url="https://llm.example.test",
        selection_provider_api_key="k",
        selection_model="gpt-test",
    )
    main.app.dependency_overrides[main.get_settings] = lambda: settings
    yield state
    main.app.dependency_overrides.clear()


def post(payload=PAYLOAD, token="Bearer teacher"):
    return TestClient(main.app).post(
        "/api/ai/v1/teacher/chapter-summary",
        json=payload,
        headers={"Authorization": token} if token else {},
    )


def test_summarises_learnings_aggregates_and_sends_no_student(setup):
    setup["reply"] = json.dumps({"summary": "Kelas rata-rata 71. Bahas ulang organel."})
    response = post()
    assert response.status_code == 200
    assert response.json()["summary"] == "Kelas rata-rata 71. Bahas ulang organel."
    assert setup["reads"] == [
        ("/api/v1/learning/gradebook/courses/course.1/chapters", {"class_id": "SEC-7A"})
    ]
    prompt = setup["prompts"][0]["system_prompt"]
    assert "Organel mana yang menghasilkan energi?" in prompt and "Indonesian" in prompt
    assert "Aisha" not in prompt and "students" not in json.dumps(
        setup["prompts"][0]["user_message"]
    )


def test_a_provider_failure_or_bad_reply_is_503_never_a_made_up_text(setup):
    assert post().status_code == 503
    setup["reply"] = json.dumps({"summary": ""})
    assert post().status_code == 503


def test_no_results_yet_asks_nothing_of_the_model(setup):
    setup["data"] = {"chapters": [chapter(taken=0)]}
    assert post().status_code == 409
    assert setup["prompts"] == []


def test_an_unknown_chapter_or_no_session_is_refused(setup):
    assert post({**PAYLOAD, "chapter": "other"}).status_code == 404
    assert post(token=None).status_code == 401
    assert post({**PAYLOAD, "class_id": "../x"}).status_code == 422

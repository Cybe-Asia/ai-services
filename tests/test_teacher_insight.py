import json

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import main
from app import teacher_insight as insight
from app.config import Settings

T = lambda en, id_=None: {"en": en, "id": id_ or en}  # noqa: E731


def student(active_days=4, attempts=3):
    return {
        "student": {"id": "STU-1", "name": "Aisha Rahma", "class_id": "7A"},
        "course": {"id": "c", "title": T("Science")},
        "period": {"days": 7, "from": "2026-11-01", "to": "2026-11-07"},
        "status": "needs_attention",
        "reasons": [{"code": "QUIZ_BELOW_PASS", "item_id": "q3", "score_pct": 45.0}],
        "last_activity_at": 1,
        "activity_available": True,
        "facts": {
            "active_days": active_days,
            "active_minutes": 40,
            "quiz_attempts": attempts,
            "quiz_retries": 2,
            "game_rounds": 1,
        },
        "opened": [
            {
                "type": "game",
                "id": "c2.water-practice",
                "title": T("Matter"),
                "opens": 4,
                "game": "water-practice",
            }
        ],
        "sessions": [
            {
                "id": "s2",
                "start": "b",
                "end": "b",
                "active_minutes": 5,
                "steps": [
                    {"at": "t4", "name": "learning.quiz_start", "props": {}},
                ],
            },
            {
                "id": "s1",
                "start": "a",
                "end": "a",
                "active_minutes": 9,
                "steps": [
                    {
                        "at": "t1",
                        "name": "learning.quiz_submit",
                        "props": {},
                        "result": {"passed": False},
                    },
                    {"at": "t2", "name": "learning.quiz_start", "props": {}},
                    {
                        "at": "t3",
                        "name": "learning.quiz_submit",
                        "props": {},
                        "result": {"passed": False},
                    },
                    {"at": "t3b", "name": "learning.chapter_open", "props": {}},
                ],
            },
        ],
        "attempts": [
            {
                "attempt_id": "a1",
                "assessment_id": "q3",
                "title": T("Heat quiz"),
                "number": 2,
                "state": "released",
                "started_at": 1,
                "submitted_at": 2,
                "score_pct": 45.0,
                "passed": False,
            }
        ],
        "mistakes": [
            {
                "assessment_id": "q3",
                "assessment_title": T("Heat quiz"),
                "question_id": "q",
                "prompt": T("Unit of heat?"),
                "wrong": 3,
                "answered": 3,
                "chosen": T("Celsius"),
                "correct": T("Joule"),
            }
        ],
        "papers": [{"id": "q3", "title": T("Heat quiz")}],
    }


@pytest.fixture
def setup(monkeypatch):
    insight._cache.clear()
    state = {"data": student(), "reply": None, "prompts": [], "reads": []}

    async def data(settings, authorization, path, params=None, subject="Student"):
        if authorization != "Bearer teacher":
            raise HTTPException(401, "Teacher session required")
        state["reads"].append((path, dict(params or {})))
        return state["data"]

    async def complete(self, **kwargs):
        state["prompts"].append(kwargs)
        return state["reply"]

    monkeypatch.setattr(insight, "learning_data", data)
    monkeypatch.setattr(insight.LlmClient, "complete", complete)
    settings = Settings(
        selection_provider_base_url="https://llm.example.test",
        selection_provider_api_key="k",
        selection_model="gpt-test",
    )
    main.app.dependency_overrides[main.get_settings] = lambda: settings
    yield state
    main.app.dependency_overrides.clear()


def post(payload, token="Bearer teacher"):
    return TestClient(main.app).post(
        "/api/ai/v1/teacher/insight-conclusion", json=payload, headers={"Authorization": token}
    )


REQ = {"course": "c", "student": "STU-1", "class_id": "7A", "days": 7, "locale": "en"}


def test_student_facts_are_numbered_without_names_and_findings_cite_them(setup):
    setup["reply"] = json.dumps(
        {
            "findings": [
                {
                    "kind": "retry",
                    "title": "Retries without reading",
                    "body": "Heat quiz retried.",
                    "evidence": ["f3", "f99"],
                },
                {
                    "kind": "misconception",
                    "title": "Celsius for heat",
                    "body": "Chose Celsius 3 times.",
                    "evidence": [],
                },
            ],
            "next_step": "Reread heat transfer before the next attempt.",
        }
    )
    r = post(REQ)
    assert r.status_code == 200, r.text
    body = r.json()
    assert setup["reads"] == [
        ("/api/v1/learning/insight/courses/c/students/STU-1", {"days": "7", "class_id": "7A"})
    ]
    prompt = setup["prompts"][0]["system_prompt"]
    assert "Aisha" not in prompt and "STU-1" not in prompt
    # failed submit → start → failed submit → open → start: two retries, one without reading.
    assert '"retries_after_failing": 2' in prompt and '"retries_without_reading": 1' in prompt
    # The uncited finding and the unknown fact id are dropped; the cited one keeps its tab.
    assert [f["kind"] for f in body["findings"]] == ["retry"]
    assert body["findings"][0]["evidence"] == [{"id": "f3", "tab": "journey"}]
    assert body["model"] == "gpt-test" and len(body["input_hash"]) == 64 and body["cached"] is False
    again = post(REQ).json()
    assert again["cached"] is True and len(setup["prompts"]) == 1


def test_a_thin_period_gets_no_conclusion(setup):
    setup["data"] = student(active_days=1, attempts=1)
    assert post(REQ).status_code == 409
    assert setup["prompts"] == []


def test_the_class_reads_aggregates_only(setup):
    setup["data"] = {
        "period": {"days": 7},
        "students_total": 28,
        "quiz_attempts": 10,
        "quiz_retries": 4,
        "glance": {
            "active_students": 20,
            "median_active_minutes": 30,
            "chapter_opens": 40,
            "game_opens": 9,
            "game_rounds": 12,
        },
        "students": [
            {
                "name": "Aisha Rahma",
                "student_id": "STU-1",
                "status": "behind",
                "reasons": [{"code": "INACTIVE"}],
            }
        ],
        "most_opened": {
            "chapters": [{"id": "c3", "opens": 30, "students": 18, "title": T("Heat")}],
            "games": [],
            "lessons": [],
        },
        "quizzes": [
            {
                "title": T("Heat quiz"),
                "students": 18,
                "attempts": 30,
                "retries": 12,
                "first_tries": 18,
                "first_try_passed": 6,
                "most_by_one": 5,
            },
            {
                "title": T("Science method quiz"),
                "students": 1,
                "attempts": 1,
                "retries": 0,
                "first_tries": 0,
                "first_try_passed": 0,
                "most_by_one": 1,
            },
        ],
    }
    setup["reply"] = json.dumps(
        {
            "findings": [{"kind": "retry", "title": "Heat quiz", "body": "b", "evidence": ["f4"]}],
            "next_step": "Re-teach heat.",
        }
    )
    r = post({"course": "c", "days": 7, "locale": "id"})
    assert r.status_code == 200, r.text
    prompt = setup["prompts"][0]["system_prompt"]
    assert "Aisha" not in prompt and "STU-1" not in prompt and '"behind": 1' in prompt
    # A quiz with no scored first try has no pass rate, so it cannot read as failed.
    assert '"scored_first_tries": 0, "passed_first_try": null' in prompt
    assert '"scored_first_tries": 18, "passed_first_try": 6' in prompt
    assert r.json()["findings"][0]["evidence"] == [{"id": "f4", "tab": "quizzes"}]


def test_a_broken_answer_is_503_and_a_missing_session_401(setup):
    setup["reply"] = "not json"
    assert post(REQ).status_code == 503
    assert post(REQ, token="").status_code == 401
    assert post({**REQ, "days": 14}).status_code == 422


def test_the_semester_period_is_passed_to_learning(setup):
    setup["reply"] = json.dumps(
        {
            "findings": [{"kind": "retry", "title": "t", "body": "b", "evidence": ["f3"]}],
            "next_step": "n",
        }
    )
    assert post({**REQ, "days": "semester"}).status_code == 200
    assert setup["reads"][0][1]["days"] == "semester"

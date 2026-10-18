import asyncio
import json

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import main
from app import teacher_assessment_selection as selection
from app.config import Settings

PAYLOAD = {
    "subject": "mathematics",
    "question_count": 6,
    "goal": "Ulangan aljabar dan geometri",
    "locale": "id",
}


def entry(i: int, topic: str, difficulty: str, grade=7):
    return {
        "owner": "staff_1",
        "subject": "mathematics",
        "revision": 3,
        "publication": "published",
        "metadata": {"grade": grade, "topics": [topic], "difficulty": difficulty},
        "question": {
            "id": f"q{i}",
            "type": "mcq",
            "prompt": {"en": f"Question {i}", "id": f"Soal {i} tentang {topic}"},
            "options": [],
            "points": 1,
        },
    }


BANK = (
    [entry(i, "Aljabar", "easy") for i in range(0, 3)]
    + [entry(i, "Aljabar", "medium") for i in range(3, 7)]
    + [entry(i, "Aljabar", "hard") for i in range(7, 9)]
    + [entry(i, "Geometri", "medium") for i in range(9, 12)]
    + [entry(i, "Statistika", "easy", grade=8) for i in range(12, 14)]
)


@pytest.fixture
def setup(monkeypatch):
    state = {
        "pages": [{"items": BANK[:10], "next": "q9"}, {"items": BANK[10:]}],
        "reply": None,
        "prompts": [],
        "reads": [],
    }

    async def data(settings, authorization, path, params=None, subject="Student"):
        if authorization != "Bearer teacher":
            raise HTTPException(401, "Teacher session required")
        state["reads"].append((path, dict(params or {})))
        return state["pages"][1] if (params or {}).get("after") else state["pages"][0]

    async def complete(self, **kwargs):
        state["prompts"].append(kwargs)
        return state["reply"]

    monkeypatch.setattr(selection, "learning_data", data)
    monkeypatch.setattr(selection.LlmClient, "complete", complete)
    settings = Settings(
        selection_provider_base_url="https://llm.example.test",
        selection_provider_api_key="k",
        selection_model="gpt-test",
    )
    main.app.dependency_overrides[main.get_settings] = lambda: settings
    yield state
    main.app.dependency_overrides.clear()


def post(payload=PAYLOAD, token="Bearer teacher"):
    client = TestClient(main.app)
    return client.post(
        "/api/ai/v1/teacher/assessment-selection",
        content=json.dumps(payload),
        headers={"Authorization": token, "Content-Type": "application/json"},
    )


def test_a_valid_model_proposal_is_returned_as_learning_selection(setup):
    setup["reply"] = json.dumps(
        {
            "groups": [
                {"topic": "Aljabar", "difficulty": "medium", "grade": None, "quota": 3},
                {"topic": "Aljabar", "difficulty": "hard", "grade": None, "quota": 1},
                {"topic": "Geometri", "difficulty": None, "grade": None, "quota": 2},
            ],
            "rationale": "Aljabar lebih banyak karena baru dipelajari; geometri pengulangan.",
        }
    )
    response = post()
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["source"] == "model"
    assert body["question_count"] == 6
    assert body["available"] == len(BANK)
    assert body["selection"] == [
        {"topic": "Aljabar", "difficulty": "medium", "quota": 3},
        {"topic": "Aljabar", "difficulty": "hard", "quota": 1},
        {"topic": "Geometri", "quota": 2},
    ]
    assert body["flags"] == []
    # Both pages were read with the teacher's token and only published entries requested.
    assert [p["status"] for _, p in setup["reads"]] == ["published", "published"]
    prompt = setup["prompts"][0]
    assert prompt["feature"] == "selection"
    assert prompt["require_complete"] is True
    assert "stu-" not in prompt["system_prompt"]
    inventory = json.loads(prompt["system_prompt"].split("Inventory (JSON): ", 1)[1])
    assert inventory["total_published"] == len(BANK)
    assert {t["topic"]: t["count"] for t in inventory["topics"]} == {
        "Aljabar": 9,
        "Geometri": 3,
        "Statistika": 2,
    }


def test_a_proposal_the_bank_cannot_fill_falls_back_to_a_heuristic_split(setup):
    setup["reply"] = json.dumps(
        {
            "groups": [{"topic": "Geometri", "difficulty": "hard", "grade": None, "quota": 6}],
            "rationale": "x",
        }
    )
    body = post().json()
    assert body["source"] == "heuristic"
    assert body["flags"] == ["validation_failed"]
    assert sum(g["quota"] for g in body["selection"]) == 6
    # The goal names Aljabar and Geometri, so Statistika is left out.
    assert {g["topic"] for g in body["selection"]} == {"Aljabar", "Geometri"}
    assert body["rationale"].startswith("Pembagian otomatis")


def test_quotas_that_do_not_add_up_or_repeat_groups_are_rejected():
    items = [selection.item_from(e) for e in BANK]
    ok = selection.Proposal(groups=[selection.Group(topic="Aljabar", quota=6)], rationale="r")
    selection.check_proposal(items, ok, 6)
    with pytest.raises(ValueError):
        selection.check_proposal(items, ok, 7)
    twice = selection.Proposal(
        groups=[
            selection.Group(topic="aljabar", quota=3),
            selection.Group(topic="Aljabar", quota=3),
        ],
        rationale="r",
    )
    with pytest.raises(ValueError):
        selection.check_proposal(items, twice, 6)
    overlapping = selection.Proposal(
        groups=[
            selection.Group(topic="Geometri", quota=3),
            selection.Group(difficulty="medium", quota=5),
        ],
        rationale="r",
    )
    # Geometri takes its three medium questions, leaving four Aljabar medium ones.
    with pytest.raises(ValueError):
        selection.check_proposal(items, overlapping, 8)


def test_provider_failure_still_gives_the_teacher_a_split(setup):
    setup["reply"] = None
    body = post().json()
    assert body["source"] == "heuristic"
    assert body["flags"] == ["provider_unavailable"]
    assert sum(g["quota"] for g in body["selection"]) == 6


def test_grade_narrows_the_bank_when_it_has_that_grade(setup):
    setup["reply"] = None
    body = post({**PAYLOAD, "grade": 8, "question_count": 2, "goal": "statistik"}).json()
    assert body["available"] == 2
    assert all(g["grade"] == 8 for g in body["selection"])
    short = post({**PAYLOAD, "grade": 8, "question_count": 5, "goal": "apa saja"}).json()
    assert short["question_count"] == 2
    assert "short_of_questions" in short["flags"]


def test_feature_off_means_404_before_any_read(setup, monkeypatch):
    main.app.dependency_overrides[main.get_settings] = lambda: Settings()
    assert post().status_code == 404
    assert setup["reads"] == []


def test_the_teacher_token_is_required_and_the_goal_is_bounded(setup):
    setup["reply"] = None
    assert post(token="Bearer student").status_code == 401
    assert post({**PAYLOAD, "goal": "x" * 1001}).status_code == 422
    assert post({**PAYLOAD, "question_count": 51}).status_code == 422
    assert post({**PAYLOAD, "subject": "Math 7"}).status_code == 422


def test_bank_items_refuse_a_malformed_page(setup):
    setup["pages"] = [{"items": [{"publication": "draft"}]}]
    with pytest.raises(HTTPException) as exc:
        asyncio.run(selection.bank_items(Settings(), "Bearer teacher", "mathematics"))
    assert exc.value.status_code == 503

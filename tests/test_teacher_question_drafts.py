import asyncio
import json

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import main
from app import teacher_question_drafts as drafts
from app.config import Settings

PAYLOAD = {
    "course_id": "seed.smp7.civics",
    "chapter_ids": ["c1"],
    "count": 3,
    "locale": "id",
}
READING_ID = "Pancasila adalah dasar negara Indonesia. Sila ketiga berbunyi Persatuan Indonesia."
READING_EN = (
    "Pancasila is the foundation of the Indonesian state. "
    "The third principle is the Unity of Indonesia."
)


def text(en, ind):
    return {"en": en, "id": ind}


COURSE = {
    "id": "seed.smp7.civics",
    "revision": 3,
    "publication": "published",
    "syllabus": {
        "subject": "civics",
        "chapters": [
            {
                "id": "c1",
                "title": text("Pancasila", "Pancasila"),
                "lessons": [
                    {
                        "id": "l1",
                        "title": text("Foundation", "Dasar negara"),
                        "blocks": [{"type": "paragraph", "text": text(READING_EN, READING_ID)}],
                    },
                    {
                        "id": "l2",
                        "title": text("Video", "Video"),
                        "blocks": [],
                        "media": {"kind": "video"},
                    },
                ],
            },
            {
                "id": "c2",
                "title": text("Video only", "Video saja"),
                "lessons": [{"id": "l3", "title": text("V", "V"), "blocks": []}],
            },
        ],
    },
}


def mcq(
    prompt="Apa dasar negara Indonesia?",
    evidence="Pancasila adalah dasar negara Indonesia",
    lesson="l1",
):
    return {
        "type": "mcq",
        "type_reason": "Definisi cocok untuk pilihan ganda.",
        "lesson_id": lesson,
        "evidence": evidence,
        "difficulty": "easy",
        "topic": "Pancasila",
        "prompt": text("What is the foundation of the Indonesian state?", prompt),
        "options": [
            text("Pancasila", "Pancasila"),
            text("UUD 1945", "UUD 1945"),
            text("Bhinneka", "Bhinneka"),
        ],
        "correct": [0],
        "explanation": text("The reading says so.", "Bacaan menyebutkannya."),
        "points": 1,
    }


def essay():
    return {
        "type": "essay",
        "type_reason": "Menjelaskan alasan butuh esai.",
        "lesson_id": "l1",
        "evidence": "Sila ketiga berbunyi Persatuan Indonesia",
        "difficulty": "medium",
        "topic": None,
        "prompt": text("Explain the third principle.", "Jelaskan sila ketiga."),
        "rubric": text(
            "Names unity and gives an example.", "Menyebut persatuan dan memberi contoh."
        ),
        "criteria": [
            {"description": text("Names unity", "Menyebut persatuan"), "points": 2},
            {"description": text("Gives an example", "Memberi contoh"), "points": 3},
        ],
        "points": 1,
    }


def matching():
    return {
        "type": "matching",
        "type_reason": "Pasangan istilah.",
        "lesson_id": "l1",
        "evidence": "The third principle is the Unity of Indonesia",
        "difficulty": "medium",
        "prompt": text("Match each term.", "Pasangkan istilah."),
        "options": [text("A", "A"), text("B", "B"), text("C", "C")],
        "matches": [text("1", "1"), text("2", "2"), text("3", "3")],
        "points": 3,
    }


@pytest.fixture
def setup(monkeypatch):
    state = {"course": COURSE, "bank": [], "reply": None, "prompts": [], "reads": []}

    async def data(settings, authorization, path, params=None, subject="Student"):
        if authorization != "Bearer teacher":
            raise HTTPException(401, "Teacher session required")
        state["reads"].append(path)
        if path.startswith("/api/v1/learning/curriculum/courses/"):
            return state["course"]
        return {"items": state["bank"], "next": None}

    async def complete(self, **kwargs):
        state["prompts"].append(kwargs)
        return state["reply"]

    monkeypatch.setattr(drafts, "learning_data", data)
    monkeypatch.setattr(drafts.LlmClient, "complete", complete)
    settings = Settings(
        drafts_provider_base_url="https://llm.example.test",
        drafts_provider_api_key="k",
        drafts_model="gpt-test",
    )
    main.app.dependency_overrides[main.get_settings] = lambda: settings
    yield state
    main.app.dependency_overrides.clear()


def post(payload=PAYLOAD, token="Bearer teacher"):
    return TestClient(main.app).post(
        "/api/ai/v1/teacher/question-drafts",
        content=json.dumps(payload),
        headers={"Authorization": token, "Content-Type": "application/json"},
    )


def test_drafts_become_bank_questions_with_their_lesson_as_source(setup):
    setup["reply"] = json.dumps({"drafts": [mcq(), essay(), matching()]})
    response = post()
    assert response.status_code == 200, response.text
    body = response.json()
    assert [d["question"]["type"] for d in body["drafts"]] == ["mcq", "essay", "matching"]
    first = body["drafts"][0]
    assert first["question"]["answer_key"] == {"option_id": "o1"}
    assert first["question"]["rubric"] is None and first["question"]["id"].startswith("ai-")
    assert first["metadata"]["source"] == {
        "course_id": "seed.smp7.civics",
        "course_revision": 3,
        "chapter_id": "c1",
        "lesson_id": "l1",
    }
    assert first["metadata"]["topics"] == ["Pancasila"] and first["type_reason"]
    essay_q = body["drafts"][1]["question"]
    assert essay_q["points"] == 5 and [c["id"] for c in essay_q["criteria"]] == ["c1", "c2"]
    pairs = body["drafts"][2]["question"]["answer_key"]["pairs"]
    assert pairs == [
        {"left_id": "o1", "right_id": "r1"},
        {"left_id": "o2", "right_id": "r2"},
        {"left_id": "o3", "right_id": "r3"},
    ]
    assert body["discarded"] == {"invalid": 0, "no_quote": 0, "duplicate": 0, "type": 0}
    # Only the chosen lessons' text reached the model; the video lesson added nothing.
    system = setup["prompts"][0]["system_prompt"]
    assert READING_ID in system and "Video saja" not in system


def test_a_quote_not_in_the_reading_or_a_broken_draft_is_dropped(setup):
    bad_options = {**mcq(prompt="Pilihan rusak"), "correct": [7]}
    setup["reply"] = json.dumps(
        {
            "drafts": [
                mcq(evidence="Pancasila dibuat oleh model sendiri"),
                bad_options,
                {"type": "essay"},
                mcq(),
            ]
        }
    )
    body = post().json()
    assert len(body["drafts"]) == 1
    assert body["discarded"]["no_quote"] == 1 and body["discarded"]["invalid"] == 2
    assert "short_of_drafts" in body["flags"]


def test_duplicates_of_the_bank_or_of_each_other_are_dropped(setup):
    setup["bank"] = [{"question": {"prompt": text("x", "APA dasar negara Indonesia")}}]
    setup["reply"] = json.dumps(
        {"drafts": [mcq(), mcq(prompt="Sebutkan sila ketiga"), mcq(prompt="Sebutkan sila ketiga!")]}
    )
    body = post().json()
    assert len(body["drafts"]) == 1 and body["discarded"]["duplicate"] == 2


def test_chosen_types_are_kept_and_other_types_dropped(setup):
    setup["reply"] = json.dumps({"drafts": [mcq(), essay()]})
    body = post({**PAYLOAD, "types": ["essay"]}).json()
    assert [d["question"]["type"] for d in body["drafts"]] == ["essay"] and body["discarded"][
        "type"
    ] == 1


def test_a_chapter_with_no_text_reading_is_refused_before_the_model(setup):
    assert post({**PAYLOAD, "chapter_ids": ["c2"]}).status_code == 409
    assert post({**PAYLOAD, "chapter_ids": ["missing"]}).status_code == 404
    assert setup["prompts"] == []


def test_provider_failure_is_unavailable_and_nothing_is_invented(setup):
    setup["reply"] = None
    assert post().status_code == 503
    setup["reply"] = "not json"
    assert post().status_code == 503


def test_feature_off_means_404_before_any_read(setup):
    main.app.dependency_overrides[main.get_settings] = lambda: Settings()
    assert post().status_code == 404
    assert setup["reads"] == []


def test_the_teacher_token_and_bounds_are_checked(setup):
    assert post(token="Bearer student").status_code == 401
    assert post({**PAYLOAD, "count": 11}).status_code == 422
    assert post({**PAYLOAD, "note": "x" * 501}).status_code == 422
    assert post({**PAYLOAD, "chapter_ids": ["c1", "c1"]}).status_code == 422
    assert post({**PAYLOAD, "types": []}).status_code == 422


def test_one_request_per_teacher_at_a_time(setup, monkeypatch):
    events = {}

    async def slow(self, **kwargs):
        events["started"].set()
        await events["release"].wait()
        return json.dumps({"drafts": [mcq()]})

    monkeypatch.setattr(drafts.LlmClient, "complete", slow)

    async def run():
        # Events belong to the running loop (Python 3.9).
        events["started"], events["release"] = asyncio.Event(), asyncio.Event()
        started, release = events["started"], events["release"]
        settings = main.app.dependency_overrides[main.get_settings]()
        payload = drafts.DraftRequest.model_validate(PAYLOAD)
        first = asyncio.create_task(drafts.draft_questions(settings, "Bearer teacher", payload))
        await started.wait()
        with pytest.raises(HTTPException) as busy:
            await drafts.draft_questions(settings, "Bearer teacher", payload)
        release.set()
        return busy.value.status_code, await first

    status, result = asyncio.run(run())
    assert status == 429 and len(result["drafts"]) == 1


def test_question_shapes_per_type():
    def build(**fields):
        return drafts.question_from(drafts.Draft.model_validate({**mcq(), **fields}))

    assert build(type="tf", options=[], correct=[], answer=False)["answer_key"] == {"value": False}
    assert build(type="multi", correct=[2, 0])["answer_key"] == {"option_ids": ["o1", "o3"]}
    assert build(type="short", options=[], correct=[], accepted=[" Pancasila "])["answer_key"] == {
        "accepted": ["Pancasila"],
        "case_sensitive": False,
    }
    assert build(type="numeric", options=[], correct=[], value=5, tolerance=0)["answer_key"] == {
        "value": 5.0,
        "tolerance": 0,
    }
    for broken in (
        {"correct": [0, 1]},
        {"options": [text("a", "a")]},
        {"type": "tf", "options": [], "correct": []},
    ):
        with pytest.raises(ValueError):
            build(**broken)

import asyncio
import copy
import json

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import main
from app import teacher_essay_review as essay
from app.config import Settings
from app.metrics import registry

PAYLOAD = {
    "class_id": "class",
    "course_id": "course",
    "assessment_id": "quiz",
    "attempt_id": "attempt",
    "locale": "en",
}
ANSWER_TEXT = (
    "Photosynthesis uses light energy to turn water and carbon dioxide into sugar.\n"
    "It happens in the chloroplast and releases oxygen as a by-product."
)
DETAIL = {"id": "quiz", "course_id": "course", "class_id": "class", "publication": "published"}
ESSAY = {
    "id": "essay",
    "type": "essay",
    "prompt": {"en": "Explain photosynthesis", "id": "Jelaskan fotosintesis"},
    "options": [],
    "points": 10,
    "answer_key": {"model_answer": "Light, water and CO2 become sugar and oxygen."},
    "rubric": {"en": "Inputs, outputs and location", "id": "Masukan, keluaran, tempat"},
    "criteria": [
        {
            "id": "inputs",
            "description": {"en": "Names light, water and CO2", "id": "Masukan"},
            "points": 4,
        },
        {
            "id": "outputs",
            "description": {"en": "Names sugar and oxygen", "id": "Keluaran"},
            "points": 4,
        },
        {
            "id": "where",
            "description": {"en": "Names the chloroplast", "id": "Tempat"},
            "points": 2,
        },
    ],
}
ATTEMPT = {
    "id": "attempt",
    "assessment_id": "quiz",
    "course_id": "course",
    "student_id": "stu-7f3a",
    "revision": 3,
    "state": "submitted",
    "questions": [
        {
            "id": "tf",
            "type": "tf",
            "prompt": {"en": "T?", "id": "B?"},
            "options": [],
            "points": 1,
            "answer_key": {"value": True},
            "rubric": None,
        },
        ESSAY,
    ],
    "answers": [
        {"question_id": "tf", "response": {"value": True}, "score": 1, "feedback": None},
        {
            "question_id": "essay",
            "response": {"text": ANSWER_TEXT},
            "score": None,
            "feedback": None,
        },
    ],
}
REVIEW = {
    "criteria": [
        {
            "id": "inputs",
            "verdict": "met",
            "evidence": ["light energy to turn water and carbon dioxide"],
            "reason": "All three inputs are named.",
        },
        {
            "id": "outputs",
            "verdict": "partial",
            "evidence": ["into sugar", "releases oxygen"],
            "reason": "Both products appear but oxygen is called a by-product only.",
        },
        {
            "id": "where",
            "verdict": "met",
            "evidence": ["in the chloroplast"],
            "reason": "The location is named.",
        },
    ],
    "feedback": "You named the inputs, the products and the chloroplast. Say that oxygen is a "
    "product of the reaction, not just a by-product.",
}
ESSAY_BLOCK = {
    "essay_provider_base_url": "https://api.openai.example/v1",
    "essay_provider_api_key": "essay-key",
    "essay_model": "gpt-6-luna",
}


def settings(**values):
    return Settings(_env_file=None, **ESSAY_BLOCK, **values)


@pytest.fixture(autouse=True)
def fresh_registry():
    registry.reset()
    yield
    registry.reset()


@pytest.fixture
def setup(monkeypatch):
    calls = {"reads": [], "prompts": [], "answer": json.dumps(REVIEW), "attempt": ATTEMPT}

    async def data(settings, authorization, path, params=None, subject="Student"):
        calls["reads"].append((path, subject, authorization))
        if path.endswith("/items/quiz"):
            return copy.deepcopy(DETAIL)
        return copy.deepcopy(calls["attempt"])

    async def complete(self, **kwargs):
        calls["prompts"].append(kwargs)
        return calls["answer"]

    monkeypatch.setattr(essay, "learning_data", data)
    monkeypatch.setattr(essay.LlmClient, "complete", complete)
    return calls


def run(current=None, payload=PAYLOAD):
    return asyncio.run(
        essay.review_essays(
            current or settings(), "Bearer teacher", essay.EssayReviewRequest(**payload)
        )
    )


def test_suggestion_per_criterion_with_anchored_quotes_and_a_code_computed_score(setup):
    result = run()
    assert result["contractVersion"] == 1 and result["revision"] == 3
    [question] = result["questions"]
    assert question["question_id"] == "essay"
    assert question["prompt_version"] == essay.PROMPT_VERSION
    assert question["model"] == "gpt-6-luna"
    assert question["maximum"] == 10
    # met 4 + partial 2 + met 2
    assert question["suggested_score"] == 8
    assert question["needs_teacher"] is False and question["flags"] == []
    verdicts = {c["id"]: c for c in question["criteria"]}
    assert verdicts["outputs"]["verdict"] == "partial"
    assert verdicts["outputs"]["evidence"] == ["into sugar", "releases oxygen"]
    assert verdicts["where"]["points"] == 2
    assert question["feedback"].startswith("You named")
    # The teacher token reads the assessment, the attempt, and the attempt again afterwards.
    assert [r[0].rsplit("/", 1)[-1] for r in setup["reads"]] == ["quiz", "attempt", "attempt"]
    assert all(r[1] == "Teacher" for r in setup["reads"])
    [prompt] = setup["prompts"]
    assert prompt["user_message"] == ANSWER_TEXT
    assert prompt["response_format"] == {"type": "json_object"}
    assert prompt["require_complete"] is True
    assert prompt["feature"] == "essay"
    for leaked in ("stu-7f3a", "attempt", '"id": "tf"', "T?"):
        assert leaked not in prompt["system_prompt"]
    assert (
        "Jelaskan" not in prompt["system_prompt"]
        and "Explain photosynthesis" in prompt["system_prompt"]
    )
    assert registry.snapshot()["outcomes"] == {("essay", "ok"): 1}


def test_feature_off_means_404_before_any_read(setup):
    with pytest.raises(HTTPException) as error:
        run(Settings(_env_file=None))
    assert error.value.status_code == 404
    assert setup["reads"] == []


def test_feature_block_is_used_for_the_call_not_the_main_provider(setup, monkeypatch):
    captured = {}
    original = essay.LlmClient.__init__

    def init(self, current):
        captured["model"] = current.ai_model
        captured["effort"] = current.ai_reasoning_effort
        captured["max_tokens"] = current.ai_max_tokens
        original(self, current)

    monkeypatch.setattr(essay.LlmClient, "__init__", init)
    run(settings(ai_model="qwen3:4b", ai_provider_base_url="http://gateway.local/v1"))
    assert captured == {"model": "gpt-6-luna", "effort": "low", "max_tokens": 1200}


@pytest.mark.parametrize(
    "answer,reason",
    [
        ("not json", "contract"),
        (json.dumps({**REVIEW, "score": 9}), "unknown key"),
        (json.dumps({"criteria": REVIEW["criteria"][:2], "feedback": "x"}), "missing id"),
        (
            json.dumps(
                {
                    "criteria": [*REVIEW["criteria"], {**REVIEW["criteria"][0], "id": "extra"}],
                    "feedback": "x",
                }
            ),
            "unknown id",
        ),
        (
            json.dumps(
                {
                    **REVIEW,
                    "criteria": [
                        {**REVIEW["criteria"][0], "evidence": ["light makes glucose"]},
                        *REVIEW["criteria"][1:],
                    ],
                }
            ),
            "invented quote",
        ),
        (
            json.dumps(
                {
                    **REVIEW,
                    "criteria": [
                        {**REVIEW["criteria"][0], "evidence": []},
                        *REVIEW["criteria"][1:],
                    ],
                }
            ),
            "met without evidence",
        ),
        (
            json.dumps({**REVIEW, "feedback": "A very intelligent student; 8/10 for this."}),
            "claims in feedback",
        ),
        (
            json.dumps(
                {
                    **REVIEW,
                    "criteria": [
                        {**REVIEW["criteria"][0], "reason": "Better than other students."},
                        *REVIEW["criteria"][1:],
                    ],
                }
            ),
            "claims in reason",
        ),
    ],
)
def test_invalid_model_output_yields_no_suggestion_and_flags_the_teacher(setup, answer, reason):
    setup["answer"] = answer
    [question] = run()["questions"]
    assert question["suggested_score"] is None, reason
    assert question["feedback"] is None
    assert question["needs_teacher"] is True
    assert set(question["flags"]) == {"validation_failed", "needs_teacher"}
    assert [c["verdict"] for c in question["criteria"]] == [None, None, None]


def test_quotes_are_reanchored_to_the_answer_when_only_whitespace_differs(setup):
    setup["answer"] = json.dumps(
        {
            **REVIEW,
            "criteria": [
                REVIEW["criteria"][0],
                {**REVIEW["criteria"][1], "evidence": ["as a  by-product"]},
                {**REVIEW["criteria"][2], "evidence": ["sugar.  It happens"]},
            ],
        }
    )
    [question] = run()["questions"]
    verdicts = {c["id"]: c for c in question["criteria"]}
    assert verdicts["outputs"]["evidence"] == ["as a by-product"]
    assert verdicts["where"]["evidence"] == ["sugar.\nIt happens"]


def test_unclear_verdict_scores_zero_for_that_criterion_and_needs_the_teacher(setup):
    setup["answer"] = json.dumps(
        {
            **REVIEW,
            "criteria": [
                REVIEW["criteria"][0],
                {**REVIEW["criteria"][1], "verdict": "unclear", "evidence": []},
                {**REVIEW["criteria"][2], "verdict": "not_met", "evidence": []},
            ],
        }
    )
    [question] = run()["questions"]
    assert question["suggested_score"] == 4
    assert question["needs_teacher"] is True and question["flags"] == ["needs_teacher"]


def test_partial_shares_round_half_up_to_the_point_unit(setup):
    attempt = copy.deepcopy(ATTEMPT)
    attempt["questions"][1]["points"] = 0.15
    attempt["questions"][1]["criteria"] = [
        {"id": "inputs", "description": {"en": "a", "id": "a"}, "points": 0.05},
        {"id": "outputs", "description": {"en": "b", "id": "b"}, "points": 0.05},
        {"id": "where", "description": {"en": "c", "id": "c"}, "points": 0.05},
    ]
    setup["attempt"] = attempt
    setup["answer"] = json.dumps(
        {**REVIEW, "criteria": [{**c, "verdict": "partial"} for c in REVIEW["criteria"]]}
    )
    [question] = run()["questions"]
    assert question["suggested_score"] == 0.09


def test_empty_answer_is_not_sent_to_the_model(setup):
    attempt = copy.deepcopy(ATTEMPT)
    attempt["answers"][1]["response"] = {"text": "   "}
    setup["attempt"] = attempt
    [question] = run()["questions"]
    assert setup["prompts"] == []
    assert question["flags"] == ["empty_answer", "needs_teacher"]
    assert registry.snapshot()["outcomes"] == {("essay", "empty_answer"): 1}


def test_essay_without_criteria_and_non_essay_questions_are_skipped(setup):
    attempt = copy.deepcopy(ATTEMPT)
    del attempt["questions"][1]["criteria"]
    setup["attempt"] = attempt
    result = run()
    assert result["questions"] == [] and setup["prompts"] == []
    # No model call: no second attempt read either.
    assert len(setup["reads"]) == 2


def test_student_projection_is_refused_as_not_a_teacher(setup):
    attempt = copy.deepcopy(ATTEMPT)
    for question in attempt["questions"]:
        question.pop("rubric")
        question.pop("answer_key")
        question.pop("criteria", None)
    setup["attempt"] = attempt
    with pytest.raises(HTTPException) as error:
        run()
    assert error.value.status_code == 403
    assert setup["prompts"] == []


@pytest.mark.parametrize(
    "change,status",
    [
        ({"state": "in_progress"}, 409),
        ({"state": "released"}, 409),
        ({"id": "other"}, 404),
        ({"course_id": "other"}, 404),
    ],
)
def test_wrong_state_or_identity_is_refused(setup, change, status):
    setup["attempt"] = {**copy.deepcopy(ATTEMPT), **change}
    with pytest.raises(HTTPException) as error:
        run()
    assert error.value.status_code == status


def test_class_mismatch_is_not_found(setup):
    with pytest.raises(HTTPException) as error:
        run(payload={**PAYLOAD, "class_id": "another"})
    assert error.value.status_code == 404
    assert setup["prompts"] == []


def test_attempt_changed_during_inference_is_a_conflict(setup, monkeypatch):
    reads = []

    async def data(settings, authorization, path, params=None, subject="Student"):
        reads.append(path)
        if path.endswith("/items/quiz"):
            return copy.deepcopy(DETAIL)
        attempt = copy.deepcopy(ATTEMPT)
        if len(reads) > 2:
            attempt["revision"] = 4
        return attempt

    monkeypatch.setattr(essay, "learning_data", data)
    with pytest.raises(HTTPException) as error:
        run()
    assert error.value.status_code == 409


def test_malformed_projection_is_unavailable_not_a_suggestion(setup):
    attempt = copy.deepcopy(ATTEMPT)
    attempt["questions"][1]["criteria"][0]["points"] = 5  # sum no longer equals points
    setup["attempt"] = attempt
    with pytest.raises(HTTPException) as error:
        run()
    assert error.value.status_code == 503


def test_every_call_failing_is_503(setup):
    setup["answer"] = None
    with pytest.raises(HTTPException) as error:
        run()
    assert error.value.status_code == 503
    assert registry.snapshot()["outcomes"] == {("essay", "provider_unavailable"): 1}


def test_route_bounds_body_and_validates_selection(monkeypatch):
    client = TestClient(main.app)
    main.app.dependency_overrides[main.get_settings] = lambda: settings()
    try:
        response = client.post(
            "/api/ai/v1/teacher/essay-review",
            headers={"Authorization": "Bearer teacher"},
            json={**PAYLOAD, "extra": 1},
        )
        assert response.status_code == 422
        response = client.post(
            "/api/ai/v1/teacher/essay-review",
            headers={"Authorization": "Bearer teacher"},
            content=b"x" * 3000,
        )
        assert response.status_code == 413

        async def review(current, authorization, payload):
            assert authorization == "Bearer teacher"
            return {"contractVersion": 1, "questions": []}

        monkeypatch.setattr(main, "review_essays", review)
        response = client.post(
            "/api/ai/v1/teacher/essay-review",
            headers={"Authorization": "Bearer teacher"},
            json=PAYLOAD,
        )
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert response.json() == {"contractVersion": 1, "questions": []}
    finally:
        main.app.dependency_overrides.clear()


def test_metadata_reports_whether_essay_review_is_enabled():
    client = TestClient(main.app)
    main.app.dependency_overrides[main.get_settings] = lambda: settings()
    try:
        assert client.get("/api/v1/ai-service/metadata").json()["essayReviewEnabled"] is True
        main.app.dependency_overrides[main.get_settings] = lambda: Settings(_env_file=None)
        assert client.get("/api/v1/ai-service/metadata").json()["essayReviewEnabled"] is False
    finally:
        main.app.dependency_overrides.clear()


def test_forbidden_claims_filter_targets_the_student_not_the_text():
    for text in (
        "The answer names light and water but not carbon dioxide.",
        "Tiga masukan disebutkan dengan jelas.",
        "Add the location of the reaction.",
    ):
        assert essay.FORBIDDEN_CLAIMS.search(text) is None, text
    for text in (
        "A brilliant student.",
        "Siswa ini malas membaca.",
        "Weaker than other students.",
        "Nilai: 8",
        "about 60% correct",
        "Score 7/10",
    ):
        assert essay.FORBIDDEN_CLAIMS.search(text), text

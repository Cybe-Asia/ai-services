import asyncio
import copy
import json

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import main, student_context
from app import student_discovery as discovery
from app.config import Settings
from app.student_context import StudentContext

PAYLOAD = {"class_id": "class", "course_id": "course", "day": "all", "locale": "en"}
CONTEXT = StudentContext("actor", "student", "school", "tenant", ("class",))
REF = "lesson:course:chapter:reading:3"
EVIDENCE = {
    "contract_version": 1,
    "class_id": "class",
    "student_id": "student",
    "courses": [
        {
            "id": "course",
            "class_id": "class",
            "fingerprint": "a" * 64,
            "title": {"en": "Fractions", "id": "Pecahan"},
            "evidence": [
                {
                    "ref": REF,
                    "kind": "lesson",
                    "course_id": "course",
                    "title": {"en": "Parts", "id": "Bagian"},
                    "state": "completed",
                    "date": None,
                    "result": None,
                    "source": {
                        "body": {
                            "en": "Fractions express parts of a whole.",
                            "id": "Pecahan menyatakan bagian.",
                        },
                        "formula": None,
                    },
                },
                {
                    "ref": "assessment:course:quiz:attempt:3:5",
                    "kind": "assessment",
                    "course_id": "course",
                    "title": {"en": "Quiz", "id": "Kuis"},
                    "state": "released",
                    "date": None,
                    "source": None,
                    "result": {"score": 7, "maximum": 10, "feedback": ["Explain the denominator."]},
                },
            ],
        }
    ],
}
# Complete canonical transport fields; missing or unknown owner fields fail closed.
EVIDENCE["courses"][0].update(
    {
        "revision": 3,
        "pacing_revision": 0,
        "progress_revision": 1,
        "subject": "mathematics",
        "completed": 1,
        "total": 1,
        "description": {"en": "Material", "id": "Materi"},
    }
)
for record in EVIDENCE["courses"][0]["evidence"]:
    record.update(
        {
            "revision": 3,
            "chapter_id": "chapter",
            "lesson_id": None,
            "assessment_id": None,
            "attempt_id": None,
            "description": {"en": "Published", "id": "Terpublikasi"},
        }
    )
EVIDENCE["courses"][0]["evidence"][0]["lesson_id"] = "reading"
EVIDENCE["courses"][0]["evidence"][1].update({"assessment_id": "quiz", "attempt_id": "attempt"})

ANSWER = {
    "summary": [{"text": "A fraction describes part of a whole.", "refs": [REF]}],
    "reflection": [{"text": "How does the denominator describe equal parts?", "refs": [REF]}],
}


@pytest.fixture
def setup(monkeypatch):
    calls = []

    async def context(*args):
        return CONTEXT

    async def data(settings, authorization, path, params=None):
        calls.append((path, params))
        return copy.deepcopy(EVIDENCE)

    async def complete(self, **kwargs):
        calls.append(kwargs)
        return json.dumps(ANSWER)

    monkeypatch.setattr(discovery, "resolve_student_context", context)
    monkeypatch.setattr(discovery, "learning_data", data)
    monkeypatch.setattr(discovery.LlmClient, "complete", complete)
    return calls


def run():
    return asyncio.run(
        discovery.reflect_student(Settings(), "Bearer valid", discovery.DiscoveryRequest(**PAYLOAD))
    )


def test_exact_readonly_sources_and_feedback_without_scores_or_privileged_tools(setup):
    result = run()
    assert result["fingerprint"] == "a" * 64
    assert result["summary"] == ANSWER["summary"]
    assert len([c for c in setup if isinstance(c, tuple)]) == 2
    prompt = next(c for c in setup if isinstance(c, dict))
    body = json.loads(prompt["user_message"])
    assert body["evidence"][1]["released_feedback"] == ["Explain the denominator."]
    assert not any(
        key in prompt["user_message"]
        for key in ['"score"', '"maximum"', '"answers"', '"questions"', '"student_id"']
    )
    assert prompt["response_format"] == {"type": "json_object"}
    assert prompt["require_complete"] is True


@pytest.mark.parametrize(
    "answer",
    [
        None,
        "",
        "not json",
        json.dumps({**ANSWER, "score": 90}),
        json.dumps({**ANSWER, "summary": [{"text": "Made up", "refs": ["peer"]}]}),
        json.dumps({**ANSWER, "summary": [{"text": "You mastered fractions.", "refs": [REF]}]}),
        json.dumps({**ANSWER, "reflection": [{"text": "Your confidence is high.", "refs": [REF]}]}),
        json.dumps({**ANSWER, "summary": [{"text": "Your score is 90%.", "refs": [REF]}]}),
        json.dumps({**ANSWER, "summary": [{"text": "Material", "refs": [REF, REF]}]}),
    ],
)
def test_provider_invalid_incomplete_uncited_or_prohibited_output_is_503(
    setup, monkeypatch, answer
):
    async def complete(*args, **kwargs):
        return answer

    monkeypatch.setattr(discovery.LlmClient, "complete", complete)
    with pytest.raises(HTTPException) as error:
        run()
    assert error.value.status_code == 503


@pytest.mark.parametrize("change", ["revision", "revoked", "identity"])
def test_post_inference_source_result_and_identity_changes_never_leak(setup, monkeypatch, change):
    reads = 0

    async def data(*args, **kwargs):
        nonlocal reads
        reads += 1
        if reads == 2:
            if change == "revoked":
                raise HTTPException(403, "revoked")
            altered = copy.deepcopy(EVIDENCE)
            altered["courses"][0]["fingerprint"] = "b" * 64
            return altered
        return copy.deepcopy(EVIDENCE)

    async def context(*args):
        return (
            CONTEXT
            if reads == 0
            else StudentContext("new", "student", "school", "tenant", ("class",))
        )

    monkeypatch.setattr(discovery, "learning_data", data)
    if change == "identity":
        monkeypatch.setattr(discovery, "resolve_student_context", context)
    with pytest.raises(HTTPException) as error:
        run()
    assert error.value.status_code in {403, 409}


@pytest.mark.parametrize("alter", ["foreign", "hidden", "duplicate", "overflow", "empty"])
def test_malformed_cross_scope_unreleased_or_overflow_evidence_fails_before_inference(
    setup, monkeypatch, alter
):
    evidence = copy.deepcopy(EVIDENCE)
    records = evidence["courses"][0]["evidence"]
    if alter == "foreign":
        evidence["student_id"] = "peer"
    elif alter == "hidden":
        records[1]["state"] = "reviewed"
    elif alter == "duplicate":
        records.append(copy.deepcopy(records[0]))
    elif alter == "overflow":
        records[0]["source"]["body"]["en"] = "x" * 24001
    else:
        records.clear()

    async def data(*args, **kwargs):
        return evidence

    monkeypatch.setattr(discovery, "learning_data", data)
    with pytest.raises(HTTPException) as error:
        run()
    assert error.value.status_code == 503
    assert not any(isinstance(call, dict) for call in setup)


def test_date_filters_use_jakarta_and_never_invent_historical_dates():
    payload = discovery.DiscoveryRequest(**{**PAYLOAD, "day": "undated"})
    _, refs = discovery.evidence_context(EVIDENCE, "student", payload)
    assert REF in refs
    assert discovery.selected_day(1790877600000, "2026-10-02")  # Oct 1 18:00 UTC
    assert not discovery.selected_day(None, "2026-10-02")
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            discovery.reflect_student(
                Settings(),
                "Bearer valid",
                discovery.DiscoveryRequest(**{**PAYLOAD, "day": "9999-10-02"}),
            )
        )
    assert error.value.status_code == 422


def test_route_forbids_identity_and_evidence_body_and_requires_real_context(monkeypatch):
    client = TestClient(main.app)
    for extra in [{"student_id": "peer"}, {"sources": []}, {"question": "freeform"}]:
        assert (
            client.post("/api/ai/v1/student/discovery", json={**PAYLOAD, **extra}).status_code
            == 422
        )
    assert client.post("/api/ai/v1/student/discovery", content=b"x" * 2049).status_code == 413

    async def deny(*args, **kwargs):
        raise HTTPException(401, "Student session required")

    monkeypatch.setattr(student_context, "learning_data", deny)
    assert client.post("/api/ai/v1/student/discovery", json=PAYLOAD).status_code == 401


@pytest.mark.parametrize(
    "model,metadata,accepted",
    [
        ("gemma2:2b", {}, False),
        ("gemma2:2b", {"finish_reason": None}, False),
        ("gemma2:2b", {"finish_reason": "length"}, False),
        ("gemma2:2b", {"finish_reason": "stop"}, True),
        ("qwen3:4b", {}, False),
        ("qwen3:4b", {"done": True}, False),
        ("qwen3:4b", {"done_reason": "stop"}, False),
        ("qwen3:4b", {"done": False, "done_reason": "stop"}, False),
        ("qwen3:4b", {"done": True, "done_reason": "length"}, False),
        ("qwen3:4b", {"done": True, "done_reason": "stop"}, True),
    ],
)
def test_discovery_requires_affirmative_provider_completion(monkeypatch, model, metadata, accepted):
    import httpx

    from app import llm_client

    original = httpx.AsyncClient
    content = json.dumps(ANSWER)
    body = (
        {"message": {"content": content}, **metadata}
        if model == "qwen3:4b"
        else {"choices": [{"message": {"content": content}, **metadata}]}
    )
    monkeypatch.setattr(
        llm_client.httpx,
        "AsyncClient",
        lambda **kwargs: original(
            **kwargs, transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))
        ),
    )
    settings = Settings(
        _env_file=None, ai_model=model, ai_provider_base_url="http://localhost:11434/v1"
    )
    answer = asyncio.run(
        llm_client.LlmClient(settings).complete("Source", "Question", require_complete=True)
    )
    assert (answer is not None) == accepted

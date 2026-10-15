import asyncio
import base64
import json

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app import main, recitation
from app.config import Settings
from app.recitation_align import align, assessment, assessment_from_words, normalize
from app.student_context import StudentContext

AN_NAS_1 = "قُلْ أَعُوذُ بِرَبِّ ٱلنَّاسِ"
CONTEXT = StudentContext("actor", "student", "school", "tenant", ("class",))
AUDIO = base64.b64encode(b"\x1aE\xdf\xa3" + bytes(64)).decode()
PAYLOAD = {
    "locale": "id",
    "verse": "114:1",
    "expected_text": AN_NAS_1,
    "mime": "audio/webm",
    "audio_base64": AUDIO,
}
VERDICT = {
    "words": [
        {"index": 0, "expected": "قل", "heard": "قل", "status": "ok", "similarity": 1.0},
        {
            "index": 1,
            "expected": "اعوذ",
            "heard": "اعود",
            "status": "unsure",
            "similarity": 0.75,
            "heard_start": 0.7,
            "heard_end": 1.4,
        },
        {"index": 2, "expected": "برب", "heard": None, "status": "missing", "similarity": 0.0},
        {"index": 3, "expected": "الناس", "heard": "الناس", "status": "ok", "similarity": 1.0},
    ],
    "extras": [],
    "flagged": [2],
    "unsure": [1],
    "heard_text": "قل أعود الناس",
    "duration_seconds": 6.1,
    "inference_seconds": 0.6,
}


# --- alignment ---------------------------------------------------------------------------


def test_normalize_is_diacritics_insensitive():
    assert normalize(AN_NAS_1) == ["قل", "اعوذ", "برب", "الناس"]
    assert normalize("قُلْ أَعُوذُ بِرَبِّ النَّاسِ") == normalize(AN_NAS_1)


def test_align_statuses():
    expected = normalize(AN_NAS_1)
    assert [v.status for v in align(expected, normalize("قل أعوذ برب الناس")).verdicts] == [
        "ok"
    ] * 4
    assert align(expected, normalize("قل اعوذ الناس")).flagged == [2]
    assert align(expected, normalize("قل أعوذ برب الفلق")).verdicts[3].status == "wrong"
    assert align(["اعوذ"], ["اعود"]).verdicts[0].status == "unsure"


def test_assessment_from_words_carries_the_timing_of_each_aligned_word():
    words = [(" قل", 0.3, 0.6), (" أعود", 0.7, 1.4), (" الناس", 2.1, 2.9)]
    result = assessment_from_words(AN_NAS_1, words)
    assert [w["status"] for w in result["words"]] == ["ok", "unsure", "missing", "ok"]
    assert result["words"][0]["heard_start"] == 0.3 and result["words"][0]["heard_end"] == 0.6
    assert result["words"][3]["heard_start"] == 2.1
    assert "heard_start" not in result["words"][2]
    # a transcribed token that splits into several words shares its timing
    split = assessment_from_words(AN_NAS_1, [("قل أعوذ", 0.0, 1.0), ("برب الناس", 1.0, 2.0)])
    assert [w["heard_start"] for w in split["words"]] == [0.0, 0.0, 1.0, 1.0]


def test_assessment_shape():
    result = assessment(AN_NAS_1, "قل اعود الناس")
    assert result["flagged"] == [2] and result["unsure"] == [1]
    assert result["words"][1] == {
        "index": 1,
        "expected": "اعوذ",
        "heard": "اعود",
        "status": "unsure",
        "similarity": 0.75,
    }


# --- request contract --------------------------------------------------------------------


@pytest.mark.parametrize(
    "change",
    [
        {"verse": "114-1"},
        {"audio_base64": "not base64!"},
        {"audio_base64": "a" * (recitation.MAX_AUDIO_BASE64 + 1)},
        {"mime": "video/webm"},
        {"expected_text": ""},
        {"student_id": "x"},
    ],
)
def test_request_rejects(change):
    with pytest.raises(ValidationError):
        recitation.RecitationRequest(**{**PAYLOAD, **change})


# --- gateway ------------------------------------------------------------------------------


@pytest.fixture
def setup(monkeypatch):
    calls = []

    async def context(settings, authorization, class_id=None):
        calls.append(("context", authorization))
        return CONTEXT

    monkeypatch.setattr(recitation, "resolve_student_context", context)
    return calls


def run(settings, payload=None):
    request = recitation.RecitationRequest(**(payload or PAYLOAD))
    return asyncio.run(recitation.assess_recitation(settings, "Bearer valid", request))


def test_relays_the_clip_without_identity_and_returns_the_checked_verdict(setup, monkeypatch):
    sent = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent["url"] = str(request.url)
        sent["body"] = json.loads(request.content)
        return httpx.Response(200, json=VERDICT)

    transport = httpx.MockTransport(handler)
    real = httpx.AsyncClient

    def client(**kwargs):
        return real(transport=transport, **kwargs)

    monkeypatch.setattr(recitation.httpx, "AsyncClient", client)
    result = run(Settings(recitation_service_url="http://recitation-service"))
    assert sent["url"] == "http://recitation-service/assess"
    assert set(sent["body"]) == {"expected_text", "mime", "audio_base64"}
    assert (
        result["contractVersion"] == 1 and result["verse"] == "114:1" and result["locale"] == "id"
    )
    assert result["flagged"] == [2] and result["unsure"] == [1]
    assert result["words"][2]["status"] == "missing"
    assert result["words"][1]["heard_start"] == 0.7 and result["words"][0]["heard_start"] is None
    assert "inference_seconds" not in result
    assert setup == [("context", "Bearer valid")]


def test_unavailable_without_a_worker_or_on_a_bad_answer(setup, monkeypatch):
    with pytest.raises(HTTPException) as exc:
        run(Settings())
    assert exc.value.status_code == 503

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"words": "nope"})

    real = httpx.AsyncClient
    monkeypatch.setattr(
        recitation.httpx,
        "AsyncClient",
        lambda **kw: real(transport=httpx.MockTransport(handler), **kw),
    )
    with pytest.raises(HTTPException) as exc:
        run(Settings(recitation_service_url="http://recitation-service"))
    assert exc.value.status_code == 503


def test_requires_a_current_student(monkeypatch):
    async def context(settings, authorization, class_id=None):
        raise HTTPException(403, "Student access required")

    monkeypatch.setattr(recitation, "resolve_student_context", context)
    with pytest.raises(HTTPException) as exc:
        run(Settings(recitation_service_url="http://recitation-service"))
    assert exc.value.status_code == 403


def test_endpoint_bounds_and_validates(monkeypatch):
    async def assess(settings, authorization, payload):
        return {"contractVersion": 1, "verse": payload.verse, "locale": payload.locale, **VERDICT}

    monkeypatch.setattr(main, "assess_recitation", assess)
    client = TestClient(main.app)
    ok = client.post("/api/ai/v1/student/recitation-assessment", json=PAYLOAD)
    assert ok.status_code == 200 and ok.headers["cache-control"] == "no-store"
    assert ok.json()["flagged"] == [2]
    assert (
        client.post(
            "/api/ai/v1/student/recitation-assessment", json={**PAYLOAD, "verse": "x"}
        ).status_code
        == 422
    )
    huge = {**PAYLOAD, "audio_base64": "A" * 3_000_001}
    assert client.post("/api/ai/v1/student/recitation-assessment", json=huge).status_code == 413

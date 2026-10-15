"""Pre-check of one recited verse for a signed-in Student (ADR 0009, Phase 1).

The Student screen sends the verse it showed and the clip it recorded. This module
confirms the caller is a current Student (the session is verified by its owners, the
same way as the other Student features), relays the clip to the in-cluster recitation
service and returns the word-by-word verdict. Nothing is stored, no identity is
forwarded, and the verdict is a pre-check: the ustadz decides the grade.
"""

from typing import Literal, Optional

import httpx
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.config import Settings
from app.student_context import resolve_student_context

MAX_AUDIO_BASE64 = 2_800_000  # about 2 MB of audio, 60-90 s of opus from a phone


class RecitationRequest(BaseModel):
    """What the screen showed and what it recorded. No identity by design."""

    model_config = ConfigDict(extra="forbid", strict=True)
    locale: Literal["en", "id"]
    verse: str = Field(pattern=r"^[0-9]{1,3}:[0-9]{1,3}$")
    expected_text: str = Field(min_length=1, max_length=600)
    mime: Literal["audio/webm", "audio/ogg", "audio/mp4", "audio/wav", "audio/mpeg"]
    audio_base64: str = Field(
        min_length=16, max_length=MAX_AUDIO_BASE64, pattern=r"^[A-Za-z0-9+/=]+$"
    )


class Word(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    index: int = Field(ge=0, le=200)
    expected: str = Field(min_length=1, max_length=64)
    heard: Optional[str] = Field(max_length=64)
    status: Literal["ok", "unsure", "wrong", "missing"]
    similarity: float = Field(ge=0.0, le=1.0)


class RecitationVerdict(BaseModel):
    """The worker's answer, checked before it reaches the screen."""

    model_config = ConfigDict(extra="ignore", strict=True)
    words: list[Word] = Field(max_length=200)
    extras: list[str] = Field(max_length=200)
    flagged: list[int] = Field(max_length=200)
    unsure: list[int] = Field(max_length=200)
    heard_text: str = Field(max_length=600)
    duration_seconds: float = Field(ge=0.0, le=600.0)


def unavailable() -> HTTPException:
    return HTTPException(503, "Recitation check unavailable")


async def assess_recitation(
    settings: Settings, authorization: Optional[str], payload: RecitationRequest
) -> dict:
    await resolve_student_context(settings, authorization)
    origin = settings.recitation_service_url
    if not origin:
        raise unavailable()
    body = {
        "expected_text": payload.expected_text,
        "mime": payload.mime,
        "audio_base64": payload.audio_base64,
    }
    try:
        async with httpx.AsyncClient(timeout=settings.recitation_timeout_seconds) as client:
            response = await client.post(f"{origin.rstrip('/')}/assess", json=body)
    except httpx.HTTPError as exc:
        raise unavailable() from exc
    if response.status_code in (413, 422):
        raise HTTPException(response.status_code, "Clip rejected")
    if response.status_code != 200:
        raise unavailable()
    try:
        verdict = RecitationVerdict.model_validate_json(response.content)
    except ValidationError as exc:
        raise unavailable() from exc
    return {
        "contractVersion": 1,
        "verse": payload.verse,
        "locale": payload.locale,
        **verdict.model_dump(),
    }

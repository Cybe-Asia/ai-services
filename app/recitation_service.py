"""The recitation worker: one clip in, a word-by-word verdict out.

Runs as its own Deployment from this image (``uvicorn app.recitation_service:app``),
on the node that holds the Qur'an-tuned Whisper model, reachable only inside the
namespace. The AI service authorises the Student and relays; this process never
sees a session token, a name or an identifier, and it keeps no audio.

Environment:
  RECITATION_MODEL_PATH   CTranslate2 model directory (required)
  RECITATION_THREADS      CPU threads for one transcription (default 8)
  RECITATION_WORKERS      clips transcribed at the same time (default 2)
  RECITATION_MAX_SECONDS  longest clip accepted (default 90)
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import os
import time
from contextlib import asynccontextmanager
from typing import Literal

import av
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.recitation_align import HeardWord, assessment_from_words

SAMPLE_RATE = 16000
MAX_BODY = 3_000_000
MIMES = ("audio/webm", "audio/ogg", "audio/mp4", "audio/wav", "audio/mpeg")


class AssessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    expected_text: str = Field(min_length=1, max_length=600)
    mime: Literal["audio/webm", "audio/ogg", "audio/mp4", "audio/wav", "audio/mpeg"]
    audio_base64: str = Field(min_length=16, max_length=2_800_000, pattern=r"^[A-Za-z0-9+/=]+$")


class State:
    model = None
    threads = int(os.environ.get("RECITATION_THREADS", "8"))
    workers = int(os.environ.get("RECITATION_WORKERS", "2"))
    max_seconds = float(os.environ.get("RECITATION_MAX_SECONDS", "90"))
    semaphore: asyncio.Semaphore | None = None


def decode(data: bytes) -> np.ndarray:
    """Any container PyAV can read, to 16 kHz mono float32."""
    out = []
    with av.open(io.BytesIO(data), mode="r") as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        for frame in container.decode(stream):
            for f in resampler.resample(frame):
                out.append(f.to_ndarray())
        for f in resampler.resample(None):
            out.append(f.to_ndarray())
    if not out:
        raise ValueError("no audio")
    return np.concatenate(out, axis=1).reshape(-1).astype(np.float32) / 32768.0


def transcribe(audio: np.ndarray) -> tuple[str, list[HeardWord]]:
    """The transcript and each word with its place in the clip, so the screen can replay a word."""
    segments, _ = State.model.transcribe(
        audio,
        language="ar",
        beam_size=5,
        vad_filter=False,
        condition_on_previous_text=False,
        word_timestamps=True,
    )
    words: list[HeardWord] = []
    for segment in segments:
        for word in segment.words or []:
            words.append((word.word, float(word.start), float(word.end)))
    return " ".join(w[0].strip() for w in words), words


@asynccontextmanager
async def lifespan(_: FastAPI):
    from faster_whisper import WhisperModel

    path = os.environ.get("RECITATION_MODEL_PATH")
    if not path:
        raise RuntimeError("RECITATION_MODEL_PATH is required")
    State.model = WhisperModel(path, device="cpu", compute_type="int8", cpu_threads=State.threads)
    State.semaphore = asyncio.Semaphore(State.workers)
    yield


app = FastAPI(title="recitation-service", lifespan=lifespan)


@app.get("/health")
async def health() -> JSONResponse:
    if State.model is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return JSONResponse({"status": "ok"})


@app.post("/assess")
async def assess(request: Request) -> JSONResponse:
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_BODY:
            raise HTTPException(413, "Clip too large")
        body.extend(chunk)
    try:
        payload = AssessRequest.model_validate_json(bytes(body))
    except ValidationError as exc:
        raise HTTPException(422, "Invalid clip") from exc
    try:
        data = base64.b64decode(payload.audio_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(422, "Invalid clip") from exc
    try:
        audio = await asyncio.to_thread(decode, data)
    except Exception as exc:  # any undecodable container
        raise HTTPException(422, "Undecodable clip") from exc
    seconds = len(audio) / SAMPLE_RATE
    if seconds < 0.5:
        raise HTTPException(422, "Clip too short")
    if seconds > State.max_seconds:
        raise HTTPException(413, "Clip too long")
    assert State.semaphore is not None
    started = time.monotonic()
    async with State.semaphore:
        heard, words = await asyncio.to_thread(transcribe, audio)
    result = assessment_from_words(payload.expected_text, words)
    result["heard_text"] = heard[:600]
    result["duration_seconds"] = round(seconds, 1)
    result["inference_seconds"] = round(time.monotonic() - started, 2)
    return JSONResponse(result, headers={"Cache-Control": "no-store"})

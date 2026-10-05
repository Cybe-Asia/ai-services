"""Bounded reads through Learning's current Auth/SIS authorization boundary."""

import json
from typing import Optional

import httpx
from fastapi import HTTPException

from app.config import Settings


async def learning_data(
    settings: Settings,
    authorization: Optional[str],
    path: str,
    params: Optional[dict[str, str]] = None,
) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Student session required")
    token = authorization[7:]
    if not token.strip() or len(token) > 16384 or any(c.isspace() for c in token):
        raise HTTPException(401, "Student session required")
    try:
        async with httpx.AsyncClient(
            timeout=settings.request_timeout_seconds, follow_redirects=False
        ) as client:
            async with client.stream(
                "GET",
                settings.learning_service_url.rstrip("/") + path,
                headers={
                    "Authorization": authorization,
                    "Accept": "application/json",
                    "Accept-Encoding": "identity",
                },
                params=params,
            ) as response:
                if response.status_code in {401, 403, 404}:
                    raise HTTPException(response.status_code, "Student source access denied")
                if response.status_code != 200:
                    raise HTTPException(503, "Student source unavailable")
                if (
                    response.headers.get("content-encoding", "identity").strip().casefold()
                    != "identity"
                ):
                    raise HTTPException(503, "Student source unavailable")
                body = bytearray()
                async for chunk in response.aiter_bytes(chunk_size=16384):
                    if len(body) + len(chunk) > 1048576:
                        raise HTTPException(503, "Student source unavailable")
                    body.extend(chunk)
                payload = json.loads(body)
                if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
                    raise HTTPException(503, "Student source unavailable")
                return payload["data"]
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503, "Student source unavailable") from exc

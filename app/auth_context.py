from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Optional

import httpx
from fastapi import HTTPException, status

from app.config import Settings
from app.schemas import ActorRole


@dataclass(frozen=True)
class AdminContext:
    owner_id: str
    actor_role: ActorRole


async def resolve_admin_context(
    settings: Settings,
    authorization: Optional[str],
) -> AdminContext:
    if not _has_bearer_token(authorization):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token required",
        )

    url = _join_url(settings.admission_service_url, "/api/leads/v1/me/roles")
    try:
        body = await _get_roles(url, authorization)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED
            if exc.response.status_code in {401, 403}
            else status.HTTP_502_BAD_GATEWAY,
            detail="Admin session validation failed",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Admin session validation unavailable",
        ) from exc

    data = _as_dict(body.get("data"))
    email = data.get("email")
    if not isinstance(email, str) or not email.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin session validation failed",
        )

    is_owner = data.get("isOwner") is True
    is_admin = data.get("isAdmin") is True
    if not (is_owner or is_admin):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )

    return AdminContext(
        owner_id=_stable_owner_id(email),
        actor_role=ActorRole.owner if is_owner else ActorRole.admin,
    )


async def _get_roles(url: str, authorization: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(
            url,
            headers={"accept": "application/json", "authorization": authorization},
        )
        response.raise_for_status()
        body = response.json()
    return body if isinstance(body, dict) else {}


def _stable_owner_id(email: str) -> str:
    return sha256(email.casefold().strip().encode("utf-8")).hexdigest()


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}{path}"


def _has_bearer_token(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.casefold().startswith("bearer ") and len(value.split(" ", 1)[1].strip()) > 0

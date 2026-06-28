import asyncio

import pytest
from fastapi import HTTPException

from app import auth_context
from app.auth_context import resolve_admin_context
from app.config import Settings
from app.schemas import ActorRole


def test_resolve_admin_context_allows_graph_staff_role(monkeypatch) -> None:
    async def fake_get_roles(url: str, authorization: str) -> dict:
        assert url == "http://admission-service/api/leads/v1/me/roles"
        assert authorization == "Bearer staff-token"
        return {
            "data": {
                "email": "staff@example.com",
                "roles": ["admissions_staff"],
                "isAdmin": False,
                "isOwner": False,
            }
        }

    monkeypatch.setattr(auth_context, "_get_roles", fake_get_roles)

    context = asyncio.run(resolve_admin_context(Settings(), "Bearer staff-token"))

    assert context.actor_role == ActorRole.admin


def test_resolve_admin_context_rejects_non_staff_roles(monkeypatch) -> None:
    async def fake_get_roles(url: str, authorization: str) -> dict:
        return {
            "data": {
                "email": "parent@example.com",
                "roles": [],
                "isAdmin": False,
                "isOwner": False,
            }
        }

    monkeypatch.setattr(auth_context, "_get_roles", fake_get_roles)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(resolve_admin_context(Settings(), "Bearer parent-token"))

    assert getattr(exc_info.value, "status_code", None) == 403

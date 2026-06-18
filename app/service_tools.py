from dataclasses import dataclass
from typing import Any, Optional

import httpx

from app.config import Settings
from app.schemas import ActorRole, ChatRequest, SourceRef, ToolCallRef


@dataclass
class ToolAnswer:
    answer: str
    sources: list[SourceRef]
    tool_calls: list[ToolCallRef]


async def answer_from_school_tools(
    payload: ChatRequest,
    settings: Settings,
    authorization: Optional[str],
) -> Optional[ToolAnswer]:
    if payload.actor_role not in {ActorRole.owner, ActorRole.admin}:
        return None

    lowered = payload.message.casefold()
    if _asks_for_eoi_count(lowered):
        return await _admission_eoi_count(settings, authorization)
    if _asks_for_payment_review_count(lowered):
        return await _payment_review_count(settings, authorization, lowered)
    return None


async def _admission_eoi_count(settings: Settings, authorization: Optional[str]) -> ToolAnswer:
    tool_name = "admission.admin_leads_count"
    if not _has_bearer_token(authorization):
        return _auth_required(tool_name, "data EOI")

    url = _join_url(settings.admission_service_url, "/api/leads/v1/admin/leads")
    try:
        body = await _get_json(url, authorization, {"limit": "1", "offset": "0"})
        total = _extract_total(body)
    except Exception:
        return _tool_failed(
            tool_name,
            "Saya belum bisa mengambil total EOI dari admission-service.",
        )

    return ToolAnswer(
        answer=f"Ada {total} EOI terdaftar di admission-service.",
        sources=[SourceRef(kind="service", title="admission-service admin leads", reference=None)],
        tool_calls=[ToolCallRef(name=tool_name, status="ok")],
    )


async def _payment_review_count(
    settings: Settings,
    authorization: Optional[str],
    lowered_message: str,
) -> ToolAnswer:
    tool_name = "payment.admin_review_count"
    if not _has_bearer_token(authorization):
        return _auth_required(tool_name, "data pembayaran")

    status = _payment_status_from_message(lowered_message)
    url = _join_url(settings.payment_service_url, "/api/v1/payments/admin/reviews")
    try:
        body = await _get_json(
            url,
            authorization,
            {"status": status, "limit": "1", "offset": "0"},
        )
        total = _extract_total(body)
    except Exception:
        return _tool_failed(
            tool_name,
            "Saya belum bisa mengambil data pembayaran dari payment-service.",
        )

    label = _payment_status_label(status)
    return ToolAnswer(
        answer=f"Ada {total} pembayaran {label} di payment-service.",
        sources=[SourceRef(kind="service", title="payment-service admin reviews", reference=None)],
        tool_calls=[ToolCallRef(name=tool_name, status="ok")],
    )


async def _get_json(
    url: str,
    authorization: Optional[str],
    params: dict[str, str],
) -> dict[str, Any]:
    headers = {"accept": "application/json"}
    if authorization:
        headers["authorization"] = authorization

    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(url, headers=headers, params=params)
        response.raise_for_status()
        body = response.json()

    return body if isinstance(body, dict) else {}


def _extract_total(body: dict[str, Any]) -> int:
    data = body.get("data")
    if not isinstance(data, dict):
        raise ValueError("service response missing data")
    total = data.get("total")
    if not isinstance(total, int):
        raise ValueError("service response missing total")
    return total


def _asks_for_eoi_count(message: str) -> bool:
    count_terms = ("berapa", "total", "jumlah", "count")
    eoi_terms = ("eoi", "email", "lead", "pendaftar", "terdaftar")
    return any(term in message for term in count_terms) and any(
        term in message for term in eoi_terms
    )


def _asks_for_payment_review_count(message: str) -> bool:
    count_terms = ("berapa", "total", "jumlah", "count")
    payment_terms = ("payment", "pembayaran", "tagihan", "invoice", "manual transfer")
    return any(term in message for term in count_terms) and any(
        term in message for term in payment_terms
    )


def _payment_status_from_message(message: str) -> str:
    if "paid" in message or "lunas" in message or "approved" in message:
        return "paid"
    if "rejected" in message or "ditolak" in message:
        return "rejected"
    if "underpaid" in message or "kurang bayar" in message:
        return "underpaid"
    return "pending_verification"


def _payment_status_label(status: str) -> str:
    if status == "pending_verification":
        return "yang menunggu verifikasi"
    if status == "paid":
        return "yang sudah lunas"
    if status == "rejected":
        return "yang ditolak"
    if status == "underpaid":
        return "yang kurang bayar"
    return f"dengan status {status}"


def _auth_required(tool_name: str, subject: str) -> ToolAnswer:
    return ToolAnswer(
        answer=f"Saya perlu token owner/admin yang valid untuk mengambil {subject}.",
        sources=[],
        tool_calls=[ToolCallRef(name=tool_name, status="auth_required")],
    )


def _tool_failed(tool_name: str, answer: str) -> ToolAnswer:
    return ToolAnswer(
        answer=answer,
        sources=[],
        tool_calls=[ToolCallRef(name=tool_name, status="error")],
    )


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}{path}"


def _has_bearer_token(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.casefold().startswith("bearer ") and len(value.split(" ", 1)[1].strip()) > 0

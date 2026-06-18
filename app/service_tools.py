import json
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from app.config import Settings
from app.llm_client import LlmClient
from app.schemas import ActorRole, ChatRequest, SourceRef, ToolCallRef

INTENT_NONE = "none"
INTENT_ADMISSION_EOI_COUNT = "admission_eoi_count"
INTENT_ADMISSION_LEADS_LIST = "admission_leads_list"
INTENT_PAYMENT_REVIEW_COUNT = "payment_review_count"
PAYMENT_STATUSES = {"pending_verification", "paid", "rejected", "underpaid"}


@dataclass
class ToolAnswer:
    answer: str
    sources: list[SourceRef]
    tool_calls: list[ToolCallRef]


@dataclass(frozen=True)
class ToolIntent:
    name: str
    payment_status: Optional[str] = None


async def answer_from_school_tools(
    payload: ChatRequest,
    settings: Settings,
    authorization: Optional[str],
) -> Optional[ToolAnswer]:
    if payload.actor_role not in {ActorRole.owner, ActorRole.admin}:
        return None

    lowered = payload.message.casefold()
    intent = _deterministic_tool_intent(lowered)
    if intent.name == INTENT_NONE:
        intent = _contextual_tool_intent(payload, lowered)
    if intent.name == INTENT_NONE:
        intent = await _classify_tool_intent(payload.message, settings)

    if intent.name == INTENT_ADMISSION_EOI_COUNT:
        return await _admission_eoi_count(settings, authorization)
    if intent.name == INTENT_ADMISSION_LEADS_LIST:
        return await _admission_leads_list(settings, authorization)
    if intent.name == INTENT_PAYMENT_REVIEW_COUNT:
        return await _payment_review_count(
            settings,
            authorization,
            lowered,
            intent.payment_status,
        )
    return None


def _deterministic_tool_intent(message: str) -> ToolIntent:
    if _asks_for_eoi_count(message):
        return ToolIntent(INTENT_ADMISSION_EOI_COUNT)
    if _asks_for_admission_lead_identity(message):
        return ToolIntent(INTENT_ADMISSION_LEADS_LIST)
    if _asks_for_payment_review_count(message):
        return ToolIntent(INTENT_PAYMENT_REVIEW_COUNT, _payment_status_from_message(message))
    return ToolIntent(INTENT_NONE)


def _contextual_tool_intent(payload: ChatRequest, lowered_message: str) -> ToolIntent:
    if _asks_for_contextual_lead_identity(lowered_message) and _history_has_tool_call(
        payload,
        "admission.admin_leads_count",
    ):
        return ToolIntent(INTENT_ADMISSION_LEADS_LIST)
    return ToolIntent(INTENT_NONE)


async def _classify_tool_intent(message: str, settings: Settings) -> ToolIntent:
    if settings.ai_provider_base_url is None:
        return ToolIntent(INTENT_NONE)

    system_prompt = (
        "Return JSON only. Classify Digital Schools admin questions into tool intent. "
        "If user asks count/total/jumlah/berapa of calon keluarga, families, applicants, "
        "registrations, leads, EOI, or registered emails, use admission_eoi_count. "
        "If user asks count of transactions, finance checks, payments, invoices, or manual "
        "transfers, use payment_review_count. Otherwise use none. "
        'Example: "berapa calon keluarga masuk?" => {"intent":"admission_eoi_count"}. '
        'Example: "berapa transaksi yang masih perlu dicek finance?" => '
        '{"intent":"payment_review_count","paymentStatus":"pending_verification"}.'
    )

    try:
        raw_intent = await LlmClient(settings).complete(
            system_prompt=system_prompt,
            user_message=message,
            temperature=0.0,
            max_tokens=32,
            timeout_seconds=60.0,
            response_format={"type": "json_object"},
        )
    except Exception:
        return ToolIntent(INTENT_NONE)

    return _parse_tool_intent(raw_intent)


def _parse_tool_intent(raw_intent: Optional[str]) -> ToolIntent:
    if not raw_intent:
        return ToolIntent(INTENT_NONE)

    start = raw_intent.find("{")
    end = raw_intent.rfind("}")
    if start < 0 or end <= start:
        return ToolIntent(INTENT_NONE)

    try:
        body = json.loads(raw_intent[start : end + 1])
    except json.JSONDecodeError:
        return ToolIntent(INTENT_NONE)

    if not isinstance(body, dict):
        return ToolIntent(INTENT_NONE)

    intent = _normalize_intent(body.get("intent"))
    if intent == INTENT_ADMISSION_EOI_COUNT:
        return ToolIntent(INTENT_ADMISSION_EOI_COUNT)
    if intent == INTENT_ADMISSION_LEADS_LIST:
        return ToolIntent(INTENT_ADMISSION_LEADS_LIST)
    if intent == INTENT_PAYMENT_REVIEW_COUNT:
        payment_status = _normalize_payment_status(
            body.get("paymentStatus") or body.get("payment_status") or body.get("status")
        )
        return ToolIntent(INTENT_PAYMENT_REVIEW_COUNT, payment_status)
    return ToolIntent(INTENT_NONE)


def _normalize_intent(value: Any) -> str:
    if not isinstance(value, str):
        return INTENT_NONE

    normalized = value.casefold().strip().replace("-", "_").replace(".", "_")
    aliases = {
        INTENT_ADMISSION_EOI_COUNT: INTENT_ADMISSION_EOI_COUNT,
        "admission_admin_leads_count": INTENT_ADMISSION_EOI_COUNT,
        "admission_leads_count": INTENT_ADMISSION_EOI_COUNT,
        "eoi_count": INTENT_ADMISSION_EOI_COUNT,
        "lead_count": INTENT_ADMISSION_EOI_COUNT,
        INTENT_ADMISSION_LEADS_LIST: INTENT_ADMISSION_LEADS_LIST,
        "admission_admin_leads_list": INTENT_ADMISSION_LEADS_LIST,
        "lead_list": INTENT_ADMISSION_LEADS_LIST,
        "eoi_list": INTENT_ADMISSION_LEADS_LIST,
        INTENT_PAYMENT_REVIEW_COUNT: INTENT_PAYMENT_REVIEW_COUNT,
        "payment_admin_review_count": INTENT_PAYMENT_REVIEW_COUNT,
        "payment_count": INTENT_PAYMENT_REVIEW_COUNT,
        "payment_pending_count": INTENT_PAYMENT_REVIEW_COUNT,
        INTENT_NONE: INTENT_NONE,
    }
    return aliases.get(normalized, INTENT_NONE)


def _normalize_payment_status(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None

    normalized = value.casefold().strip().replace("-", "_").replace(" ", "_")
    aliases = {
        "pending": "pending_verification",
        "pending_review": "pending_verification",
        "pending_verification": "pending_verification",
        "waiting_verification": "pending_verification",
        "needs_verification": "pending_verification",
        "need_verification": "pending_verification",
        "menunggu_verifikasi": "pending_verification",
        "verified": "paid",
        "approved": "paid",
        "lunas": "paid",
        "paid": "paid",
        "rejected": "rejected",
        "ditolak": "rejected",
        "underpaid": "underpaid",
        "kurang_bayar": "underpaid",
    }
    return aliases.get(normalized) if normalized not in PAYMENT_STATUSES else normalized


async def _admission_eoi_count(settings: Settings, authorization: Optional[str]) -> ToolAnswer:
    tool_name = "admission.admin_leads_count"
    if not _has_bearer_token(authorization):
        return _auth_required(tool_name, "data EOI")

    url = _join_url(settings.admission_service_url, "/api/leads/v1/admin/leads")
    try:
        body = await _get_json(url, authorization, {"limit": "1", "offset": "0"})
        total = _extract_total(body)
    except httpx.HTTPStatusError as exc:
        auth_error = _auth_status_tool_error(tool_name, exc.response.status_code)
        if auth_error is not None:
            return auth_error
        return _tool_failed(
            tool_name,
            "Saya belum bisa mengambil total EOI dari admission-service.",
        )
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


async def _admission_leads_list(settings: Settings, authorization: Optional[str]) -> ToolAnswer:
    tool_name = "admission.admin_leads_list"
    if not _has_bearer_token(authorization):
        return _auth_required(tool_name, "daftar EOI")

    url = _join_url(settings.admission_service_url, "/api/leads/v1/admin/leads")
    try:
        body = await _get_json(url, authorization, {"limit": "5", "offset": "0"})
        rows, total = _extract_lead_rows(body)
    except httpx.HTTPStatusError as exc:
        auth_error = _auth_status_tool_error(tool_name, exc.response.status_code)
        if auth_error is not None:
            return auth_error
        return _tool_failed(
            tool_name,
            "Saya belum bisa mengambil daftar EOI dari admission-service.",
        )
    except Exception:
        return _tool_failed(
            tool_name,
            "Saya belum bisa mengambil daftar EOI dari admission-service.",
        )

    if total == 0 or not rows:
        answer = "Belum ada EOI terdaftar di admission-service."
    else:
        shown = min(len(rows), 5)
        lines = [f"Ada {total} EOI. Saya tampilkan {shown} yang terbaru:"]
        for index, row in enumerate(rows[:shown], start=1):
            name = _clean_text(row.get("parentName")) or "Nama belum tersedia"
            email = _clean_text(row.get("email"))
            school = _clean_text(row.get("school"))
            lead_status = _clean_text(row.get("leadStatus"))
            details = []
            if email:
                details.append(email)
            if school:
                details.append(f"sekolah: {school}")
            if lead_status:
                details.append(f"status: {lead_status}")
            suffix = f" ({'; '.join(details)})" if details else ""
            lines.append(f"{index}. {name}{suffix}")
        answer = "\n".join(lines)

    return ToolAnswer(
        answer=answer,
        sources=[SourceRef(kind="service", title="admission-service admin leads", reference=None)],
        tool_calls=[ToolCallRef(name=tool_name, status="ok")],
    )


async def _payment_review_count(
    settings: Settings,
    authorization: Optional[str],
    lowered_message: str,
    status_override: Optional[str] = None,
) -> ToolAnswer:
    tool_name = "payment.admin_review_count"
    if not _has_bearer_token(authorization):
        return _auth_required(tool_name, "data pembayaran")

    status = status_override or _payment_status_from_message(lowered_message)
    url = _join_url(settings.payment_service_url, "/api/v1/payments/admin/reviews")
    try:
        body = await _get_json(
            url,
            authorization,
            {"status": status, "limit": "1", "offset": "0"},
        )
        total = _extract_total(body)
    except httpx.HTTPStatusError as exc:
        auth_error = _auth_status_tool_error(tool_name, exc.response.status_code)
        if auth_error is not None:
            return auth_error
        return _tool_failed(
            tool_name,
            "Saya belum bisa mengambil data pembayaran dari payment-service.",
        )
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


def _extract_lead_rows(body: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    data = body.get("data")
    if not isinstance(data, dict):
        raise ValueError("service response missing data")
    rows = data.get("rows")
    if not isinstance(rows, list):
        raise ValueError("service response missing rows")
    total = data.get("total")
    if not isinstance(total, int):
        raise ValueError("service response missing total")
    return [row for row in rows if isinstance(row, dict)], total


def _asks_for_eoi_count(message: str) -> bool:
    count_terms = ("berapa", "total", "jumlah", "count", "how many", "number of")
    eoi_terms = (
        "eoi",
        "email",
        "lead",
        "pendaftar",
        "terdaftar",
        "daftar",
        "mendaftar",
        "registrasi",
        "registration",
        "registered",
        "enquiry",
        "enquiries",
        "inquiry",
        "inquiries",
    )
    return any(term in message for term in count_terms) and any(
        term in message for term in eoi_terms
    )


def _asks_for_admission_lead_identity(message: str) -> bool:
    identity_terms = ("siapa", "nama", "email", "who", "name")
    lead_terms = (
        "eoi",
        "lead",
        "pendaftar",
        "daftar",
        "mendaftar",
        "registrasi",
        "registration",
        "registered",
        "enquiry",
        "enquiries",
        "inquiry",
        "inquiries",
    )
    return any(term in message for term in identity_terms) and any(
        term in message for term in lead_terms
    )


def _asks_for_contextual_lead_identity(message: str) -> bool:
    followup_terms = (
        "siapa orang itu",
        "siapa itu",
        "orang itu siapa",
        "orangnya siapa",
        "siapa saja",
        "siapa aja",
        "nama mereka",
        "tampilkan orangnya",
        "show them",
        "list them",
        "who is it",
        "who are they",
        "who registered",
    )
    return any(term in message for term in followup_terms)


def _history_has_tool_call(payload: ChatRequest, tool_name: str) -> bool:
    for turn in payload.history[-10:]:
        if any(tool.name == tool_name and tool.status == "ok" for tool in turn.tool_calls):
            return True
    return False


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())


def _asks_for_payment_review_count(message: str) -> bool:
    count_terms = ("berapa", "total", "jumlah", "count", "how many", "number of")
    payment_terms = ("payment", "pembayaran", "tagihan", "invoice", "manual transfer")
    return any(term in message for term in count_terms) and any(
        term in message for term in payment_terms
    )


def _payment_status_from_message(message: str) -> str:
    if "underpaid" in message or "kurang bayar" in message:
        return "underpaid"
    if "rejected" in message or "ditolak" in message:
        return "rejected"
    if "paid" in message or "lunas" in message or "approved" in message:
        return "paid"
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


def _auth_status_tool_error(tool_name: str, status_code: int) -> Optional[ToolAnswer]:
    if status_code == 401:
        return ToolAnswer(
            answer=(
                "Sesi admin tidak valid atau sudah kedaluwarsa. "
                "Silakan login ulang sebelum meminta data operasional."
            ),
            sources=[],
            tool_calls=[ToolCallRef(name=tool_name, status="auth_required")],
        )
    if status_code == 403:
        return ToolAnswer(
            answer=(
                "Akun ini belum punya akses admin untuk mengambil data operasional. "
                "Pastikan email akun terdaftar di allowlist admin."
            ),
            sources=[],
            tool_calls=[ToolCallRef(name=tool_name, status="forbidden")],
        )
    return None


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}{path}"


def _has_bearer_token(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.casefold().startswith("bearer ") and len(value.split(" ", 1)[1].strip()) > 0

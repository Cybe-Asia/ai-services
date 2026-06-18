import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

import httpx

from app.config import Settings
from app.llm_client import LlmClient
from app.schemas import ActorRole, ChatRequest, SourceRef, ToolCallRef

INTENT_NONE = "none"
INTENT_ADMISSION_EOI_COUNT = "admission_eoi_count"
INTENT_ADMISSION_LEADS_LIST = "admission_leads_list"
INTENT_ADMISSION_LEAD_CHILD_COUNT = "admission_lead_child_count"
INTENT_ADMISSION_LEAD_STUDENTS_LIST = "admission_lead_students_list"
INTENT_PAYMENT_REVIEW_COUNT = "payment_review_count"
INTENT_PAYMENT_APPLICATION_FEE = "payment_application_fee_quote"
PAYMENT_STATUSES = {"pending_verification", "paid", "rejected", "underpaid"}
DEFAULT_SCHOOL_CODES = ("IIHS", "IISS", "IIBS")
SCHOOL_TIME_ZONE = ZoneInfo("Asia/Jakarta")
ID_MONTH_NAMES = {
    1: "Januari",
    2: "Februari",
    3: "Maret",
    4: "April",
    5: "Mei",
    6: "Juni",
    7: "Juli",
    8: "Agustus",
    9: "September",
    10: "Oktober",
    11: "November",
    12: "Desember",
}
MONTH_ALIASES = {
    "jan": 1,
    "januari": 1,
    "january": 1,
    "feb": 2,
    "februari": 2,
    "february": 2,
    "mar": 3,
    "maret": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "mei": 5,
    "may": 5,
    "jun": 6,
    "juni": 6,
    "june": 6,
    "jul": 7,
    "juli": 7,
    "july": 7,
    "agu": 8,
    "agustus": 8,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "okt": 10,
    "oktober": 10,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "des": 12,
    "desember": 12,
    "dec": 12,
    "december": 12,
}


@dataclass
class ToolAnswer:
    answer: str
    sources: list[SourceRef]
    tool_calls: list[ToolCallRef]


@dataclass(frozen=True)
class ToolIntent:
    name: str
    payment_status: Optional[str] = None


@dataclass(frozen=True)
class DateRange:
    start: datetime
    end: datetime
    label: str


async def answer_from_school_tools(
    payload: ChatRequest,
    settings: Settings,
    authorization: Optional[str],
) -> Optional[ToolAnswer]:
    if payload.actor_role not in {ActorRole.owner, ActorRole.admin}:
        return None

    lowered = payload.message.casefold()
    date_range = _date_range_from_message(lowered)
    intent = _deterministic_tool_intent(lowered)
    if intent.name == INTENT_NONE:
        intent = _contextual_tool_intent(payload, lowered, date_range)
    if intent.name == INTENT_NONE:
        intent = await _classify_tool_intent(payload.message, settings)

    if intent.name == INTENT_ADMISSION_EOI_COUNT:
        return await _admission_eoi_count(settings, authorization, date_range)
    if intent.name == INTENT_ADMISSION_LEADS_LIST:
        return await _admission_leads_list(settings, authorization, date_range)
    if intent.name == INTENT_ADMISSION_LEAD_CHILD_COUNT:
        return await _admission_lead_child_count(settings, authorization, payload, lowered)
    if intent.name == INTENT_ADMISSION_LEAD_STUDENTS_LIST:
        return await _admission_lead_students_list(settings, authorization, payload, lowered)
    if intent.name == INTENT_PAYMENT_REVIEW_COUNT:
        return await _payment_review_count(
            settings,
            authorization,
            lowered,
            intent.payment_status,
        )
    if intent.name == INTENT_PAYMENT_APPLICATION_FEE:
        return await _payment_application_fee_quote(settings, authorization, lowered)
    return None


def _deterministic_tool_intent(message: str) -> ToolIntent:
    if _asks_for_lead_child_count(message):
        return ToolIntent(INTENT_ADMISSION_LEAD_CHILD_COUNT)
    if _asks_for_lead_child_identity(message):
        return ToolIntent(INTENT_ADMISSION_LEAD_STUDENTS_LIST)
    if _asks_for_payment_application_fee(message):
        return ToolIntent(INTENT_PAYMENT_APPLICATION_FEE)
    if _asks_for_eoi_count(message):
        return ToolIntent(INTENT_ADMISSION_EOI_COUNT)
    if _asks_for_admission_lead_identity(message):
        return ToolIntent(INTENT_ADMISSION_LEADS_LIST)
    if _asks_for_payment_review_count(message):
        return ToolIntent(INTENT_PAYMENT_REVIEW_COUNT, _payment_status_from_message(message))
    return ToolIntent(INTENT_NONE)


def _contextual_tool_intent(
    payload: ChatRequest,
    lowered_message: str,
    date_range: Optional[DateRange],
) -> ToolIntent:
    if date_range is not None and _asks_for_contextual_period_followup(lowered_message):
        last_tool = _last_ok_tool_call(payload)
        if last_tool == "admission.admin_leads_list":
            return ToolIntent(INTENT_ADMISSION_LEADS_LIST)
        if last_tool == "admission.admin_leads_count":
            return ToolIntent(INTENT_ADMISSION_EOI_COUNT)

    if _asks_for_contextual_lead_child_identity(lowered_message) and _history_has_tool_call(
        payload,
        "admission.admin_lead_child_count",
    ):
        return ToolIntent(INTENT_ADMISSION_LEAD_STUDENTS_LIST)
    if (
        _asks_for_contextual_lead_identity(lowered_message)
        or _asks_for_contextual_lead_detail(lowered_message)
    ) and (
        _history_has_tool_call(payload, "admission.admin_leads_count")
        or _history_has_tool_call(payload, "admission.admin_leads_list")
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
        "If user asks how many children/students a specific lead or EOI parent registered, "
        "use admission_lead_child_count. "
        "If user asks who/name/list of children/students for a specific lead, "
        "use admission_lead_students_list. "
        "If user asks count of transactions, finance checks, payments, invoices, or manual "
        "transfers, use payment_review_count. "
        "If user asks admission/application price, biaya pendaftaran, fee, tariff, or cost, "
        "use payment_application_fee_quote. Otherwise use none. "
        'Example: "berapa calon keluarga masuk?" => {"intent":"admission_eoi_count"}. '
        'Example: "Arief Nugraha di EOI daftarin berapa anak?" => '
        '{"intent":"admission_lead_child_count"}. '
        'Example: "berapa transaksi yang masih perlu dicek finance?" => '
        '{"intent":"payment_review_count","paymentStatus":"pending_verification"}. '
        'Example: "berapa harga admissions?" => {"intent":"payment_application_fee_quote"}.'
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
    if intent == INTENT_ADMISSION_LEAD_CHILD_COUNT:
        return ToolIntent(INTENT_ADMISSION_LEAD_CHILD_COUNT)
    if intent == INTENT_ADMISSION_LEAD_STUDENTS_LIST:
        return ToolIntent(INTENT_ADMISSION_LEAD_STUDENTS_LIST)
    if intent == INTENT_PAYMENT_REVIEW_COUNT:
        payment_status = _normalize_payment_status(
            body.get("paymentStatus") or body.get("payment_status") or body.get("status")
        )
        return ToolIntent(INTENT_PAYMENT_REVIEW_COUNT, payment_status)
    if intent == INTENT_PAYMENT_APPLICATION_FEE:
        return ToolIntent(INTENT_PAYMENT_APPLICATION_FEE)
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
        INTENT_ADMISSION_LEAD_CHILD_COUNT: INTENT_ADMISSION_LEAD_CHILD_COUNT,
        "admission_lead_student_count": INTENT_ADMISSION_LEAD_CHILD_COUNT,
        "lead_child_count": INTENT_ADMISSION_LEAD_CHILD_COUNT,
        "lead_student_count": INTENT_ADMISSION_LEAD_CHILD_COUNT,
        INTENT_ADMISSION_LEAD_STUDENTS_LIST: INTENT_ADMISSION_LEAD_STUDENTS_LIST,
        "admission_lead_child_list": INTENT_ADMISSION_LEAD_STUDENTS_LIST,
        "lead_child_list": INTENT_ADMISSION_LEAD_STUDENTS_LIST,
        "lead_student_list": INTENT_ADMISSION_LEAD_STUDENTS_LIST,
        INTENT_PAYMENT_REVIEW_COUNT: INTENT_PAYMENT_REVIEW_COUNT,
        "payment_admin_review_count": INTENT_PAYMENT_REVIEW_COUNT,
        "payment_count": INTENT_PAYMENT_REVIEW_COUNT,
        "payment_pending_count": INTENT_PAYMENT_REVIEW_COUNT,
        INTENT_PAYMENT_APPLICATION_FEE: INTENT_PAYMENT_APPLICATION_FEE,
        "payment_fee_quote": INTENT_PAYMENT_APPLICATION_FEE,
        "application_fee_quote": INTENT_PAYMENT_APPLICATION_FEE,
        "admission_fee_quote": INTENT_PAYMENT_APPLICATION_FEE,
        "admission_price_quote": INTENT_PAYMENT_APPLICATION_FEE,
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


async def _admission_eoi_count(
    settings: Settings,
    authorization: Optional[str],
    date_range: Optional[DateRange] = None,
) -> ToolAnswer:
    tool_name = "admission.admin_leads_count"
    if not _has_bearer_token(authorization):
        return _auth_required(tool_name, "data EOI")

    url = _join_url(settings.admission_service_url, "/api/leads/v1/admin/leads")
    try:
        body = await _get_json(
            url,
            authorization,
            _lead_list_params(limit=1, offset=0, date_range=date_range),
        )
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

    scope = f" {date_range.label}" if date_range is not None else ""
    return ToolAnswer(
        answer=f"Ada {total} EOI terdaftar{scope} di admission-service.",
        sources=[SourceRef(kind="service", title="admission-service admin leads", reference=None)],
        tool_calls=[ToolCallRef(name=tool_name, status="ok")],
    )


async def _admission_leads_list(
    settings: Settings,
    authorization: Optional[str],
    date_range: Optional[DateRange] = None,
) -> ToolAnswer:
    tool_name = "admission.admin_leads_list"
    if not _has_bearer_token(authorization):
        return _auth_required(tool_name, "daftar EOI")

    url = _join_url(settings.admission_service_url, "/api/leads/v1/admin/leads")
    try:
        body = await _get_json(
            url,
            authorization,
            _lead_list_params(limit=5, offset=0, date_range=date_range),
        )
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

    scope = f" {date_range.label}" if date_range is not None else ""
    if total == 0 or not rows:
        answer = f"Belum ada EOI terdaftar{scope} di admission-service."
    else:
        shown = min(len(rows), 5)
        lines = [f"Ada {total} EOI{scope}. Saya tampilkan {shown} yang terbaru:"]
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


async def _admission_lead_child_count(
    settings: Settings,
    authorization: Optional[str],
    payload: ChatRequest,
    lowered_message: str,
) -> ToolAnswer:
    tool_name = "admission.admin_lead_child_count"
    if not _has_bearer_token(authorization):
        return _auth_required(tool_name, "jumlah anak di EOI")

    query = _lead_search_query_from_context(payload, lowered_message)
    if not query:
        return _tool_failed(
            tool_name,
            "Saya perlu nama parent atau email EOI untuk mengecek jumlah anaknya.",
        )

    try:
        row, total = await _find_lead_row(settings, authorization, query)
    except httpx.HTTPStatusError as exc:
        auth_error = _auth_status_tool_error(tool_name, exc.response.status_code)
        if auth_error is not None:
            return auth_error
        return _tool_failed(
            tool_name,
            "Saya belum bisa mengambil detail EOI dari admission-service.",
        )
    except Exception:
        return _tool_failed(
            tool_name,
            "Saya belum bisa mengambil detail EOI dari admission-service.",
        )

    if row is None:
        answer = f"Saya belum menemukan EOI untuk '{query}' di admission-service."
    else:
        name = _clean_text(row.get("parentName")) or query
        count = _safe_int(row.get("applicantCount"))
        school = _clean_text(row.get("school"))
        status = _clean_text(row.get("leadStatus"))
        suffix = []
        if school:
            suffix.append(f"sekolah: {school}")
        if status:
            suffix.append(f"status: {status}")
        if total > 1:
            suffix.append(f"ada {total} hasil; saya pakai hasil teratas")
        detail = f" ({'; '.join(suffix)})" if suffix else ""
        if count == 0:
            answer = f"{name} belum punya data anak yang tersimpan di aplikasi{detail}."
        else:
            answer = f"{name} mendaftarkan {count} anak di aplikasi{detail}."

    return ToolAnswer(
        answer=answer,
        sources=[
            SourceRef(kind="service", title="admission-service admin lead detail", reference=None)
        ],
        tool_calls=[ToolCallRef(name=tool_name, status="ok" if row is not None else "not_found")],
    )


async def _admission_lead_students_list(
    settings: Settings,
    authorization: Optional[str],
    payload: ChatRequest,
    lowered_message: str,
) -> ToolAnswer:
    tool_name = "admission.admin_lead_students_list"
    if not _has_bearer_token(authorization):
        return _auth_required(tool_name, "daftar anak di EOI")

    query = _lead_search_query_from_context(payload, lowered_message)
    if not query:
        return _tool_failed(
            tool_name,
            "Saya perlu nama parent atau email EOI untuk mengecek daftar anaknya.",
        )

    try:
        row, _total = await _find_lead_row(settings, authorization, query)
        if row is None:
            return ToolAnswer(
                answer=f"Saya belum menemukan EOI untuk '{query}' di admission-service.",
                sources=[],
                tool_calls=[ToolCallRef(name=tool_name, status="not_found")],
            )
        lead_id = _clean_text(row.get("leadId"))
        if not lead_id:
            raise ValueError("lead row missing leadId")
        detail_url = _join_url(
            settings.admission_service_url,
            f"/api/leads/v1/admin/leads/{lead_id}",
        )
        body = await _get_json(detail_url, authorization, {})
        students = _extract_student_rows(body)
    except httpx.HTTPStatusError as exc:
        auth_error = _auth_status_tool_error(tool_name, exc.response.status_code)
        if auth_error is not None:
            return auth_error
        return _tool_failed(
            tool_name,
            "Saya belum bisa mengambil daftar anak dari admission-service.",
        )
    except Exception:
        return _tool_failed(
            tool_name,
            "Saya belum bisa mengambil daftar anak dari admission-service.",
        )

    parent_name = _clean_text(row.get("parentName")) or query
    if not students:
        answer = f"{parent_name} belum punya data anak yang tersimpan di aplikasi."
    else:
        lines = [f"{parent_name} punya {len(students)} anak di aplikasi:"]
        for index, student in enumerate(students[:5], start=1):
            name = _clean_text(student.get("fullName")) or "Nama belum tersedia"
            target_grade = _clean_text(student.get("targetGradeLevel"))
            target_school = _clean_text(student.get("targetSchool"))
            student_status = _clean_text(student.get("applicantStatus"))
            details = []
            if target_grade:
                details.append(f"grade: {target_grade}")
            if target_school:
                details.append(f"sekolah: {target_school}")
            if student_status:
                details.append(f"status: {student_status}")
            suffix = f" ({'; '.join(details)})" if details else ""
            lines.append(f"{index}. {name}{suffix}")
        answer = "\n".join(lines)

    return ToolAnswer(
        answer=answer,
        sources=[
            SourceRef(kind="service", title="admission-service admin lead detail", reference=None)
        ],
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


async def _payment_application_fee_quote(
    settings: Settings,
    authorization: Optional[str],
    lowered_message: str,
) -> ToolAnswer:
    tool_name = "payment.application_fee_quote"
    if not _has_bearer_token(authorization):
        return _auth_required(tool_name, "data biaya pendaftaran")

    payment_type = _payment_type_from_message(lowered_message)
    schools = _school_codes_from_message(lowered_message)
    rows: list[dict[str, Any]] = []
    try:
        for school_code in schools:
            url = _join_url(settings.payment_service_url, f"/api/v1/payments/fees/{school_code}")
            body = await _get_json(url, authorization, {"payment_type": payment_type})
            rows.append(_extract_fee_row(body))
    except httpx.HTTPStatusError as exc:
        auth_error = _auth_status_tool_error(tool_name, exc.response.status_code)
        if auth_error is not None:
            return auth_error
        return _tool_failed(
            tool_name,
            "Saya belum bisa mengambil biaya pendaftaran dari payment-service.",
        )
    except Exception:
        return _tool_failed(
            tool_name,
            "Saya belum bisa mengambil biaya pendaftaran dari payment-service.",
        )

    fee_label = _payment_type_label(payment_type)
    parts = [
        (
            f"{_clean_text(row.get('schoolCode')) or school}: "
            f"{_format_money(row.get('amount'), row.get('currency'))}"
        )
        for row, school in zip(rows, schools)
    ]
    answer = f"{fee_label.capitalize()} saat ini: {', '.join(parts)}."
    return ToolAnswer(
        answer=answer,
        sources=[SourceRef(kind="service", title="payment-service fee structures", reference=None)],
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


async def _find_lead_row(
    settings: Settings,
    authorization: Optional[str],
    query: str,
) -> tuple[Optional[dict[str, Any]], int]:
    url = _join_url(settings.admission_service_url, "/api/leads/v1/admin/leads")
    body = await _get_json(
        url,
        authorization,
        {"limit": "5", "offset": "0", "search": query},
    )
    rows, total = _extract_lead_rows(body)
    return (rows[0] if rows else None), total


def _lead_list_params(
    limit: int,
    offset: int,
    date_range: Optional[DateRange] = None,
) -> dict[str, str]:
    params = {"limit": str(limit), "offset": str(offset)}
    if date_range is not None:
        params["dateFrom"] = date_range.start.astimezone(timezone.utc).isoformat()
        params["dateTo"] = date_range.end.astimezone(timezone.utc).isoformat()
    return params


def _extract_student_rows(body: dict[str, Any]) -> list[dict[str, Any]]:
    data = body.get("data")
    if not isinstance(data, dict):
        raise ValueError("service response missing data")
    students = data.get("students")
    if not isinstance(students, list):
        raise ValueError("service response missing students")
    return [student for student in students if isinstance(student, dict)]


def _extract_fee_row(body: dict[str, Any]) -> dict[str, Any]:
    data = body.get("data")
    if not isinstance(data, dict):
        raise ValueError("service response missing data")
    amount = data.get("amount")
    if not isinstance(amount, int):
        raise ValueError("service response missing amount")
    return data


def _date_range_from_message(message: str) -> Optional[DateRange]:
    now = _now_jakarta()
    today = now.date()

    relative_day = _relative_day_range_from_message(message, today)
    if relative_day is not None:
        return relative_day

    if any(term in message for term in ("hari ini", "today")):
        return _single_day_range(today, "hari ini")
    if any(term in message for term in ("kemarin", "yesterday")):
        day = today - timedelta(days=1)
        return _single_day_range(day, f"kemarin ({_format_day_label(day)})")
    if any(term in message for term in ("bulan lalu", "last month")):
        first_this_month = today.replace(day=1)
        last_month_end = first_this_month - timedelta(days=1)
        first_last_month = last_month_end.replace(day=1)
        return _range_from_dates(
            first_last_month,
            first_this_month,
            f"bulan lalu ({ID_MONTH_NAMES[first_last_month.month]} {first_last_month.year})",
        )
    if any(term in message for term in ("bulan ini", "this month")):
        first_this_month = today.replace(day=1)
        next_month = _add_month(first_this_month)
        return _range_from_dates(
            first_this_month,
            next_month,
            f"bulan ini ({ID_MONTH_NAMES[first_this_month.month]} {first_this_month.year})",
        )

    specific = _specific_date_from_message(message, now.year)
    if specific is not None:
        return _single_day_range(specific, _format_day_label(specific))

    return None


def _relative_day_range_from_message(message: str, today) -> Optional[DateRange]:
    match = re.search(
        r"\b(?P<count>\d{1,4})\s*(?:hari|day|days)\s*(?:yang\s+)?(?:lalu|ago)\b",
        message,
    )
    if not match:
        return None

    days = int(match.group("count"))
    if days <= 0:
        return None

    day = today - timedelta(days=days)
    return _single_day_range(day, f"{days} hari lalu ({_format_day_label(day)})")


def _specific_date_from_message(message: str, default_year: int):
    iso_match = re.search(r"\b(?P<year>20\d{2})-(?P<month>\d{1,2})-(?P<day>\d{1,2})\b", message)
    if iso_match:
        return _safe_date(
            int(iso_match.group("year")),
            int(iso_match.group("month")),
            int(iso_match.group("day")),
        )

    slash_match = re.search(
        r"\b(?P<day>\d{1,2})[/-](?P<month>\d{1,2})(?:[/-](?P<year>\d{2,4}))?\b",
        message,
    )
    if slash_match:
        year = _normalize_year(slash_match.group("year"), default_year)
        return _safe_date(year, int(slash_match.group("month")), int(slash_match.group("day")))

    month_names = "|".join(sorted(MONTH_ALIASES, key=len, reverse=True))
    day_month_match = re.search(
        rf"\b(?P<day>\d{{1,2}})\s+(?P<month>{month_names})(?:\s+(?P<year>\d{{2,4}}))?\b",
        message,
    )
    if day_month_match:
        year = _normalize_year(day_month_match.group("year"), default_year)
        return _safe_date(
            year,
            MONTH_ALIASES[day_month_match.group("month")],
            int(day_month_match.group("day")),
        )

    month_day_match = re.search(
        rf"\b(?P<month>{month_names})\s+(?P<day>\d{{1,2}})(?:,?\s+(?P<year>\d{{2,4}}))?\b",
        message,
    )
    if month_day_match:
        year = _normalize_year(month_day_match.group("year"), default_year)
        return _safe_date(
            year,
            MONTH_ALIASES[month_day_match.group("month")],
            int(month_day_match.group("day")),
        )

    return None


def _single_day_range(day, label: str) -> DateRange:
    return _range_from_dates(day, day + timedelta(days=1), label)


def _range_from_dates(start_date, end_date, label: str) -> DateRange:
    start = datetime.combine(start_date, datetime.min.time(), tzinfo=SCHOOL_TIME_ZONE)
    end = datetime.combine(end_date, datetime.min.time(), tzinfo=SCHOOL_TIME_ZONE) - timedelta(
        milliseconds=1
    )
    return DateRange(start=start, end=end, label=label)


def _add_month(value):
    if value.month == 12:
        return value.replace(year=value.year + 1, month=1, day=1)
    return value.replace(month=value.month + 1, day=1)


def _format_day_label(day) -> str:
    return f"{day.day} {ID_MONTH_NAMES[day.month]} {day.year}"


def _normalize_year(value: Optional[str], default_year: int) -> int:
    if not value:
        return default_year
    year = int(value)
    return 2000 + year if year < 100 else year


def _safe_date(year: int, month: int, day: int):
    try:
        return datetime(year, month, day, tzinfo=SCHOOL_TIME_ZONE).date()
    except ValueError:
        return None


def _now_jakarta() -> datetime:
    return datetime.now(SCHOOL_TIME_ZONE)


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


def _asks_for_lead_child_count(message: str) -> bool:
    count_terms = ("berapa", "jumlah", "count", "how many", "number of")
    child_terms = ("anak", "siswa", "murid", "student", "students", "child", "children", "kid")
    lead_terms = (
        "eoi",
        "lead",
        "admission",
        "admissions",
        "daftarin",
        "mendaftarkan",
        "didaftarkan",
        "applicant",
        "applicants",
    )
    return (
        any(term in message for term in count_terms)
        and any(term in message for term in child_terms)
        and any(term in message for term in lead_terms)
    )


def _asks_for_lead_child_identity(message: str) -> bool:
    identity_terms = ("siapa", "nama", "who", "name", "list", "daftar")
    child_terms = (
        "anak",
        "anaknya",
        "siswa",
        "murid",
        "child",
        "children",
        "kid",
    )
    return any(term in message for term in identity_terms) and any(
        term in message for term in child_terms
    )


def _asks_for_contextual_lead_identity(message: str) -> bool:
    followup_terms = (
        "siapa orang itu",
        "siapa",
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


def _asks_for_contextual_lead_detail(message: str) -> bool:
    followup_terms = (
        "detail",
        "detailnya",
        "lihat detail",
        "lihat datanya",
        "mau lihat",
        "info lengkap",
        "lebih lengkap",
        "lengkapnya",
        "tampilkan detail",
        "show detail",
        "show details",
        "more detail",
        "more details",
        "data lengkap",
    )
    return any(term in message for term in followup_terms)


def _asks_for_contextual_period_followup(message: str) -> bool:
    if re.search(
        r"\b\d{1,4}\s*(?:hari|day|days)\s*(?:yang\s+)?(?:lalu|ago)\b",
        message,
    ):
        return True

    period_terms = (
        "hari ini",
        "today",
        "kemarin",
        "yesterday",
        "bulan lalu",
        "last month",
        "bulan ini",
        "this month",
    )
    if any(term in message for term in period_terms):
        return True
    return _specific_date_from_message(message, _now_jakarta().year) is not None


def _asks_for_contextual_lead_child_identity(message: str) -> bool:
    followup_terms = (
        "siapa",
        "siapa anaknya",
        "anaknya siapa",
        "nama anaknya",
        "list anaknya",
        "tampilkan anaknya",
        "who",
        "who are they",
        "list them",
    )
    return any(term == message.strip() or term in message for term in followup_terms)


def _history_has_tool_call(payload: ChatRequest, tool_name: str) -> bool:
    for turn in payload.history[-10:]:
        if any(tool.name == tool_name and tool.status == "ok" for tool in turn.tool_calls):
            return True
    return False


def _last_ok_tool_call(payload: ChatRequest) -> Optional[str]:
    for turn in reversed(payload.history[-10:]):
        for tool in reversed(turn.tool_calls):
            if tool.status == "ok":
                return tool.name
    return None


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())


def _safe_int(value: Any) -> int:
    return value if isinstance(value, int) else 0


def _lead_search_query_from_context(payload: ChatRequest, lowered_message: str) -> str:
    query = _lead_search_query_from_message(lowered_message)
    if query:
        return query

    for turn in reversed(payload.history[-10:]):
        if turn.role == "user":
            query = _lead_search_query_from_message(turn.content.casefold())
            if query:
                return query
        if turn.role == "assistant":
            query = _lead_search_query_from_assistant(turn.content)
            if query:
                return query
    return ""


def _lead_search_query_from_message(message: str) -> str:
    original = " ".join(message.strip().split())
    if not original:
        return ""

    patterns = (
        r"^(?P<name>.+?)\s+(?:di|dalam|in)\s+(?:eoi|lead|admission|admissions)\b",
        r"^(?P<name>.+?)\s+(?:dia|mereka)?\s*(?:daftarin|mendaftarkan|didaftarkan)\b",
        r"^(?P<name>.+?)\s+(?:punya|memiliki)\s+(?:berapa\s+)?(?:anak|siswa|murid)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, original, flags=re.IGNORECASE)
        if match:
            return _clean_search_query(match.group("name"))

    if "@" in original:
        tokens = [token for token in re.split(r"\s+", original) if "@" in token]
        return _clean_search_query(tokens[0]) if tokens else ""
    return ""


def _lead_search_query_from_assistant(message: str) -> str:
    match = re.search(r"^\s*\d+\.\s+([^(\n]+)", message, flags=re.MULTILINE)
    if match:
        return _clean_search_query(match.group(1))
    match = re.search(r"^(.+?)\s+(?:mendaftarkan|belum punya data anak)", message)
    if match:
        return _clean_search_query(match.group(1))
    return ""


def _clean_search_query(value: str) -> str:
    cleaned = re.sub(
        r"\b(?:berapa|jumlah|anak|siswa|murid|child|children|student|students|eoi|lead)\b",
        " ",
        value,
        flags=re.IGNORECASE,
    )
    return " ".join(cleaned.strip(" ?.,:;()[]{}").split())


def _asks_for_payment_review_count(message: str) -> bool:
    count_terms = ("berapa", "total", "jumlah", "count", "how many", "number of")
    payment_terms = ("payment", "pembayaran", "tagihan", "invoice", "manual transfer")
    review_terms = (
        "pending",
        "menunggu",
        "verifikasi",
        "verification",
        "review",
        "cek finance",
        "sampai",
        "masuk",
        "bukti",
        "proof",
    )
    if not any(term in message for term in payment_terms):
        return False
    return any(term in message for term in count_terms) or any(
        term in message for term in review_terms
    )


def _asks_for_payment_application_fee(message: str) -> bool:
    price_terms = ("harga", "biaya", "fee", "tarif", "cost", "price")
    admission_terms = (
        "admission",
        "admissions",
        "pendaftaran",
        "registrasi",
        "application",
        "application fee",
    )
    return any(term in message for term in price_terms) and any(
        term in message for term in admission_terms
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


def _payment_type_from_message(message: str) -> str:
    if "enrolment" in message or "enrollment" in message:
        return "enrolment_fee"
    return "application_fee"


def _payment_type_label(payment_type: str) -> str:
    if payment_type == "enrolment_fee":
        return "biaya enrolment"
    if payment_type == "application_fee":
        return "biaya pendaftaran"
    return payment_type.replace("_", " ")


def _school_codes_from_message(message: str) -> list[str]:
    codes = [code for code in DEFAULT_SCHOOL_CODES if code.casefold() in message]
    return codes or list(DEFAULT_SCHOOL_CODES)


def _format_money(amount: Any, currency: Any) -> str:
    if not isinstance(amount, int):
        raise ValueError("invalid amount")
    currency_text = currency if isinstance(currency, str) and currency.strip() else "IDR"
    if currency_text.upper() == "IDR":
        return f"Rp {amount:,}".replace(",", ".")
    return f"{currency_text.upper()} {amount:,}"


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

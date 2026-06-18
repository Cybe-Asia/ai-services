import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Any, Optional
from urllib.parse import urlencode
from xml.sax.saxutils import escape as xml_escape
from zipfile import ZIP_DEFLATED, ZipFile
from zoneinfo import ZoneInfo

import httpx

from app.config import Settings
from app.llm_client import LlmClient
from app.schemas import ActorRole, ChatRequest, SourceRef, ToolCallRef

INTENT_NONE = "none"
INTENT_ADMISSION_EOI_COUNT = "admission_eoi_count"
INTENT_ADMISSION_LEADS_LIST = "admission_leads_list"
INTENT_ADMISSION_LEAD_DETAIL = "admission_lead_detail"
INTENT_ADMISSION_LEAD_CHILD_COUNT = "admission_lead_child_count"
INTENT_ADMISSION_LEAD_STUDENTS_LIST = "admission_lead_students_list"
INTENT_PAYMENT_REVIEW_COUNT = "payment_review_count"
INTENT_PAYMENT_APPLICATION_FEE = "payment_application_fee_quote"
INTENT_ADMISSIONS_PAYMENTS_REPORT = "admissions_payments_report"
INTENT_CURRENT_DATE = "current_date"
PAYMENT_STATUSES = {"pending_verification", "paid", "rejected", "underpaid"}
DEFAULT_SCHOOL_CODES = ("IIHS", "IISS", "IIBS")
REPORT_EXPORT_FORMATS = {"xlsx", "pdf", "docx", "md"}
REPORT_EXPORT_LIMIT = 500
REPORT_PREVIEW_LIMIT = 5
OPENXML_CONTENT_TYPES_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
OPENXML_PACKAGE_RELS_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
OPENXML_OFFICE_RELS_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
OPENXML_RELS_CONTENT_TYPE = "application/vnd.openxmlformats-package.relationships+xml"
XLSX_WORKBOOK_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"
)
XLSX_WORKSHEET_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"
)
DOCX_DOCUMENT_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"
)
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
ID_WEEKDAY_NAMES = {
    0: "Senin",
    1: "Selasa",
    2: "Rabu",
    3: "Kamis",
    4: "Jumat",
    5: "Sabtu",
    6: "Minggu",
}
EN_WEEKDAY_NAMES = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
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


@dataclass(frozen=True)
class ReportRequest:
    date_range: Optional[DateRange]
    language: str = "id"
    payment_status: Optional[str] = None
    school: str = ""
    search: str = ""
    limit: int = REPORT_EXPORT_LIMIT


@dataclass(frozen=True)
class AdmissionsPaymentsReport:
    request: ReportRequest
    lead_rows: list[dict[str, Any]]
    lead_total: int
    payment_rows: list[dict[str, Any]]
    payment_total: int
    generated_at: datetime


@dataclass(frozen=True)
class ReportFile:
    filename: str
    media_type: str
    body: bytes


async def answer_from_school_tools(
    payload: ChatRequest,
    settings: Settings,
    authorization: Optional[str],
) -> Optional[ToolAnswer]:
    lowered = payload.message.casefold()
    language = _answer_language(payload, lowered)
    if _asks_for_current_date(lowered):
        return _current_date_answer(language)

    if payload.actor_role not in {ActorRole.owner, ActorRole.admin}:
        return None

    date_range = _date_range_from_message(lowered)
    intent = _deterministic_tool_intent(lowered)
    if intent.name == INTENT_NONE:
        intent = _contextual_tool_intent(payload, lowered, date_range)
    if intent.name == INTENT_NONE:
        intent = await _classify_tool_intent(payload.message, settings)

    if intent.name == INTENT_ADMISSIONS_PAYMENTS_REPORT:
        return await _admissions_payments_report_answer(
            settings,
            authorization,
            lowered,
            language,
        )
    if intent.name == INTENT_ADMISSION_EOI_COUNT:
        return await _admission_eoi_count(settings, authorization, date_range, language)
    if intent.name == INTENT_ADMISSION_LEADS_LIST:
        return await _admission_leads_list(settings, authorization, date_range, language)
    if intent.name == INTENT_ADMISSION_LEAD_DETAIL:
        return await _admission_lead_detail(settings, authorization, payload, lowered, language)
    if intent.name == INTENT_ADMISSION_LEAD_CHILD_COUNT:
        return await _admission_lead_child_count(
            settings,
            authorization,
            payload,
            lowered,
            language,
        )
    if intent.name == INTENT_ADMISSION_LEAD_STUDENTS_LIST:
        return await _admission_lead_students_list(
            settings,
            authorization,
            payload,
            lowered,
            language,
        )
    if intent.name == INTENT_PAYMENT_REVIEW_COUNT:
        return await _payment_review_count(
            settings,
            authorization,
            lowered,
            date_range,
            intent.payment_status,
            language,
        )
    if intent.name == INTENT_PAYMENT_APPLICATION_FEE:
        return await _payment_application_fee_quote(settings, authorization, lowered, language)
    if intent.name == INTENT_CURRENT_DATE:
        return _current_date_answer(language)
    return None


def _deterministic_tool_intent(message: str) -> ToolIntent:
    if _asks_for_current_date(message):
        return ToolIntent(INTENT_CURRENT_DATE)
    if _asks_for_admissions_payments_report(message):
        return ToolIntent(INTENT_ADMISSIONS_PAYMENTS_REPORT)
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
        if last_tool == "payment.admin_review_count":
            return ToolIntent(
                INTENT_PAYMENT_REVIEW_COUNT,
                _payment_status_from_message(lowered_message),
            )

    if _asks_for_contextual_lead_child_identity(lowered_message) and _history_has_tool_call(
        payload,
        "admission.admin_lead_child_count",
    ):
        return ToolIntent(INTENT_ADMISSION_LEAD_STUDENTS_LIST)
    if _asks_for_contextual_lead_detail(lowered_message):
        if (
            _history_has_tool_call(payload, "admission.admin_leads_list")
            or _history_has_tool_call(payload, "admission.admin_lead_child_count")
            or _history_has_tool_call(payload, "admission.admin_lead_students_list")
            or _history_has_tool_call(payload, "admission.admin_lead_detail")
        ):
            return ToolIntent(INTENT_ADMISSION_LEAD_DETAIL)
        if _history_has_tool_call(payload, "admission.admin_leads_count"):
            return ToolIntent(INTENT_ADMISSION_LEADS_LIST)

    if _asks_for_contextual_lead_identity(lowered_message) and (
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
        "If user asks to report/export/download/show EOI together with payment data, "
        "use admissions_payments_report. "
        "If user asks admission/application price, biaya pendaftaran, fee, tariff, or cost, "
        "use payment_application_fee_quote. Otherwise use none. "
        'Example: "berapa calon keluarga masuk?" => {"intent":"admission_eoi_count"}. '
        'Example: "Arief Nugraha di EOI daftarin berapa anak?" => '
        '{"intent":"admission_lead_child_count"}. '
        'Example: "berapa transaksi yang masih perlu dicek finance?" => '
        '{"intent":"payment_review_count","paymentStatus":"pending_verification"}. '
        'Example: "export EOI and payment this month" => '
        '{"intent":"admissions_payments_report"}. '
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
    if intent == INTENT_ADMISSION_LEAD_DETAIL:
        return ToolIntent(INTENT_ADMISSION_LEAD_DETAIL)
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
    if intent == INTENT_ADMISSIONS_PAYMENTS_REPORT:
        return ToolIntent(INTENT_ADMISSIONS_PAYMENTS_REPORT)
    if intent == INTENT_CURRENT_DATE:
        return ToolIntent(INTENT_CURRENT_DATE)
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
        INTENT_ADMISSION_LEAD_DETAIL: INTENT_ADMISSION_LEAD_DETAIL,
        "admission_admin_lead_detail": INTENT_ADMISSION_LEAD_DETAIL,
        "lead_detail": INTENT_ADMISSION_LEAD_DETAIL,
        "eoi_detail": INTENT_ADMISSION_LEAD_DETAIL,
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
        INTENT_ADMISSIONS_PAYMENTS_REPORT: INTENT_ADMISSIONS_PAYMENTS_REPORT,
        "admission_payment_report": INTENT_ADMISSIONS_PAYMENTS_REPORT,
        "admissions_payment_report": INTENT_ADMISSIONS_PAYMENTS_REPORT,
        "eoi_payment_report": INTENT_ADMISSIONS_PAYMENTS_REPORT,
        "report_admissions_payments": INTENT_ADMISSIONS_PAYMENTS_REPORT,
        INTENT_CURRENT_DATE: INTENT_CURRENT_DATE,
        "today": INTENT_CURRENT_DATE,
        "current_day": INTENT_CURRENT_DATE,
        "current_date": INTENT_CURRENT_DATE,
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
    language: str = "id",
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

    scope = _date_scope(date_range, language)
    answer = (
        f"There are {total} EOIs registered{scope} in admission-service."
        if language == "en"
        else f"Ada {total} EOI terdaftar{scope} di admission-service."
    )
    return ToolAnswer(
        answer=answer,
        sources=[SourceRef(kind="service", title="admission-service admin leads", reference=None)],
        tool_calls=[ToolCallRef(name=tool_name, status="ok")],
    )


async def _admission_leads_list(
    settings: Settings,
    authorization: Optional[str],
    date_range: Optional[DateRange] = None,
    language: str = "id",
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

    scope = _date_scope(date_range, language)
    if total == 0 or not rows:
        answer = (
            f"No EOIs are registered{scope} in admission-service."
            if language == "en"
            else f"Belum ada EOI terdaftar{scope} di admission-service."
        )
    else:
        shown = min(len(rows), 5)
        lines = [
            f"There are {total} EOIs{scope}. Showing the latest {shown}:"
            if language == "en"
            else f"Ada {total} EOI{scope}. Saya tampilkan {shown} yang terbaru:"
        ]
        for index, row in enumerate(rows[:shown], start=1):
            name = _clean_text(row.get("parentName")) or (
                "Name unavailable" if language == "en" else "Nama belum tersedia"
            )
            email = _clean_text(row.get("email"))
            school = _clean_text(row.get("school"))
            lead_status = _clean_text(row.get("leadStatus"))
            details = []
            if email:
                details.append(email)
            if school:
                details.append(f"school: {school}" if language == "en" else f"sekolah: {school}")
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


async def _admission_lead_detail(
    settings: Settings,
    authorization: Optional[str],
    payload: ChatRequest,
    lowered_message: str,
    language: str = "id",
) -> ToolAnswer:
    tool_name = "admission.admin_lead_detail"
    if not _has_bearer_token(authorization):
        return _auth_required(tool_name, "detail EOI")

    query = _lead_search_query_from_context(payload, lowered_message)
    if not query:
        return _tool_failed(
            tool_name,
            "I need the parent name or EOI email to show the full detail."
            if language == "en"
            else "Saya perlu nama parent atau email EOI untuk menampilkan detail lengkap.",
        )

    try:
        row, total = await _find_lead_row(settings, authorization, query)
        if row is None:
            return ToolAnswer(
                answer=(
                    f"I could not find an EOI for '{query}' in admission-service."
                    if language == "en"
                    else f"Saya belum menemukan EOI untuk '{query}' di admission-service."
                ),
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
        detail, lead, students = _extract_lead_detail(body)
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

    answer = _format_lead_detail_answer(row, detail, lead, students, total, language)
    return ToolAnswer(
        answer=answer,
        sources=[
            SourceRef(kind="service", title="admission-service admin lead detail", reference=None)
        ],
        tool_calls=[ToolCallRef(name=tool_name, status="ok")],
    )


async def _admission_lead_child_count(
    settings: Settings,
    authorization: Optional[str],
    payload: ChatRequest,
    lowered_message: str,
    language: str = "id",
) -> ToolAnswer:
    tool_name = "admission.admin_lead_child_count"
    if not _has_bearer_token(authorization):
        return _auth_required(tool_name, "jumlah anak di EOI")

    query = _lead_search_query_from_context(payload, lowered_message)
    if not query:
        return _tool_failed(
            tool_name,
            "I need the parent name or EOI email to check the child count."
            if language == "en"
            else "Saya perlu nama parent atau email EOI untuk mengecek jumlah anaknya.",
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
        answer = (
            f"I could not find an EOI for '{query}' in admission-service."
            if language == "en"
            else f"Saya belum menemukan EOI untuk '{query}' di admission-service."
        )
    else:
        name = _clean_text(row.get("parentName")) or query
        count = _safe_int(row.get("applicantCount"))
        school = _clean_text(row.get("school"))
        status = _clean_text(row.get("leadStatus"))
        suffix = []
        if school:
            suffix.append(f"school: {school}" if language == "en" else f"sekolah: {school}")
        if status:
            suffix.append(f"status: {status}")
        if total > 1:
            suffix.append(
                f"{total} results found; using the top result"
                if language == "en"
                else f"ada {total} hasil; saya pakai hasil teratas"
            )
        detail = f" ({'; '.join(suffix)})" if suffix else ""
        if count == 0:
            answer = (
                f"{name} does not have child data saved in the application yet{detail}."
                if language == "en"
                else f"{name} belum punya data anak yang tersimpan di aplikasi{detail}."
            )
        else:
            answer = (
                f"{name} registered {count} children in the application{detail}."
                if language == "en"
                else f"{name} mendaftarkan {count} anak di aplikasi{detail}."
            )

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
    language: str = "id",
) -> ToolAnswer:
    tool_name = "admission.admin_lead_students_list"
    if not _has_bearer_token(authorization):
        return _auth_required(tool_name, "daftar anak di EOI")

    query = _lead_search_query_from_context(payload, lowered_message)
    if not query:
        return _tool_failed(
            tool_name,
            "I need the parent name or EOI email to check the child list."
            if language == "en"
            else "Saya perlu nama parent atau email EOI untuk mengecek daftar anaknya.",
        )

    try:
        row, _total = await _find_lead_row(settings, authorization, query)
        if row is None:
            return ToolAnswer(
                answer=(
                    f"I could not find an EOI for '{query}' in admission-service."
                    if language == "en"
                    else f"Saya belum menemukan EOI untuk '{query}' di admission-service."
                ),
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
        answer = (
            f"{parent_name} does not have child data saved in the application yet."
            if language == "en"
            else f"{parent_name} belum punya data anak yang tersimpan di aplikasi."
        )
    else:
        lines = [
            f"{parent_name} has {len(students)} children in the application:"
            if language == "en"
            else f"{parent_name} punya {len(students)} anak di aplikasi:"
        ]
        for index, student in enumerate(students[:5], start=1):
            name = _clean_text(student.get("fullName")) or (
                "Name unavailable" if language == "en" else "Nama belum tersedia"
            )
            date_of_birth = _clean_text(student.get("dateOfBirth"))
            age = _string_value(student.get("ageAtApplication"))
            current_school = _clean_text(student.get("currentSchool"))
            target_grade = _clean_text(student.get("targetGradeLevel"))
            target_school = _clean_text(student.get("targetSchool"))
            application_mode = _clean_text(student.get("applicationMode"))
            student_status = _clean_text(student.get("applicantStatus"))
            details = []
            if date_of_birth:
                details.append(
                    f"DOB: {date_of_birth}"
                    if language == "en"
                    else f"tanggal lahir: {date_of_birth}"
                )
            if age:
                details.append(f"age: {age}" if language == "en" else f"usia: {age}")
            if current_school:
                details.append(
                    f"current school: {current_school}"
                    if language == "en"
                    else f"sekolah asal: {current_school}"
                )
            if target_grade:
                details.append(f"grade: {target_grade}")
            if target_school:
                details.append(
                    f"school: {target_school}" if language == "en" else f"sekolah: {target_school}"
                )
            if application_mode:
                details.append(f"mode: {application_mode}")
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
    date_range: Optional[DateRange] = None,
    status_override: Optional[str] = None,
    language: str = "id",
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
            _payment_review_params(
                status=status,
                limit=1,
                offset=0,
                date_range=date_range,
            ),
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

    label = _payment_status_label(status, language)
    scope = _date_scope(date_range, language)
    answer = (
        f"There are {total} payments {label}{scope} in payment-service."
        if language == "en"
        else f"Ada {total} pembayaran {label}{scope} di payment-service."
    )
    return ToolAnswer(
        answer=answer,
        sources=[SourceRef(kind="service", title="payment-service admin reviews", reference=None)],
        tool_calls=[ToolCallRef(name=tool_name, status="ok")],
    )


async def _payment_application_fee_quote(
    settings: Settings,
    authorization: Optional[str],
    lowered_message: str,
    language: str = "id",
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

    fee_label = _payment_type_label(payment_type, language)
    parts = [
        (
            f"{_clean_text(row.get('schoolCode')) or school}: "
            f"{_format_money(row.get('amount'), row.get('currency'))}"
        )
        for row, school in zip(rows, schools)
    ]
    answer = (
        f"Current {fee_label}: {', '.join(parts)}."
        if language == "en"
        else f"{fee_label.capitalize()} saat ini: {', '.join(parts)}."
    )
    return ToolAnswer(
        answer=answer,
        sources=[SourceRef(kind="service", title="payment-service fee structures", reference=None)],
        tool_calls=[ToolCallRef(name=tool_name, status="ok")],
    )


def _current_date_answer(language: str = "id") -> ToolAnswer:
    now = _now_jakarta()
    day = now.date()
    if language == "en":
        weekday = EN_WEEKDAY_NAMES[day.weekday()]
        answer = f"Today is {weekday}, {_format_day_label_en(day)}."
    else:
        weekday = ID_WEEKDAY_NAMES[day.weekday()]
        answer = f"Hari ini {weekday}, {_format_day_label(day)}."
    return ToolAnswer(
        answer=answer,
        sources=[],
        tool_calls=[ToolCallRef(name="system.current_date", status="ok")],
    )


async def _admissions_payments_report_answer(
    settings: Settings,
    authorization: Optional[str],
    lowered_message: str,
    language: str,
) -> ToolAnswer:
    tool_name = "report.admissions_payments"
    if not _has_bearer_token(authorization):
        return _auth_required(tool_name, "laporan EOI dan pembayaran")

    request = _report_request_from_message(lowered_message, language, REPORT_PREVIEW_LIMIT)
    try:
        report = await build_admissions_payments_report(settings, authorization, request)
    except httpx.HTTPStatusError as exc:
        auth_error = _auth_status_tool_error(tool_name, exc.response.status_code)
        if auth_error is not None:
            return auth_error
        return _tool_failed(
            tool_name,
            "Saya belum bisa membuat laporan dari admission/payment service.",
        )
    except Exception:
        return _tool_failed(
            tool_name,
            "Saya belum bisa membuat laporan dari admission/payment service.",
        )

    answer = _format_report_preview_answer(report)
    sources = _report_export_sources(
        request=ReportRequest(
            date_range=request.date_range,
            language=language,
            payment_status=request.payment_status,
            school=request.school,
            search=request.search,
            limit=REPORT_EXPORT_LIMIT,
        )
    )
    return ToolAnswer(
        answer=answer,
        sources=sources,
        tool_calls=[
            ToolCallRef(name=tool_name, status="ok"),
            ToolCallRef(name="admission.admin_leads_list", status="ok"),
            ToolCallRef(name="payment.admin_reviews_list", status="ok"),
        ],
    )


async def build_admissions_payments_report(
    settings: Settings,
    authorization: Optional[str],
    request: ReportRequest,
) -> AdmissionsPaymentsReport:
    lead_rows, lead_total = await _fetch_report_lead_rows(settings, authorization, request)
    payment_rows, payment_total = await _fetch_report_payment_rows(settings, authorization, request)
    return AdmissionsPaymentsReport(
        request=request,
        lead_rows=lead_rows,
        lead_total=lead_total,
        payment_rows=payment_rows,
        payment_total=payment_total,
        generated_at=_now_jakarta(),
    )


def report_request_from_export_query(
    *,
    date_from: Optional[str],
    date_to: Optional[str],
    payment_status: Optional[str],
    school: Optional[str],
    search: Optional[str],
    limit: int,
    language: str,
) -> ReportRequest:
    date_range = _date_range_from_query(date_from, date_to)
    normalized_payment_status = (
        _normalize_payment_status(payment_status) if payment_status else None
    )
    clean_school = _clean_text(school).upper()
    if clean_school and clean_school not in DEFAULT_SCHOOL_CODES:
        clean_school = ""
    return ReportRequest(
        date_range=date_range,
        language="en" if language.casefold().startswith("en") else "id",
        payment_status=normalized_payment_status,
        school=clean_school,
        search=_clean_text(search),
        limit=max(1, min(limit, REPORT_EXPORT_LIMIT)),
    )


def render_admissions_payments_report(
    report: AdmissionsPaymentsReport,
    export_format: str,
) -> ReportFile:
    normalized = export_format.casefold().strip().lstrip(".")
    if normalized not in REPORT_EXPORT_FORMATS:
        raise ValueError("unsupported report export format")

    filename_base = _report_filename_base(report)
    if normalized == "xlsx":
        return ReportFile(
            filename=f"{filename_base}.xlsx",
            media_type=(
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ),
            body=_render_report_xlsx(report),
        )
    if normalized == "docx":
        return ReportFile(
            filename=f"{filename_base}.docx",
            media_type=(
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ),
            body=_render_report_docx(report),
        )
    if normalized == "pdf":
        return ReportFile(
            filename=f"{filename_base}.pdf",
            media_type="application/pdf",
            body=_render_report_pdf(report),
        )
    return ReportFile(
        filename=f"{filename_base}.md",
        media_type="text/markdown; charset=utf-8",
        body=_render_report_markdown(report).encode("utf-8"),
    )


async def _fetch_report_lead_rows(
    settings: Settings,
    authorization: Optional[str],
    request: ReportRequest,
) -> tuple[list[dict[str, Any]], int]:
    url = _join_url(settings.admission_service_url, "/api/leads/v1/admin/leads")
    rows: list[dict[str, Any]] = []
    total = 0
    offset = 0
    while len(rows) < request.limit:
        page_limit = min(200, request.limit - len(rows))
        params = _lead_list_params(
            limit=page_limit,
            offset=offset,
            date_range=request.date_range,
            school=request.school,
            search=request.search,
        )
        body = await _get_json(url, authorization, params)
        page_rows, total = _extract_lead_rows(body)
        rows.extend(page_rows)
        if not page_rows or len(rows) >= total:
            break
        offset += len(page_rows)
    return rows[: request.limit], total


async def _fetch_report_payment_rows(
    settings: Settings,
    authorization: Optional[str],
    request: ReportRequest,
) -> tuple[list[dict[str, Any]], int]:
    url = _join_url(settings.payment_service_url, "/api/v1/payments/admin/reviews")
    rows: list[dict[str, Any]] = []
    total = 0
    offset = 0
    while len(rows) < min(request.limit, REPORT_EXPORT_LIMIT):
        page_limit = min(200, request.limit - len(rows))
        params = _payment_review_params(
            status=request.payment_status,
            limit=page_limit,
            offset=offset,
            school=request.school,
            search=request.search,
            date_range=request.date_range,
        )
        body = await _get_json(url, authorization, params)
        page_rows, total = _extract_payment_review_rows(body)
        rows.extend(page_rows)
        if not page_rows or len(rows) >= total:
            break
        offset += len(page_rows)
    return rows[: request.limit], total


def _format_report_preview_answer(report: AdmissionsPaymentsReport) -> str:
    request = report.request
    language = request.language
    period = _report_period_label(request)
    lead_shown = min(len(report.lead_rows), REPORT_PREVIEW_LIMIT)
    payment_shown = min(len(report.payment_rows), REPORT_PREVIEW_LIMIT)
    if language == "en":
        lines = [
            "Admissions + payment report is ready.",
            f"Period: {period}.",
            f"EOI rows: {report.lead_total} total; previewing {lead_shown}.",
            f"Payment review rows: {report.payment_total} total; previewing {payment_shown}.",
            "Export links are available as Excel, PDF, Word, and Markdown.",
        ]
    else:
        lines = [
            "Laporan EOI + pembayaran sudah siap.",
            f"Periode: {period}.",
            f"Data EOI: {report.lead_total} total; preview {lead_shown}.",
            f"Data review pembayaran: {report.payment_total} total; preview {payment_shown}.",
            "Link export tersedia sebagai Excel, PDF, Word, dan Markdown.",
        ]

    if report.lead_rows:
        lines.append("")
        lines.extend(_preview_lead_lines(report.lead_rows, language))
    if report.payment_rows:
        lines.append("")
        lines.extend(_preview_payment_lines(report.payment_rows, language))
    if report.lead_total > report.request.limit or report.payment_total > report.request.limit:
        lines.append("")
        lines.append(
            f"Note: exports are capped at {REPORT_EXPORT_LIMIT} rows per table."
            if language == "en"
            else f"Catatan: export dibatasi {REPORT_EXPORT_LIMIT} baris per tabel."
        )
    return "\n".join(lines)


def _preview_lead_lines(rows: list[dict[str, Any]], language: str) -> list[str]:
    title = "Latest EOIs:" if language == "en" else "EOI terbaru:"
    lines = [title]
    for index, row in enumerate(rows[:REPORT_PREVIEW_LIMIT], start=1):
        name = _clean_text(row.get("parentName")) or (
            "Name unavailable" if language == "en" else "Nama belum tersedia"
        )
        school = _clean_text(row.get("school"))
        payment_status = _clean_text(row.get("latestPaymentStatus")) or "-"
        lines.append(f"{index}. {name} ({school or '-'}; payment: {payment_status})")
    return lines


def _preview_payment_lines(rows: list[dict[str, Any]], language: str) -> list[str]:
    title = "Payment review queue:" if language == "en" else "Antrian review pembayaran:"
    lines = [title]
    for index, row in enumerate(rows[:REPORT_PREVIEW_LIMIT], start=1):
        name = _clean_text(row.get("parentName")) or (
            "Name unavailable" if language == "en" else "Nama belum tersedia"
        )
        amount = _format_money(row.get("amount"), row.get("currency"))
        status = _clean_text(row.get("status")) or "-"
        lines.append(f"{index}. {name} ({status}; {amount})")
    return lines


def _report_export_sources(request: ReportRequest) -> list[SourceRef]:
    labels = {
        "xlsx": "Download Excel",
        "pdf": "Download PDF",
        "docx": "Download Word",
        "md": "Download Markdown",
    }
    return [
        SourceRef(
            kind="file",
            title=labels[export_format],
            reference=(
                "/api/admin/ai/reports/admissions-payments?"
                f"{_report_query(request, export_format)}"
            ),
        )
        for export_format in ("xlsx", "pdf", "docx", "md")
    ]


def _report_query(request: ReportRequest, export_format: str) -> str:
    params: dict[str, str] = {
        "format": export_format,
        "limit": str(request.limit),
        "locale": request.language,
    }
    if request.date_range is not None:
        params["dateFrom"] = request.date_range.start.astimezone(timezone.utc).isoformat()
        params["dateTo"] = request.date_range.end.astimezone(timezone.utc).isoformat()
    if request.payment_status:
        params["paymentStatus"] = request.payment_status
    if request.school:
        params["school"] = request.school
    if request.search:
        params["search"] = request.search
    return urlencode(params)


def _report_request_from_message(
    lowered_message: str,
    language: str,
    limit: int,
) -> ReportRequest:
    date_range = None if _asks_for_all_time(lowered_message) else _date_range_from_message(
        lowered_message
    )
    schools = [code for code in DEFAULT_SCHOOL_CODES if code.casefold() in lowered_message]
    payment_status = _explicit_payment_status_from_message(lowered_message)
    return ReportRequest(
        date_range=date_range,
        language=language,
        payment_status=payment_status,
        school=schools[0] if len(schools) == 1 else "",
        search="",
        limit=limit,
    )


def _date_range_from_query(date_from: Optional[str], date_to: Optional[str]) -> Optional[DateRange]:
    if not date_from and not date_to:
        return None
    if not date_from or not date_to:
        raise ValueError("dateFrom and dateTo must be provided together")
    start = _parse_query_datetime(date_from)
    end = _parse_query_datetime(date_to)
    if start is None or end is None or end < start:
        raise ValueError("invalid date range")
    label = _format_query_date_range_label(start, end)
    return DateRange(start=start, end=end, label=label)


def _parse_query_datetime(value: str) -> Optional[datetime]:
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=SCHOOL_TIME_ZONE)
    return parsed.astimezone(SCHOOL_TIME_ZONE)


def _format_query_date_range_label(start: datetime, end: datetime) -> str:
    start_date = start.astimezone(SCHOOL_TIME_ZONE).date()
    end_date = end.astimezone(SCHOOL_TIME_ZONE).date()
    if start_date == end_date:
        return _format_day_label(start_date)
    return f"{_format_day_label(start_date)} - {_format_day_label(end_date)}"


def _payment_review_params(
    *,
    status: Optional[str],
    limit: int,
    offset: int,
    school: str = "",
    search: str = "",
    date_range: Optional[DateRange] = None,
) -> dict[str, str]:
    params = {
        "status": status or "",
        "limit": str(limit),
        "offset": str(offset),
    }
    if school:
        params["school"] = school
    if search:
        params["search"] = search
    if date_range is not None:
        params["dateFrom"] = date_range.start.astimezone(timezone.utc).isoformat()
        params["dateTo"] = date_range.end.astimezone(timezone.utc).isoformat()
    return params


def _extract_payment_review_rows(body: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
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


def _report_period_label(request: ReportRequest) -> str:
    if request.date_range is None:
        return "all time" if request.language == "en" else "semua waktu"
    if request.language == "en":
        return _format_date_range_label_en(request.date_range)
    return request.date_range.label


def _report_filename_base(report: AdmissionsPaymentsReport) -> str:
    generated = report.generated_at.strftime("%Y%m%d-%H%M")
    return f"digital-school-admissions-payments-{generated}"


def _report_tables(
    report: AdmissionsPaymentsReport,
) -> list[tuple[str, list[str], list[list[str]]]]:
    return [
        ("EOI + Latest Payment", _lead_report_headers(), _lead_report_rows(report.lead_rows)),
        (
            "Payment Review Queue",
            _payment_report_headers(),
            _payment_report_rows(report.payment_rows),
        ),
    ]


def _lead_report_headers() -> list[str]:
    return [
        "submitted_at",
        "lead_id",
        "parent_name",
        "email",
        "whatsapp",
        "school",
        "lead_status",
        "has_application",
        "application_status",
        "applicant_count",
        "latest_payment_status",
        "latest_payment_type",
        "reference_code",
        "reference_owner",
        "campaign",
    ]


def _lead_report_rows(rows: list[dict[str, Any]]) -> list[list[str]]:
    return [
        [
            _clean_text(row.get("submittedAt")),
            _clean_text(row.get("leadId")),
            _clean_text(row.get("parentName")),
            _clean_text(row.get("email")),
            _clean_text(row.get("whatsapp")),
            _clean_text(row.get("school")),
            _clean_text(row.get("leadStatus")),
            "yes" if row.get("hasApplication") is True else "no",
            _clean_text(row.get("applicationStatus")),
            str(_safe_int(row.get("applicantCount"))),
            _clean_text(row.get("latestPaymentStatus")),
            _clean_text(row.get("latestPaymentType")),
            _clean_text(row.get("referenceCode")),
            _clean_text(row.get("referenceOwnerName")),
            _clean_text(row.get("campaignName")),
        ]
        for row in rows
    ]


def _payment_report_headers() -> list[str]:
    return [
        "activity_at",
        "created_at",
        "paid_at",
        "reviewed_at",
        "payment_id",
        "lead_id",
        "parent_name",
        "parent_email",
        "school",
        "payment_type",
        "status",
        "amount",
        "currency",
        "amount_submitted",
        "amount_verified",
        "short_amount",
        "latest_proof_amount",
        "latest_proof_uploaded_at",
        "latest_proof_paid_at",
        "age_days",
    ]


def _payment_report_rows(rows: list[dict[str, Any]]) -> list[list[str]]:
    return [
        [
            _clean_text(row.get("activityAt")),
            _clean_text(row.get("createdAt")),
            _clean_text(row.get("paidAt")),
            _clean_text(row.get("reviewedAt")),
            _clean_text(row.get("paymentId")),
            _clean_text(row.get("leadId")),
            _clean_text(row.get("parentName")),
            _clean_text(row.get("parentEmail")),
            _clean_text(row.get("school")),
            _clean_text(row.get("paymentType")),
            _clean_text(row.get("status")),
            _string_value(row.get("amount")),
            _clean_text(row.get("currency")),
            _string_value(row.get("amountSubmitted")),
            _string_value(row.get("amountVerified")),
            _string_value(row.get("shortAmount")),
            _string_value(row.get("latestProofAmount")),
            _clean_text(row.get("latestProofUploadedAt")),
            _clean_text(row.get("latestProofPaidAt")),
            _string_value(row.get("ageDays")),
        ]
        for row in rows
    ]


def _render_report_markdown(report: AdmissionsPaymentsReport) -> str:
    lines = [
        "# Digital School Admissions + Payment Report",
        "",
        f"Generated: {report.generated_at.isoformat()}",
        f"Period: {_report_period_label(report.request)}",
        f"EOI total: {report.lead_total}",
        f"Payment review total: {report.payment_total}",
        "",
    ]
    for title, headers, rows in _report_tables(report):
        lines.extend([f"## {title}", ""])
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        if rows:
            for row in rows:
                lines.append("| " + " | ".join(_markdown_cell(value) for value in row) + " |")
        else:
            lines.append("| " + " | ".join("-" for _ in headers) + " |")
        lines.append("")
    return "\n".join(lines)


def _render_report_xlsx(report: AdmissionsPaymentsReport) -> bytes:
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            (
                '<?xml version="1.0" encoding="UTF-8"?>'
                f'<Types xmlns="{OPENXML_CONTENT_TYPES_NS}">'
                f'<Default Extension="rels" ContentType="{OPENXML_RELS_CONTENT_TYPE}"/>'
                '<Default Extension="xml" ContentType="application/xml"/>'
                '<Override PartName="/xl/workbook.xml" '
                f'ContentType="{XLSX_WORKBOOK_CONTENT_TYPE}"/>'
                '<Override PartName="/xl/worksheets/sheet1.xml" '
                f'ContentType="{XLSX_WORKSHEET_CONTENT_TYPE}"/>'
                '<Override PartName="/xl/worksheets/sheet2.xml" '
                f'ContentType="{XLSX_WORKSHEET_CONTENT_TYPE}"/>'
                "</Types>"
            ),
        )
        archive.writestr(
            "_rels/.rels",
            (
                '<?xml version="1.0" encoding="UTF-8"?>'
                f'<Relationships xmlns="{OPENXML_PACKAGE_RELS_NS}">'
                '<Relationship Id="rId1" '
                f'Type="{OPENXML_OFFICE_RELS_NS}/officeDocument" '
                'Target="xl/workbook.xml"/>'
                "</Relationships>"
            ),
        )
        archive.writestr(
            "xl/workbook.xml",
            (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                f'xmlns:r="{OPENXML_OFFICE_RELS_NS}">'
                "<sheets>"
                '<sheet name="EOI Latest Payment" sheetId="1" r:id="rId1"/>'
                '<sheet name="Payment Review" sheetId="2" r:id="rId2"/>'
                "</sheets></workbook>"
            ),
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            (
                '<?xml version="1.0" encoding="UTF-8"?>'
                f'<Relationships xmlns="{OPENXML_PACKAGE_RELS_NS}">'
                '<Relationship Id="rId1" '
                f'Type="{OPENXML_OFFICE_RELS_NS}/worksheet" '
                'Target="worksheets/sheet1.xml"/>'
                '<Relationship Id="rId2" '
                f'Type="{OPENXML_OFFICE_RELS_NS}/worksheet" '
                'Target="worksheets/sheet2.xml"/>'
                "</Relationships>"
            ),
        )
        for index, (_title, headers, rows) in enumerate(_report_tables(report), start=1):
            archive.writestr(
                f"xl/worksheets/sheet{index}.xml",
                _xlsx_sheet_xml(headers, rows),
            )
    return output.getvalue()


def _xlsx_sheet_xml(headers: list[str], rows: list[list[str]]) -> str:
    all_rows = [headers, *rows]
    sheet_rows = []
    for row_index, row in enumerate(all_rows, start=1):
        cells = []
        for column_index, value in enumerate(row, start=1):
            ref = f"{_xlsx_column_name(column_index)}{row_index}"
            cells.append(
                f'<c r="{ref}" t="inlineStr"><is><t>{xml_escape(value)}</t></is></c>'
            )
        sheet_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(sheet_rows)}</sheetData>'
        "</worksheet>"
    )


def _xlsx_column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _render_report_docx(report: AdmissionsPaymentsReport) -> bytes:
    body_parts = [
        _docx_paragraph("Digital School Admissions + Payment Report", style="Title"),
        _docx_paragraph(f"Generated: {report.generated_at.isoformat()}"),
        _docx_paragraph(f"Period: {_report_period_label(report.request)}"),
        _docx_paragraph(f"EOI total: {report.lead_total}"),
        _docx_paragraph(f"Payment review total: {report.payment_total}"),
    ]
    for title, headers, rows in _report_tables(report):
        body_parts.append(_docx_paragraph(title, style="Heading1"))
        body_parts.append(_docx_table(headers, rows))
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f'<w:body>{"".join(body_parts)}'
        '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/>'
        '<w:pgMar w:top="720" w:right="720" w:bottom="720" w:left="720"/>'
        "</w:sectPr>"
        "</w:body></w:document>"
    )
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            (
                '<?xml version="1.0" encoding="UTF-8"?>'
                f'<Types xmlns="{OPENXML_CONTENT_TYPES_NS}">'
                f'<Default Extension="rels" ContentType="{OPENXML_RELS_CONTENT_TYPE}"/>'
                '<Default Extension="xml" ContentType="application/xml"/>'
                '<Override PartName="/word/document.xml" '
                f'ContentType="{DOCX_DOCUMENT_CONTENT_TYPE}"/>'
                "</Types>"
            ),
        )
        archive.writestr(
            "_rels/.rels",
            (
                '<?xml version="1.0" encoding="UTF-8"?>'
                f'<Relationships xmlns="{OPENXML_PACKAGE_RELS_NS}">'
                '<Relationship Id="rId1" '
                f'Type="{OPENXML_OFFICE_RELS_NS}/officeDocument" '
                'Target="word/document.xml"/>'
                "</Relationships>"
            ),
        )
        archive.writestr("word/document.xml", document_xml)
    return output.getvalue()


def _docx_paragraph(text: str, style: Optional[str] = None) -> str:
    style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f"<w:p>{style_xml}<w:r><w:t>{xml_escape(text)}</w:t></w:r></w:p>"


def _docx_table(headers: list[str], rows: list[list[str]]) -> str:
    table_rows = [_docx_table_row(headers)]
    table_rows.extend(_docx_table_row(row) for row in rows[:REPORT_EXPORT_LIMIT])
    if len(table_rows) == 1:
        table_rows.append(_docx_table_row(["-" for _ in headers]))
    return "<w:tbl>" + "".join(table_rows) + "</w:tbl>"


def _docx_table_row(values: list[str]) -> str:
    cells = "".join(
        f"<w:tc><w:p><w:r><w:t>{xml_escape(value)}</w:t></w:r></w:p></w:tc>"
        for value in values
    )
    return f"<w:tr>{cells}</w:tr>"


def _render_report_pdf(report: AdmissionsPaymentsReport) -> bytes:
    lines = [
        "Digital School Admissions + Payment Report",
        f"Generated: {report.generated_at.isoformat()}",
        f"Period: {_report_period_label(report.request)}",
        f"EOI total: {report.lead_total}",
        f"Payment review total: {report.payment_total}",
        "",
    ]
    for title, headers, rows in _report_tables(report):
        lines.append(title)
        lines.append(" | ".join(headers))
        for row in rows[:80]:
            lines.append(" | ".join(row))
        if not rows:
            lines.append("-")
        lines.append("")
    return _simple_pdf(lines)


def _simple_pdf(lines: list[str]) -> bytes:
    wrapped = []
    for line in lines:
        wrapped.extend(_wrap_text(line, 110) or [""])
    pages = [wrapped[index : index + 48] for index in range(0, len(wrapped), 48)] or [[]]
    objects: list[bytes] = [b""]
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_numbers = []
    for page_lines in pages:
        stream = _pdf_page_stream(page_lines)
        content_number = len(objects)
        objects.append(
            b"<< /Length "
            + str(len(stream)).encode("ascii")
            + b" >>\nstream\n"
            + stream
            + b"\nendstream"
        )
        page_number = len(objects)
        page_numbers.append(page_number)
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 842 595] "
                f"/Resources << /Font << /F1 3 0 R >> >> "
                f"/Contents {content_number} 0 R >>"
            ).encode("ascii")
        )
    kids = " ".join(f"{number} 0 R" for number in page_numbers)
    objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_numbers)} >>".encode(
        "ascii"
    )
    return _pdf_document(objects)


def _pdf_page_stream(lines: list[str]) -> bytes:
    chunks = ["BT", "/F1 8 Tf", "36 558 Td", "10 TL"]
    for line in lines:
        chunks.append(f"({_pdf_escape(line)}) Tj")
        chunks.append("T*")
    chunks.append("ET")
    return "\n".join(chunks).encode("latin-1", "replace")


def _pdf_document(objects: list[bytes]) -> bytes:
    output = BytesIO()
    output.write(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects[1:], start=1):
        offsets.append(output.tell())
        output.write(f"{index} 0 obj\n".encode("ascii"))
        output.write(obj)
        output.write(b"\nendobj\n")
    xref_offset = output.tell()
    output.write(f"xref\n0 {len(objects)}\n".encode("ascii"))
    output.write(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.write(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.write(
        (
            f"trailer\n<< /Size {len(objects)} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return output.getvalue()


def _pdf_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _wrap_text(value: str, width: int) -> list[str]:
    if len(value) <= width:
        return [value]
    words = value.split()
    lines: list[str] = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            if current:
                lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _string_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return _clean_text(value)


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
    school: str = "",
    search: str = "",
) -> dict[str, str]:
    params = {"limit": str(limit), "offset": str(offset)}
    if date_range is not None:
        params["dateFrom"] = date_range.start.astimezone(timezone.utc).isoformat()
        params["dateTo"] = date_range.end.astimezone(timezone.utc).isoformat()
    if school:
        params["school"] = school
    if search:
        params["search"] = search
    return params


def _extract_student_rows(body: dict[str, Any]) -> list[dict[str, Any]]:
    data = body.get("data")
    if not isinstance(data, dict):
        raise ValueError("service response missing data")
    students = data.get("students")
    if not isinstance(students, list):
        raise ValueError("service response missing students")
    return [student for student in students if isinstance(student, dict)]


def _extract_lead_detail(
    body: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    data = body.get("data")
    if not isinstance(data, dict):
        raise ValueError("service response missing data")
    detail = data.get("detail")
    if not isinstance(detail, dict):
        raise ValueError("service response missing detail")
    lead = detail.get("lead")
    if not isinstance(lead, dict):
        lead = {}
    students = data.get("students")
    if not isinstance(students, list):
        students = []
    return detail, lead, [student for student in students if isinstance(student, dict)]


def _extract_fee_row(body: dict[str, Any]) -> dict[str, Any]:
    data = body.get("data")
    if not isinstance(data, dict):
        raise ValueError("service response missing data")
    amount = data.get("amount")
    if not isinstance(amount, int):
        raise ValueError("service response missing amount")
    return data


def _format_lead_detail_answer(
    row: dict[str, Any],
    detail: dict[str, Any],
    lead: dict[str, Any],
    students: list[dict[str, Any]],
    total: int,
    language: str,
) -> str:
    name = _first_text(lead, row, "parent_name", "parentName") or (
        "Name unavailable" if language == "en" else "Nama belum tersedia"
    )
    email = _first_text(lead, row, "email")
    mobile = _first_text(lead, row, "mobile", "whatsapp")
    school = _first_text(lead, row, "target_school_preference", "school")
    lead_status = _first_text(lead, row, "status", "leadStatus")
    setup_step = _clean_text(lead.get("setupStep"))
    application_id = _clean_text(detail.get("applicationId"))
    application_status = _clean_text(detail.get("applicationStatus"))
    payment_status = _clean_text(detail.get("latestPaymentStatus"))
    payment_type = _clean_text(detail.get("latestPaymentType"))
    payment_amount = detail.get("latestPaymentAmount")

    if language == "en":
        lines = [f"Full EOI detail for {name}:"]
        _append_labeled(lines, "Email", email)
        _append_labeled(lines, "Mobile/WhatsApp", mobile)
        _append_labeled(lines, "Target school", school)
        _append_labeled(lines, "Lead status", lead_status)
        _append_labeled(lines, "Setup step", setup_step)
        _append_labeled(lines, "Application", _status_with_id(application_status, application_id))
        _append_labeled(
            lines,
            "Latest payment",
            _payment_summary(payment_status, payment_type, payment_amount),
        )
        if total > 1:
            lines.append(f"Note: {total} matching EOIs found; showing the top result.")
        lines.extend(_format_student_detail_lines(students, language))
        if not students:
            lines.extend(_format_prospective_child_lines(lead, language))
    else:
        lines = [f"Detail EOI lengkap untuk {name}:"]
        _append_labeled(lines, "Email", email)
        _append_labeled(lines, "Mobile/WhatsApp", mobile)
        _append_labeled(lines, "Sekolah tujuan", school)
        _append_labeled(lines, "Status lead", lead_status)
        _append_labeled(lines, "Setup step", setup_step)
        _append_labeled(lines, "Aplikasi", _status_with_id(application_status, application_id))
        _append_labeled(
            lines,
            "Payment terakhir",
            _payment_summary(payment_status, payment_type, payment_amount),
        )
        if total > 1:
            lines.append(f"Catatan: ada {total} EOI cocok; saya tampilkan hasil teratas.")
        lines.extend(_format_student_detail_lines(students, language))
        if not students:
            lines.extend(_format_prospective_child_lines(lead, language))

    return "\n".join(lines)


def _format_student_detail_lines(students: list[dict[str, Any]], language: str) -> list[str]:
    if not students:
        return ["Children: no submitted child records yet."] if language == "en" else [
            "Anak: belum ada data anak yang sudah disubmit."
        ]

    lines = ["Children:"] if language == "en" else ["Anak:"]
    for index, student in enumerate(students[:5], start=1):
        name = _clean_text(student.get("fullName")) or (
            "Name unavailable" if language == "en" else "Nama belum tersedia"
        )
        fields = []
        _append_detail(
            fields,
            "date of birth" if language == "en" else "tanggal lahir",
            student,
        )
        _append_detail(
            fields,
            "age at application" if language == "en" else "usia saat daftar",
            student,
        )
        _append_detail(fields, "current school" if language == "en" else "sekolah asal", student)
        _append_detail(fields, "target grade" if language == "en" else "grade tujuan", student)
        _append_detail(fields, "target school" if language == "en" else "sekolah tujuan", student)
        _append_detail(fields, "mode", student)
        _append_detail(fields, "status", student)
        suffix = f" ({'; '.join(fields)})" if fields else ""
        lines.append(f"{index}. {name}{suffix}")
    return lines


def _append_detail(fields: list[str], label: str, student: dict[str, Any]) -> None:
    key_aliases = {
        "date of birth": ("dateOfBirth",),
        "tanggal lahir": ("dateOfBirth",),
        "age at application": ("ageAtApplication",),
        "usia saat daftar": ("ageAtApplication",),
        "current school": ("currentSchool",),
        "sekolah asal": ("currentSchool",),
        "target grade": ("targetGradeLevel",),
        "grade tujuan": ("targetGradeLevel",),
        "target school": ("targetSchool",),
        "sekolah tujuan": ("targetSchool",),
        "mode": ("applicationMode",),
        "status": ("applicantStatus",),
    }
    for key in key_aliases[label]:
        raw_value = student.get(key)
        value = _clean_text(raw_value)
        if not value and isinstance(raw_value, int) and not isinstance(raw_value, bool):
            value = str(raw_value)
        if value:
            fields.append(f"{label}: {value}")
            return


def _format_prospective_child_lines(lead: dict[str, Any], language: str) -> list[str]:
    ages = lead.get("prospective_children_ages")
    birth_dates = lead.get("prospective_children_birth_dates")
    if not isinstance(ages, list) and not isinstance(birth_dates, list):
        return []

    lines = ["EOI child estimate:"] if language == "en" else ["Estimasi anak dari EOI:"]
    max_len = max(
        len(ages) if isinstance(ages, list) else 0,
        len(birth_dates) if isinstance(birth_dates, list) else 0,
    )
    for index in range(max_len):
        fields = []
        if isinstance(ages, list) and index < len(ages) and isinstance(ages[index], int):
            fields.append(f"age: {ages[index]}" if language == "en" else f"usia: {ages[index]}")
        if (
            isinstance(birth_dates, list)
            and index < len(birth_dates)
            and isinstance(birth_dates[index], str)
            and birth_dates[index].strip()
        ):
            label = "birth date" if language == "en" else "tanggal lahir"
            fields.append(f"{label}: {birth_dates[index].strip()}")
        if fields:
            lines.append(f"{index + 1}. {'; '.join(fields)}")
    return lines if len(lines) > 1 else []


def _append_labeled(lines: list[str], label: str, value: Optional[str]) -> None:
    if value:
        lines.append(f"- {label}: {value}")


def _status_with_id(status: Optional[str], item_id: Optional[str]) -> Optional[str]:
    if status and item_id:
        return f"{status} ({item_id})"
    return status or item_id


def _payment_summary(
    status: Optional[str],
    payment_type: Optional[str],
    amount: Any,
) -> Optional[str]:
    parts = []
    if status:
        parts.append(status)
    if payment_type:
        parts.append(payment_type)
    if isinstance(amount, int):
        parts.append(_format_money(amount, "IDR"))
    return "; ".join(parts) if parts else None


def _first_text(primary: dict[str, Any], fallback: dict[str, Any], *keys: str) -> Optional[str]:
    for key in keys:
        text = _clean_text(primary.get(key))
        if text:
            return text
    for key in keys:
        text = _clean_text(fallback.get(key))
        if text:
            return text
    return None


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


def _date_scope(date_range: Optional[DateRange], language: str) -> str:
    if date_range is None:
        return ""
    if language != "en":
        return f" {date_range.label}"
    return f" on {_format_date_range_label_en(date_range)}"


def _format_date_range_label_en(date_range: DateRange) -> str:
    start = date_range.start.astimezone(SCHOOL_TIME_ZONE).date()
    end = date_range.end.astimezone(SCHOOL_TIME_ZONE).date()
    if start == end:
        return _format_day_label_en(start)
    return f"{_format_day_label_en(start)} to {_format_day_label_en(end)}"


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


def _format_day_label_en(day) -> str:
    month_names = {
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December",
    }
    return f"{month_names[day.month]} {day.day}, {day.year}"


def _asks_for_current_date(message: str) -> bool:
    direct_phrases = (
        "what day is today",
        "what date is today",
        "what is the date today",
        "today's date",
        "todays date",
        "hari apa hari ini",
        "tanggal berapa hari ini",
        "hari ini hari apa",
        "hari ini tanggal berapa",
        "sekarang tanggal berapa",
    )
    if any(phrase in message for phrase in direct_phrases):
        return True
    has_today = "today" in message or "hari ini" in message
    asks_day_or_date = any(term in message for term in ("day", "date", "hari", "tanggal"))
    operational_terms = (
        "eoi",
        "lead",
        "payment",
        "pembayaran",
        "pendaftar",
        "registrasi",
        "admission",
        "admissions",
    )
    return has_today and asks_day_or_date and not any(term in message for term in operational_terms)


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


def _asks_for_admissions_payments_report(message: str) -> bool:
    report_terms = (
        "report",
        "laporan",
        "export",
        "download",
        "excel",
        "xlsx",
        "pdf",
        "word",
        "docx",
        "markdown",
        "spreadsheet",
        "table",
        "tabel",
        "show me",
        "tampilkan",
        "lihat",
        "list",
        "daftar",
    )
    eoi_terms = (
        "eoi",
        "lead",
        "admission",
        "admissions",
        "pendaftar",
        "pendaftaran",
        "registrasi",
        "registration",
    )
    payment_terms = (
        "payment",
        "payments",
        "pembayaran",
        "tagihan",
        "invoice",
        "finance",
    )
    has_eoi = any(term in message for term in eoi_terms)
    has_payment = any(term in message for term in payment_terms)
    has_report = any(term in message for term in report_terms)
    return has_eoi and has_payment and has_report


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
    detail_match = re.search(
        r"^(?:Detail EOI lengkap untuk|Full EOI detail for)\s+([^:\n]+):",
        message,
        flags=re.MULTILINE,
    )
    if detail_match:
        return _clean_search_query(detail_match.group(1))
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


def _explicit_payment_status_from_message(message: str) -> Optional[str]:
    status_terms = (
        "underpaid",
        "kurang bayar",
        "rejected",
        "ditolak",
        "paid",
        "lunas",
        "approved",
        "pending",
        "menunggu",
        "verification",
        "verifikasi",
    )
    if not any(term in message for term in status_terms):
        return None
    return _payment_status_from_message(message)


def _asks_for_all_time(message: str) -> bool:
    all_time_terms = (
        "all time",
        "all-time",
        "all of time",
        "semua waktu",
        "semua data",
        "seluruh waktu",
        "dari awal",
        "sejak awal",
    )
    return any(term in message for term in all_time_terms)


def _payment_status_label(status: str, language: str = "id") -> str:
    if language == "en":
        if status == "pending_verification":
            return "pending verification"
        if status == "paid":
            return "already paid"
        if status == "rejected":
            return "rejected"
        if status == "underpaid":
            return "underpaid"
        return f"with status {status}"
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


def _payment_type_label(payment_type: str, language: str = "id") -> str:
    if language == "en":
        if payment_type == "enrolment_fee":
            return "enrolment fee"
        if payment_type == "application_fee":
            return "application fee"
        return payment_type.replace("_", " ")
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


def _answer_language(payload: ChatRequest, lowered_message: str) -> str:
    if payload.locale.casefold().startswith("en"):
        return "en"
    english_terms = (
        "how many",
        "how much",
        "what is",
        "what are",
        "what day",
        "what date",
        "who is",
        "who are",
        "which",
        "show me",
        "show the",
        "list the",
        "more detail",
        "more details",
        "full detail",
        "show details",
        "details",
        "registered",
        "pending verification",
        "application fee",
        "translate",
        "english",
        "in english",
        "today",
        "children",
        "student",
        "students",
    )
    return "en" if any(term in lowered_message for term in english_terms) else "id"


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

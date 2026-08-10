import asyncio
import json
import secrets
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Optional
from uuid import uuid4
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import Response, StreamingResponse

from app.auth_context import AdminContext, resolve_admin_context
from app.config import Settings, get_settings
from app.document_analysis import DocumentAnalysisResponse, analyze_birth_certificate
from app.llm_client import LlmClient
from app.policy import is_allowed_for_message, requires_privileged_role
from app.schemas import (
    ActorRole,
    ChatRequest,
    ChatResponse,
    ChatTurn,
    CreateThreadRequest,
    DeleteThreadResponse,
    HealthResponse,
    MetadataResponse,
    SourceRef,
    ThreadAuditResponse,
    ThreadListResponse,
    ThreadResponse,
    ToolCallRef,
)
from app.service_tools import (
    answer_from_school_tools,
    build_admissions_payments_report,
    build_thread_context,
    render_admissions_payments_report,
    report_request_from_export_query,
)
from app.thread_store import ThreadNotFound, ThreadStore

app = FastAPI(
    title="Digital Schools AI Service",
    version="0.1.0",
    docs_url="/api/v1/ai-service/swagger-ui",
    openapi_url="/api/v1/ai-service/api-docs/openapi.json",
)

SETTINGS_DEPENDENCY = Depends(get_settings)
AUTH_HEADER = Header(default=None)
PRIVILEGED_DATA_ROLES = {ActorRole.owner, ActorRole.admin, ActorRole.teacher}
SCHOOL_TIME_ZONE = ZoneInfo("Asia/Jakarta")
DRAFT_STREAM_CHUNK_TIMEOUT_SECONDS = 20.0
ID_DAY_NAMES = ("Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu")
ID_MONTH_NAMES = (
    "Januari",
    "Februari",
    "Maret",
    "April",
    "Mei",
    "Juni",
    "Juli",
    "Agustus",
    "September",
    "Oktober",
    "November",
    "Desember",
)
EN_DAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
EN_MONTH_NAMES = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


@app.get("/api/v1/ai-service/health", response_model=HealthResponse)
async def health(settings: Settings = SETTINGS_DEPENDENCY) -> HealthResponse:
    return HealthResponse(status="ok", service=settings.service_name, environment=settings.app_env)


@app.get("/api/v1/ai-service/metadata", response_model=MetadataResponse)
async def metadata(settings: Settings = SETTINGS_DEPENDENCY) -> MetadataResponse:
    return MetadataResponse(
        service=settings.service_name,
        model=settings.ai_model,
        max_tokens=settings.ai_max_tokens,
        provider_configured=settings.ai_provider_base_url is not None,
        sis_service_url=settings.sis_service_url,
        admission_service_url=settings.admission_service_url,
        payment_service_url=settings.payment_service_url,
    )


@app.post(
    "/api/ai/v1/internal/document-analysis",
    response_model=DocumentAnalysisResponse,
    response_model_by_alias=True,
)
async def document_analysis(
    request: Request,
    authorization: Optional[str] = AUTH_HEADER,
    content_type: Optional[str] = Header(default=None, alias="content-type"),
    expected_document_type: Optional[str] = Header(
        default=None, alias="x-expected-document-type"
    ),
    settings: Settings = SETTINGS_DEPENDENCY,
) -> DocumentAnalysisResponse:
    if not settings.document_analysis_enabled:
        raise HTTPException(status_code=503, detail="Document analysis is disabled")
    expected_token = settings.document_analysis_internal_token
    supplied_token = authorization.removeprefix("Bearer ") if authorization else ""
    if not expected_token or not secrets.compare_digest(supplied_token, expected_token):
        raise HTTPException(status_code=401, detail="Invalid internal credentials")
    if expected_document_type != "birth_certificate":
        raise HTTPException(status_code=422, detail="Unsupported document type")
    media_type = (content_type or "").split(";", 1)[0].strip().lower()
    try:
        chunks = bytearray()
        async for chunk in request.stream():
            chunks.extend(chunk)
            if len(chunks) > settings.document_analysis_max_bytes:
                raise ValueError("file size is outside the allowed range")
        return await analyze_birth_certificate(bytes(chunks), media_type, settings)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Document analysis failed") from exc


@app.get("/api/ai/v1/threads", response_model=ThreadListResponse)
async def list_threads(
    authorization: Optional[str] = AUTH_HEADER,
    settings: Settings = SETTINGS_DEPENDENCY,
) -> ThreadListResponse:
    admin_context = await resolve_admin_context(settings, authorization)
    try:
        threads = await ThreadStore(settings).list_threads(admin_context.owner_id)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI thread history unavailable",
        ) from exc
    return ThreadListResponse(threads=threads)


@app.post("/api/ai/v1/threads", response_model=ThreadResponse)
async def create_thread(
    payload: CreateThreadRequest,
    authorization: Optional[str] = AUTH_HEADER,
    settings: Settings = SETTINGS_DEPENDENCY,
) -> ThreadResponse:
    admin_context = await resolve_admin_context(settings, authorization)
    store = ThreadStore(settings)
    try:
        record = await store.create_thread(admin_context.owner_id, payload.title)
        summary, messages = await store.get_thread(admin_context.owner_id, record["id"])
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI thread history unavailable",
        ) from exc
    return ThreadResponse(thread=summary, messages=messages)


@app.get("/api/ai/v1/threads/{thread_id}", response_model=ThreadResponse)
async def get_thread(
    thread_id: str,
    authorization: Optional[str] = AUTH_HEADER,
    settings: Settings = SETTINGS_DEPENDENCY,
) -> ThreadResponse:
    admin_context = await resolve_admin_context(settings, authorization)
    try:
        summary, messages = await ThreadStore(settings).get_thread(
            admin_context.owner_id,
            thread_id,
        )
    except ThreadNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thread not found",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI thread history unavailable",
        ) from exc
    return ThreadResponse(thread=summary, messages=messages)


@app.get("/api/ai/v1/threads/{thread_id}/audit", response_model=ThreadAuditResponse)
async def get_thread_audit(
    thread_id: str,
    authorization: Optional[str] = AUTH_HEADER,
    settings: Settings = SETTINGS_DEPENDENCY,
) -> ThreadAuditResponse:
    admin_context = await resolve_admin_context(settings, authorization)
    try:
        events = await ThreadStore(settings).get_audit_events(
            admin_context.owner_id,
            thread_id,
        )
    except ThreadNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thread not found",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI thread audit unavailable",
        ) from exc
    return ThreadAuditResponse(events=events)


@app.delete("/api/ai/v1/threads/{thread_id}", response_model=DeleteThreadResponse)
async def delete_thread(
    thread_id: str,
    authorization: Optional[str] = AUTH_HEADER,
    settings: Settings = SETTINGS_DEPENDENCY,
) -> DeleteThreadResponse:
    admin_context = await resolve_admin_context(settings, authorization)
    try:
        await ThreadStore(settings).delete_thread(admin_context.owner_id, thread_id)
    except ThreadNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thread not found",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI thread history unavailable",
        ) from exc
    return DeleteThreadResponse()


@app.get("/api/ai/v1/reports/admissions-payments")
async def export_admissions_payments_report(
    export_format: str = Query(default="xlsx", alias="format", max_length=8),
    date_from: Optional[str] = Query(default=None, alias="dateFrom", max_length=64),
    date_to: Optional[str] = Query(default=None, alias="dateTo", max_length=64),
    payment_status: Optional[str] = Query(default=None, alias="paymentStatus", max_length=64),
    school: Optional[str] = Query(default=None, max_length=16),
    search: Optional[str] = Query(default=None, max_length=128),
    limit: int = Query(default=500, ge=1, le=500),
    locale: str = Query(default="id", max_length=8),
    authorization: Optional[str] = AUTH_HEADER,
    settings: Settings = SETTINGS_DEPENDENCY,
) -> Response:
    await resolve_admin_context(settings, authorization)
    try:
        request = report_request_from_export_query(
            date_from=date_from,
            date_to=date_to,
            payment_status=payment_status,
            school=school,
            search=search,
            limit=limit,
            language=locale,
        )
        report = await build_admissions_payments_report(settings, authorization, request)
        export_file = render_admissions_payments_report(report, export_format)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI report export unavailable",
        ) from exc

    return Response(
        content=export_file.body,
        media_type=export_file.media_type,
        headers={
            "Cache-Control": "no-store",
            "Content-Disposition": f'attachment; filename="{export_file.filename}"',
        },
    )


@app.post("/api/ai/v1/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    authorization: Optional[str] = AUTH_HEADER,
    settings: Settings = SETTINGS_DEPENDENCY,
) -> ChatResponse:
    payload, thread_store, admin_context = await _prepare_chat_exchange(
        payload,
        authorization,
        settings,
    )
    response = await _resolve_chat_response(payload, authorization, settings)
    return await _persist_thread_exchange(response, payload, thread_store, admin_context)


@app.post("/api/ai/v1/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    authorization: Optional[str] = AUTH_HEADER,
    settings: Settings = SETTINGS_DEPENDENCY,
) -> StreamingResponse:
    payload, thread_store, admin_context = await _prepare_chat_exchange(
        payload,
        authorization,
        settings,
    )
    if payload.conversation_id is None:
        payload = payload.model_copy(update={"conversation_id": str(uuid4())})

    return StreamingResponse(
        _stream_chat_events(payload, authorization, settings, thread_store, admin_context),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _prepare_chat_exchange(
    payload: ChatRequest,
    authorization: Optional[str],
    settings: Settings,
) -> tuple[ChatRequest, Optional[ThreadStore], Optional[AdminContext]]:
    if payload.actor_role != ActorRole.public and not _has_bearer_token(authorization):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token required for non-public AI requests",
        )

    admin_context: Optional[AdminContext] = None
    thread_store: Optional[ThreadStore] = None
    if payload.actor_role in {ActorRole.owner, ActorRole.admin}:
        admin_context = await resolve_admin_context(settings, authorization)
        payload, thread_store = await _with_server_thread_history(
            payload,
            settings,
            admin_context,
        )

    return payload, thread_store, admin_context


async def _resolve_chat_response(
    payload: ChatRequest,
    authorization: Optional[str],
    settings: Settings,
) -> ChatResponse:
    if not is_allowed_for_message(payload.actor_role, payload.message):
        return ChatResponse(
            conversation_id=payload.conversation_id or str(uuid4()),
            status="refused",
            answer=(
                "Saya belum bisa membantu pertanyaan itu untuk role ini. "
                "Data siswa, nilai, pembayaran, dan ranking hanya boleh diakses lewat role "
                "yang berwenang."
            ),
        )

    static_answer = _deterministic_general_answer(payload)
    if static_answer is not None:
        return ChatResponse(
            conversation_id=payload.conversation_id or str(uuid4()),
            answer=static_answer,
            sources=[SourceRef(kind="system", title="AI service Jakarta clock", reference=None)],
            tool_calls=[ToolCallRef(name="system.current_date", status="ok")],
        )

    tool_response = await answer_from_school_tools(payload, settings, authorization)
    if tool_response is not None:
        return ChatResponse(
            conversation_id=payload.conversation_id or str(uuid4()),
            answer=tool_response.answer,
            sources=tool_response.sources,
            tool_calls=tool_response.tool_calls,
        )

    if (
        payload.actor_role in PRIVILEGED_DATA_ROLES
        and requires_privileged_role(payload.message)
    ):
        return ChatResponse(
            conversation_id=payload.conversation_id or str(uuid4()),
            answer=_fallback_answer(payload),
            sources=_sources_for_message(payload.message),
            tool_calls=_tool_calls_for_message(payload.message),
        )

    answer = await _draft_answer(payload, settings)
    return ChatResponse(
        conversation_id=payload.conversation_id or str(uuid4()),
        answer=answer,
        sources=_sources_for_message(payload.message),
        tool_calls=_tool_calls_for_message(payload.message),
    )


async def _stream_chat_events(
    payload: ChatRequest,
    authorization: Optional[str],
    settings: Settings,
    thread_store: Optional[ThreadStore],
    admin_context: Optional[AdminContext],
) -> AsyncIterator[str]:
    yield _sse_event("thread", {"conversationId": payload.conversation_id})
    yield _sse_event("status", {"label": "understanding"})

    if not is_allowed_for_message(payload.actor_role, payload.message):
        response = ChatResponse(
            conversation_id=payload.conversation_id,
            status="refused",
            answer=(
                "Saya belum bisa membantu pertanyaan itu untuk role ini. "
                "Data siswa, nilai, pembayaran, dan ranking hanya boleh diakses lewat role "
                "yang berwenang."
            ),
        )
        async for event in _stream_final_response(response, payload, thread_store, admin_context):
            yield event
        return

    static_answer = _deterministic_general_answer(payload)
    if static_answer is not None:
        response = ChatResponse(
            conversation_id=payload.conversation_id,
            answer=static_answer,
            sources=[SourceRef(kind="system", title="AI service Jakarta clock", reference=None)],
            tool_calls=[ToolCallRef(name="system.current_date", status="ok")],
        )
        async for event in _stream_final_response(response, payload, thread_store, admin_context):
            yield event
        return

    yield _sse_event("status", {"label": "checking_tools"})
    tool_response = await answer_from_school_tools(payload, settings, authorization)
    if tool_response is not None:
        response = ChatResponse(
            conversation_id=payload.conversation_id,
            answer=tool_response.answer,
            sources=tool_response.sources,
            tool_calls=tool_response.tool_calls,
        )
        async for event in _stream_final_response(response, payload, thread_store, admin_context):
            yield event
        return

    if (
        payload.actor_role in PRIVILEGED_DATA_ROLES
        and requires_privileged_role(payload.message)
    ):
        response = ChatResponse(
            conversation_id=payload.conversation_id,
            answer=_fallback_answer(payload),
            sources=_sources_for_message(payload.message),
            tool_calls=_tool_calls_for_message(payload.message),
        )
        async for event in _stream_final_response(response, payload, thread_store, admin_context):
            yield event
        return

    yield _sse_event("status", {"label": "drafting"})
    chunks: list[str] = []
    async for chunk in _draft_answer_chunks(payload, settings):
        chunks.append(chunk)
        yield _sse_event("delta", {"text": chunk})

    answer = "".join(chunks).strip() or _fallback_answer(payload)
    if not chunks:
        for chunk in _text_chunks(answer):
            yield _sse_event("delta", {"text": chunk})

    response = ChatResponse(
        conversation_id=payload.conversation_id,
        answer=answer,
        sources=_sources_for_message(payload.message),
        tool_calls=_tool_calls_for_message(payload.message),
    )
    response = await _persist_thread_exchange(response, payload, thread_store, admin_context)
    yield _sse_event("done", response.model_dump(by_alias=True))


async def _stream_final_response(
    response: ChatResponse,
    payload: ChatRequest,
    thread_store: Optional[ThreadStore],
    admin_context: Optional[AdminContext],
) -> AsyncIterator[str]:
    for tool in response.tool_calls:
        yield _sse_event("tool_call", tool.model_dump(by_alias=True))
    yield _sse_event("status", {"label": "drafting"})
    for chunk in _text_chunks(response.answer):
        yield _sse_event("delta", {"text": chunk})
    response = await _persist_thread_exchange(response, payload, thread_store, admin_context)
    yield _sse_event("done", response.model_dump(by_alias=True))


async def _with_server_thread_history(
    payload: ChatRequest,
    settings: Settings,
    admin_context: AdminContext,
) -> tuple[ChatRequest, Optional[ThreadStore]]:
    store = ThreadStore(settings)
    try:
        record = await store.ensure_thread(
            admin_context.owner_id,
            payload.conversation_id,
            payload.message,
        )
    except Exception:
        return payload, None

    history = _history_from_thread_record(record)
    if not history:
        history = payload.history
    metadata = dict(payload.metadata)
    thread_context = _context_from_thread_record(record)
    if thread_context:
        metadata["threadContext"] = thread_context

    return (
        payload.model_copy(
            update={
                "actor_role": admin_context.actor_role,
                "conversation_id": record["id"],
                "history": history,
                "metadata": metadata,
            }
        ),
        store,
    )


async def _persist_thread_exchange(
    response: ChatResponse,
    payload: ChatRequest,
    thread_store: Optional[ThreadStore],
    admin_context: Optional[AdminContext],
) -> ChatResponse:
    if thread_store is None or admin_context is None or response.conversation_id is None:
        return response
    try:
        context = build_thread_context(
            payload,
            response,
            _context_from_payload(payload),
        )
        await thread_store.append_exchange(
            admin_context.owner_id,
            response.conversation_id,
            payload.message,
            response.answer,
            response.status,
            response.sources,
            response.tool_calls,
            context,
        )
    except Exception:
        return response
    return response


def _history_from_thread_record(record: dict) -> list[ChatTurn]:
    rows = record.get("messages")
    if not isinstance(rows, list):
        return []
    turns: list[ChatTurn] = []
    for row in rows[-20:]:
        if not isinstance(row, dict):
            continue
        role = row.get("role")
        content = row.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str) or not content.strip():
            continue
        turns.append(
            ChatTurn(
                role=role,
                content=content,
                status=row.get("status") if isinstance(row.get("status"), str) else None,
                tool_calls=[
                    ToolCallRef.model_validate(tool)
                    for tool in row.get("toolCalls", [])
                    if isinstance(tool, dict)
                ],
            )
        )
    return turns


def _context_from_thread_record(record: dict) -> dict:
    context = record.get("context")
    return context if isinstance(context, dict) else {}


def _context_from_payload(payload: ChatRequest) -> dict:
    context = payload.metadata.get("threadContext")
    return context if isinstance(context, dict) else {}


async def _draft_answer(payload: ChatRequest, settings: Settings) -> str:
    local_fallback = _fallback_answer(payload)
    client = LlmClient(settings)
    try:
        generated = await client.complete(
            system_prompt=_draft_system_prompt(),
            user_message=_draft_user_message(payload),
        )
    except Exception:
        generated = None
    return generated or local_fallback


async def _draft_answer_chunks(
    payload: ChatRequest,
    settings: Settings,
) -> AsyncIterator[str]:
    client = LlmClient(settings)
    stream_timeout = min(settings.request_timeout_seconds, DRAFT_STREAM_CHUNK_TIMEOUT_SECONDS)
    stream = client.stream_complete(
        system_prompt=_draft_system_prompt(),
        user_message=_draft_user_message(payload),
        timeout_seconds=stream_timeout,
    )
    try:
        while True:
            chunk = await asyncio.wait_for(stream.__anext__(), timeout=stream_timeout)
            yield chunk
    except StopAsyncIteration:
        return
    except Exception:
        if hasattr(stream, "aclose"):
            await stream.aclose()
        for chunk in _text_chunks(_fallback_answer(payload)):
            yield chunk


def _draft_system_prompt() -> str:
    now = _now_jakarta()
    return (
        "You are the Digital Schools AI assistant. Answer in Indonesian unless asked otherwise. "
        "Do not invent school facts, student records, grades, payment data, or dates. "
        "If official source data or a required tool is unavailable, say that clearly. "
        f"The current date in Asia/Jakarta is {_format_date_en(now)} "
        f"({_format_date_id(now)})."
    )


def _draft_user_message(payload: ChatRequest) -> str:
    safe_history = []
    for turn in payload.history[-8:]:
        if turn.tool_calls:
            continue
        if requires_privileged_role(turn.content):
            continue
        safe_history.append(f"{turn.role}: {turn.content.strip()}")

    if not safe_history:
        return payload.message

    history_text = "\n".join(safe_history)
    return (
        "Recent safe conversation context:\n"
        f"{history_text}\n\n"
        "Current user message:\n"
        f"{payload.message}"
    )


def _fallback_answer(payload: ChatRequest) -> str:
    lowered = payload.message.casefold()
    if any(term in lowered for term in ("marketing", "ppdb", "campaign", "promosi")):
        return (
            "Untuk draft marketing PPDB, saya bisa bantu membuat variasi copy, headline, dan CTA. "
            "Sumber resmi sekolah belum dihubungkan ke AI service, jadi hasil awal harus "
            "direview oleh tim sekolah."
        )
    if "berdiri" in lowered or "founded" in lowered:
        return (
            "Saya belum menemukan sumber resmi tentang tanggal berdiri sekolah di knowledge "
            "base AI. "
            "Tambahkan profil sekolah resmi dulu agar jawaban bisa grounded dan tidak mengarang."
        )
    if requires_privileged_role(payload.message):
        return (
            "Saya bisa bantu data operasional admin kalau ada tool resmi yang cocok. "
            "Tool yang aktif saat ini: jumlah dan daftar EOI, detail lead dan anak, "
            "funnel lead per step, lead yang stuck, appointment, lead paid tanpa "
            "appointment, review pembayaran, biaya pendaftaran, laporan EOI + pembayaran, "
            "dan briefing operasional. Untuk pertanyaan ini saya belum menemukan tool yang "
            "tepat, jadi saya tidak akan mengarang data. Coba tanyakan salah satu data itu."
        )
    return (
        "AI service sudah aktif sebagai gateway awal. Untuk jawaban faktual, hubungkan "
        "dokumen resmi "
        "atau tool dari service pemilik data terlebih dahulu."
    )


def _deterministic_general_answer(payload: ChatRequest) -> Optional[str]:
    lowered = payload.message.casefold().strip()
    if not _asks_current_date(lowered):
        return None

    now = _now_jakarta()
    if _prefers_english(lowered, payload.locale):
        return f"Today is {_format_date_en(now)} in Asia/Jakarta."
    return f"Hari ini {_format_date_id(now)} di zona waktu Asia/Jakarta."


def _asks_current_date(message: str) -> bool:
    date_terms = (
        "what day is today",
        "what date is today",
        "what is today",
        "today date",
        "current date",
        "tanggal berapa hari ini",
        "hari apa hari ini",
        "sekarang tanggal berapa",
        "tanggal hari ini",
    )
    return any(term in message for term in date_terms)


def _prefers_english(message: str, locale: str) -> bool:
    if locale.casefold().startswith("en"):
        return True
    english_terms = ("what day", "what date", "current date", "today")
    return any(term in message for term in english_terms)


def _format_date_id(value: datetime) -> str:
    return (
        f"{ID_DAY_NAMES[value.weekday()]}, {value.day} "
        f"{ID_MONTH_NAMES[value.month - 1]} {value.year}"
    )


def _format_date_en(value: datetime) -> str:
    return (
        f"{EN_DAY_NAMES[value.weekday()]}, {EN_MONTH_NAMES[value.month - 1]} "
        f"{value.day}, {value.year}"
    )


def _now_jakarta() -> datetime:
    return datetime.now(SCHOOL_TIME_ZONE)


def _sources_for_message(message: str) -> list[SourceRef]:
    if requires_privileged_role(message):
        return [SourceRef(kind="service", title="SIS/admission tool required", reference=None)]
    return []


def _tool_calls_for_message(message: str) -> list[ToolCallRef]:
    if requires_privileged_role(message):
        return [ToolCallRef(name="pending.safe_school_data_tool", status="not_configured")]
    return []


def _has_bearer_token(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.casefold().startswith("bearer ") and len(value.split(" ", 1)[1].strip()) > 0


def _sse_event(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n"


def _text_chunks(text: str, size: int = 28) -> list[str]:
    if len(text) <= size:
        return [text] if text else []

    chunks: list[str] = []
    current = ""
    for word in text.split(" "):
        next_value = f"{current} {word}" if current else word
        if len(next_value) <= size:
            current = next_value
            continue
        if current:
            chunks.append(f"{current} ")
        current = word
    if current:
        chunks.append(current)
    return chunks

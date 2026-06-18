import json
from collections.abc import AsyncIterator
from typing import Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import StreamingResponse

from app.auth_context import AdminContext, resolve_admin_context
from app.config import Settings, get_settings
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
    ThreadListResponse,
    ThreadResponse,
    ToolCallRef,
)
from app.service_tools import answer_from_school_tools
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

    return (
        payload.model_copy(
            update={
                "actor_role": admin_context.actor_role,
                "conversation_id": record["id"],
                "history": history,
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
        await thread_store.append_exchange(
            admin_context.owner_id,
            response.conversation_id,
            payload.message,
            response.answer,
            response.status,
            response.sources,
            response.tool_calls,
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
    try:
        async for chunk in client.stream_complete(
            system_prompt=_draft_system_prompt(),
            user_message=_draft_user_message(payload),
        ):
            yield chunk
    except Exception:
        for chunk in _text_chunks(_fallback_answer(payload)):
            yield chunk


def _draft_system_prompt() -> str:
    return (
        "You are the Digital Schools AI assistant. Answer in Indonesian unless asked otherwise. "
        "Do not invent school facts, student records, grades, payment data, or dates. "
        "If official source data or a required tool is unavailable, say that clearly."
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
            "Tool yang aktif saat ini mencakup EOI, detail anak pada lead, review pembayaran, "
            "dan biaya pendaftaran. Untuk pertanyaan ini saya belum menemukan tool yang tepat, "
            "jadi saya tidak akan mengarang data."
        )
    return (
        "AI service sudah aktif sebagai gateway awal. Untuk jawaban faktual, hubungkan "
        "dokumen resmi "
        "atau tool dari service pemilik data terlebih dahulu."
    )


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

from typing import Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, status

from app.config import Settings, get_settings
from app.llm_client import LlmClient
from app.policy import is_allowed_for_message, requires_privileged_role
from app.schemas import (
    ActorRole,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    MetadataResponse,
    SourceRef,
    ToolCallRef,
)
from app.service_tools import answer_from_school_tools

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


@app.post("/api/ai/v1/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    authorization: Optional[str] = AUTH_HEADER,
    settings: Settings = SETTINGS_DEPENDENCY,
) -> ChatResponse:
    if payload.actor_role != ActorRole.public and not _has_bearer_token(authorization):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token required for non-public AI requests",
        )

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


async def _draft_answer(payload: ChatRequest, settings: Settings) -> str:
    local_fallback = _fallback_answer(payload)
    client = LlmClient(settings)
    system_prompt = (
        "You are the Digital Schools AI assistant. Answer in Indonesian unless asked otherwise. "
        "Do not invent school facts, student records, grades, payment data, or dates. "
        "If official source data or a required tool is unavailable, say that clearly."
    )
    try:
        generated = await client.complete(system_prompt=system_prompt, user_message=payload.message)
    except Exception:
        generated = None
    return generated or local_fallback


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
            "Pertanyaan ini perlu tool data resmi dari SIS/admission/payment service. "
            "Endpoint AI sudah siap sebagai gateway, tetapi tool backend untuk data tersebut "
            "belum diaktifkan."
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

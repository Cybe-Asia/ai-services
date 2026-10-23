from functools import lru_cache
from typing import Literal, Optional
from urllib.parse import urlsplit

from pydantic import Field, HttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

FEATURE_PROVIDERS = ("discovery", "assistant", "chat", "essay", "selection", "drafts")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    service_name: str = "ai-service"
    app_env: str = "local"
    server_port: int = 8082

    # Production runs qwen3:4b via the shared on-prem CYBE AI Gateway
    # (AI_PROVIDER_BASE_URL) — chosen for CPU-inference latency. Override
    # AI_MODEL locally if your provider serves a different model.
    ai_model: str = "qwen3:4b"
    ai_provider_base_url: Optional[HttpUrl] = None
    ai_provider_api_key: Optional[str] = None
    # Explicit deployment protocol avoids routing by model family/host naming.
    # auto retains the existing local Ollama compatibility behavior.
    ai_provider_protocol: Literal[
        "auto", "openai_compatible", "ollama_native", "anthropic", "openai"
    ] = "auto"
    # OpenAI reasoning models only; unset leaves the model's own default.
    ai_reasoning_effort: Optional[Literal["none", "minimal", "low", "medium", "high"]] = None
    ai_provider_message_max_chars: Optional[int] = Field(default=None, ge=1, le=32768)
    ai_max_tokens: int = Field(default=256, ge=1, le=2048)

    # Optional provider per feature. Each block is used only when its base URL, key and
    # model are all set; otherwise that feature stays on the main provider above.
    # - discovery: Student Discovery reflection and the Activity Arena conclusion. The
    #   prompt carries only teacher-published material, released feedback and counts.
    # - assistant: Kibo's explanation of one selected published reading.
    # - chat: Admin assist intent classification and drafting (never Student chat).
    # - essay: Teacher essay-review suggestions (ADR 0003). Unlike the others this block has
    #   no fallback: unset means the feature is off, because it sends student-written text.
    discovery_provider_protocol: Literal["openai_compatible", "anthropic", "openai"] = "anthropic"
    discovery_provider_base_url: Optional[HttpUrl] = None
    discovery_provider_api_key: Optional[str] = None
    discovery_model: Optional[str] = None
    discovery_max_tokens: int = Field(default=768, ge=1, le=2048)
    # The reflection is a short grounded rewrite; reasoning only adds latency and cost.
    discovery_reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] = "none"
    discovery_timeout_seconds: float = Field(default=20.0, ge=1.0, le=60.0)

    assistant_provider_protocol: Literal["openai_compatible", "anthropic", "openai"] = "openai"
    assistant_provider_base_url: Optional[HttpUrl] = None
    assistant_provider_api_key: Optional[str] = None
    assistant_model: Optional[str] = None
    assistant_max_tokens: int = Field(default=512, ge=1, le=2048)
    assistant_reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] = "none"
    assistant_timeout_seconds: float = Field(default=20.0, ge=1.0, le=60.0)

    chat_provider_protocol: Literal["openai_compatible", "anthropic", "openai"] = "openai"
    chat_provider_base_url: Optional[HttpUrl] = None
    chat_provider_api_key: Optional[str] = None
    chat_model: Optional[str] = None
    chat_max_tokens: int = Field(default=512, ge=1, le=2048)
    chat_reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] = "none"
    chat_timeout_seconds: float = Field(default=30.0, ge=1.0, le=120.0)
    essay_provider_protocol: Literal["openai_compatible", "anthropic", "openai"] = "openai"
    essay_provider_base_url: Optional[HttpUrl] = None
    essay_provider_api_key: Optional[str] = None
    essay_model: Optional[str] = None
    # Up to eight criteria with quotes and reasons plus feedback; a cut answer is rejected.
    essay_max_tokens: int = Field(default=1200, ge=1, le=2048)
    # Grading needs more care than summarising (ADR 0003 §5).
    essay_reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] = "low"
    essay_timeout_seconds: float = Field(default=30.0, ge=1.0, le=120.0)
    # Learning calls POST /api/ai/v1/internal/essay-review at submit time with this bearer
    # (ADR 0010). Unset keeps that endpoint closed; the teacher endpoint is unaffected.
    essay_internal_token: Optional[str] = None
    # selection: a teacher's AI-proposed question split for an assessment (ADR 0011). The
    # prompt carries bank counts and short prompt excerpts of the teacher's own questions.
    # Unset means the feature is off (the Teacher web hides the button).
    selection_provider_protocol: Literal["openai_compatible", "anthropic", "openai"] = "openai"
    selection_provider_base_url: Optional[HttpUrl] = None
    selection_provider_api_key: Optional[str] = None
    selection_model: Optional[str] = None
    selection_max_tokens: int = Field(default=800, ge=1, le=2048)
    selection_reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] = "none"
    selection_timeout_seconds: float = Field(default=30.0, ge=1.0, le=120.0)
    # drafts: questions an AI drafts for a teacher from a module's chapter readings (ADR 0011
    # §9 follow-up). The prompt carries the chosen lessons' text and the teacher's settings;
    # ten bilingual questions with options and explanations need a larger answer.
    # Unset means the feature is off (the Teacher web hides the button).
    drafts_provider_protocol: Literal["openai_compatible", "anthropic", "openai"] = "openai"
    drafts_provider_base_url: Optional[HttpUrl] = None
    drafts_provider_api_key: Optional[str] = None
    drafts_model: Optional[str] = None
    drafts_max_tokens: int = Field(default=8000, ge=1, le=16000)
    drafts_reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] = "low"
    drafts_timeout_seconds: float = Field(default=45.0, ge=1.0, le=120.0)

    # Admin assist: when no school tool matches an owner/admin/teacher question, answer with
    # the tool catalogue instead of a free model draft. Public chat is not affected.
    chat_open_drafting: bool = False

    # In-cluster recitation worker (app.recitation_service) for the Student tahfiz
    # pre-check. Unset keeps the pre-check unavailable (503); no other feature uses it.
    recitation_service_url: Optional[str] = None
    recitation_timeout_seconds: float = Field(default=45.0, ge=5.0, le=120.0)

    sis_service_url: str = "http://sis-service"
    learning_service_url: str = "https://learning-api"
    admission_service_url: str = "http://admission-service"
    payment_service_url: str = "http://payment-service"
    request_timeout_seconds: float = Field(default=30.0, ge=1.0, le=120.0)

    redis_url: Optional[str] = None
    ai_thread_retention_days: int = Field(default=30, ge=1, le=365)
    ai_thread_max_threads: int = Field(default=50, ge=1, le=500)
    ai_thread_max_messages: int = Field(default=100, ge=1, le=500)

    # Private document-analysis surface. Admission service is the only caller;
    # it retains ownership of storage and applicant facts.
    document_analysis_enabled: bool = False
    document_analysis_internal_token: Optional[str] = None
    document_analysis_adapter: Literal["openai_compatible", "cybe_gateway_vision"] = (
        "openai_compatible"
    )
    document_analysis_model: str = "qwen2.5vl:7b"
    document_analysis_schema_version: str = "1.0"
    document_analysis_max_bytes: int = Field(default=10 * 1024 * 1024, ge=1024)
    document_analysis_max_pdf_pages: int = Field(default=3, ge=1, le=10)
    document_analysis_render_dpi: int = Field(default=144, ge=72, le=200)

    def feature_configured(self, feature: str) -> bool:
        """True when the feature's provider block is complete (base URL, key and model)."""
        if feature not in FEATURE_PROVIDERS:
            raise ValueError(f"unknown feature provider: {feature}")
        return (
            getattr(self, f"{feature}_provider_base_url") is not None
            and bool(getattr(self, f"{feature}_provider_api_key"))
            and bool(getattr(self, f"{feature}_model"))
        )

    def for_feature(self, feature: str) -> "Settings":
        """Provider settings for one feature; unchanged unless its block is fully configured."""
        if not self.feature_configured(feature):
            return self
        base_url = getattr(self, f"{feature}_provider_base_url")
        api_key = getattr(self, f"{feature}_provider_api_key")
        model = getattr(self, f"{feature}_model")
        return self.model_copy(
            update={
                "ai_provider_protocol": getattr(self, f"{feature}_provider_protocol"),
                "ai_provider_base_url": base_url,
                "ai_provider_api_key": api_key,
                "ai_model": model,
                "ai_max_tokens": getattr(self, f"{feature}_max_tokens"),
                "ai_reasoning_effort": getattr(self, f"{feature}_reasoning_effort"),
                "ai_provider_message_max_chars": None,
                "request_timeout_seconds": getattr(self, f"{feature}_timeout_seconds"),
            }
        )

    def for_discovery(self) -> "Settings":
        """Provider settings for Discovery reflection and the Arena conclusion."""
        return self.for_feature("discovery")

    @field_validator("learning_service_url")
    @classmethod
    def learning_owner_origin(cls, value: str) -> str:
        url = urlsplit(value)
        host = url.hostname
        loopback = host in {"localhost", "127.0.0.1", "::1"}
        if (
            not host
            or url.username
            or url.password
            or url.path not in {"", "/"}
            or url.query
            or url.fragment
            or url.scheme not in {"http", "https"}
            or url.scheme == "http"
            and not loopback
        ):
            raise ValueError("Learning owner must be a secure service origin")
        # Accessing port rejects malformed/out-of-range values before token forwarding.
        _ = url.port
        return value.rstrip("/")


@lru_cache
def get_settings() -> Settings:
    return Settings()

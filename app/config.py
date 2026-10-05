from functools import lru_cache
from typing import Literal, Optional
from urllib.parse import urlsplit

from pydantic import Field, HttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    ai_provider_protocol: Literal["auto", "openai_compatible", "ollama_native"] = "auto"
    ai_provider_message_max_chars: Optional[int] = Field(default=None, ge=1, le=32768)
    ai_max_tokens: int = Field(default=256, ge=1, le=2048)

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

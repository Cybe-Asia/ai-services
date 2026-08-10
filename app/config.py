from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, HttpUrl
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
    ai_max_tokens: int = Field(default=256, ge=1, le=2048)

    sis_service_url: str = "http://sis-service"
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


@lru_cache
def get_settings() -> Settings:
    return Settings()

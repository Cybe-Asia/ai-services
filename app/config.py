from functools import lru_cache
from typing import Optional

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    service_name: str = "ai-service"
    app_env: str = "local"
    server_port: int = 8082

    # Production runs qwen3:14b via the shared on-prem CYBE AI Gateway
    # (AI_PROVIDER_BASE_URL). Override AI_MODEL locally if your provider
    # serves a smaller model.
    ai_model: str = "qwen3:14b"
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


@lru_cache
def get_settings() -> Settings:
    return Settings()

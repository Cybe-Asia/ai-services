from functools import lru_cache
from typing import Optional

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    service_name: str = "ai-service"
    app_env: str = "local"
    server_port: int = 8082

    ai_model: str = "llama3.1:8b"
    ai_provider_base_url: Optional[HttpUrl] = None
    ai_provider_api_key: Optional[str] = None
    ai_max_tokens: int = Field(default=256, ge=1, le=2048)

    sis_service_url: str = "http://sis-service"
    admission_service_url: str = "http://admission-service"
    request_timeout_seconds: float = Field(default=30.0, ge=1.0, le=120.0)


@lru_cache
def get_settings() -> Settings:
    return Settings()

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


def to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.capitalize() for part in tail)


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class ActorRole(str, Enum):
    public = "public"
    owner = "owner"
    admin = "admin"
    teacher = "teacher"
    parent = "parent"
    student = "student"


class ChatRequest(CamelModel):
    message: str = Field(min_length=1, max_length=4000)
    actor_role: ActorRole = ActorRole.public
    locale: str = Field(default="id", max_length=8)
    conversation_id: Optional[str] = Field(default=None, max_length=128)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceRef(CamelModel):
    kind: str
    title: str
    reference: Optional[str] = None


class ToolCallRef(CamelModel):
    name: str
    status: str


class ChatResponse(CamelModel):
    conversation_id: Optional[str] = None
    answer: str
    status: str = "ok"
    sources: list[SourceRef] = Field(default_factory=list)
    tool_calls: list[ToolCallRef] = Field(default_factory=list)


class HealthResponse(CamelModel):
    status: str
    service: str
    environment: str


class MetadataResponse(CamelModel):
    service: str
    model: str
    provider_configured: bool
    sis_service_url: str
    admission_service_url: str

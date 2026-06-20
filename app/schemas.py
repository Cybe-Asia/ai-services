from enum import Enum
from typing import Any, Literal, Optional

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


class SourceRef(CamelModel):
    kind: str
    title: str
    reference: Optional[str] = None


class ToolCallRef(CamelModel):
    name: str
    status: str


class ChatTurn(CamelModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)
    status: Optional[str] = Field(default=None, max_length=32)
    tool_calls: list[ToolCallRef] = Field(default_factory=list, max_length=8)


class ChatRequest(CamelModel):
    message: str = Field(min_length=1, max_length=4000)
    actor_role: ActorRole = ActorRole.public
    locale: str = Field(default="id", max_length=8)
    conversation_id: Optional[str] = Field(default=None, max_length=128)
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(CamelModel):
    conversation_id: Optional[str] = None
    answer: str
    status: str = "ok"
    sources: list[SourceRef] = Field(default_factory=list)
    tool_calls: list[ToolCallRef] = Field(default_factory=list)


class ThreadMessage(CamelModel):
    id: str
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)
    status: Optional[str] = Field(default=None, max_length=32)
    sources: list[SourceRef] = Field(default_factory=list)
    tool_calls: list[ToolCallRef] = Field(default_factory=list, max_length=8)
    created_at: str


class ThreadSummary(CamelModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = Field(ge=0)


class ThreadListResponse(CamelModel):
    threads: list[ThreadSummary] = Field(default_factory=list)


class CreateThreadRequest(CamelModel):
    title: Optional[str] = Field(default=None, max_length=120)


class ThreadResponse(CamelModel):
    thread: ThreadSummary
    messages: list[ThreadMessage] = Field(default_factory=list)


class AuditEvent(CamelModel):
    id: str
    created_at: str
    actor_role: str
    tool: str
    status: str
    date_range_label: Optional[str] = None
    source_kinds: list[str] = Field(default_factory=list, max_length=8)


class ThreadAuditResponse(CamelModel):
    events: list[AuditEvent] = Field(default_factory=list)


class DeleteThreadResponse(CamelModel):
    status: str = "ok"


class HealthResponse(CamelModel):
    status: str
    service: str
    environment: str


class MetadataResponse(CamelModel):
    service: str
    model: str
    max_tokens: int
    provider_configured: bool
    sis_service_url: str
    admission_service_url: str
    payment_service_url: str

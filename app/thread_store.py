import json
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

import redis.asyncio as redis

from app.config import Settings
from app.schemas import SourceRef, ThreadMessage, ThreadSummary, ToolCallRef

_MEMORY_THREADS: dict[str, dict[str, Any]] = {}
_MEMORY_OWNER_INDEX: dict[str, set[str]] = {}
_REDIS_CLIENTS: dict[str, redis.Redis] = {}


class ThreadStoreError(Exception):
    pass


class ThreadNotFound(ThreadStoreError):
    pass


class ThreadStore:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._ttl_seconds = settings.ai_thread_retention_days * 24 * 60 * 60

    async def list_threads(self, owner_id: str) -> list[ThreadSummary]:
        records = await self._list_records(owner_id)
        return [_summary_from_record(record) for record in records]

    async def create_thread(self, owner_id: str, title: Optional[str] = None) -> dict[str, Any]:
        now = _now_iso()
        thread_id = str(uuid4())
        record = {
            "id": thread_id,
            "ownerId": owner_id,
            "title": _clean_title(title) or "New chat",
            "createdAt": now,
            "updatedAt": now,
            "messages": [],
        }
        await self._save_record(record)
        await self._touch_owner_index(owner_id, thread_id, now)
        await self._prune_owner_threads(owner_id)
        return record

    async def ensure_thread(
        self,
        owner_id: str,
        thread_id: Optional[str],
        first_message: str,
    ) -> dict[str, Any]:
        if thread_id:
            try:
                return await self.get_record(owner_id, thread_id)
            except ThreadNotFound:
                pass
        return await self.create_thread(owner_id, _title_from_message(first_message))

    async def get_record(self, owner_id: str, thread_id: str) -> dict[str, Any]:
        record = await self._get_record(thread_id)
        if record is None or record.get("ownerId") != owner_id:
            raise ThreadNotFound("thread not found")
        return record

    async def get_thread(
        self,
        owner_id: str,
        thread_id: str,
    ) -> tuple[ThreadSummary, list[ThreadMessage]]:
        record = await self.get_record(owner_id, thread_id)
        messages = [_message_from_record(row) for row in record["messages"]]
        return _summary_from_record(record), messages

    async def append_exchange(
        self,
        owner_id: str,
        thread_id: str,
        user_content: str,
        assistant_content: str,
        assistant_status: str,
        sources: list[SourceRef],
        tool_calls: list[ToolCallRef],
    ) -> dict[str, Any]:
        record = await self.get_record(owner_id, thread_id)
        now = _now_iso()
        messages = list(record.get("messages") or [])
        messages.extend(
            [
                _message_record("user", user_content, now),
                _message_record(
                    "assistant",
                    assistant_content,
                    now,
                    status=assistant_status,
                    sources=[source.model_dump(by_alias=True) for source in sources],
                    tool_calls=[tool.model_dump(by_alias=True) for tool in tool_calls],
                ),
            ]
        )
        record["messages"] = messages[-self._settings.ai_thread_max_messages :]
        if record.get("title") == "New chat":
            record["title"] = _title_from_message(user_content)
        record["updatedAt"] = now
        await self._save_record(record)
        await self._touch_owner_index(owner_id, thread_id, now)
        await self._prune_owner_threads(owner_id)
        return record

    async def delete_thread(self, owner_id: str, thread_id: str) -> None:
        record = await self._get_record(thread_id)
        if record is None or record.get("ownerId") != owner_id:
            raise ThreadNotFound("thread not found")
        if self._settings.redis_url:
            client = _redis_client(self._settings.redis_url)
            await client.delete(_thread_key(thread_id))
            await client.zrem(_owner_index_key(owner_id), thread_id)
            return

        _MEMORY_THREADS.pop(thread_id, None)
        _MEMORY_OWNER_INDEX.get(owner_id, set()).discard(thread_id)

    async def _list_records(self, owner_id: str) -> list[dict[str, Any]]:
        if self._settings.redis_url:
            client = _redis_client(self._settings.redis_url)
            thread_ids = await client.zrevrange(
                _owner_index_key(owner_id),
                0,
                self._settings.ai_thread_max_threads - 1,
            )
            records = []
            for thread_id in thread_ids:
                record = await self._get_record(str(thread_id))
                if record is not None and record.get("ownerId") == owner_id:
                    records.append(record)
            return records

        thread_ids = _MEMORY_OWNER_INDEX.get(owner_id, set())
        records = [
            record
            for thread_id in thread_ids
            if (record := _MEMORY_THREADS.get(thread_id)) is not None
        ]
        return sorted(records, key=lambda row: row.get("updatedAt", ""), reverse=True)[
            : self._settings.ai_thread_max_threads
        ]

    async def _get_record(self, thread_id: str) -> Optional[dict[str, Any]]:
        if self._settings.redis_url:
            client = _redis_client(self._settings.redis_url)
            raw = await client.get(_thread_key(thread_id))
            if not raw:
                return None
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ThreadStoreError("stored thread is invalid") from exc
            return record if isinstance(record, dict) else None

        return _MEMORY_THREADS.get(thread_id)

    async def _save_record(self, record: dict[str, Any]) -> None:
        if self._settings.redis_url:
            client = _redis_client(self._settings.redis_url)
            await client.setex(
                _thread_key(record["id"]),
                self._ttl_seconds,
                json.dumps(record, separators=(",", ":")),
            )
            return

        _MEMORY_THREADS[record["id"]] = record

    async def _touch_owner_index(self, owner_id: str, thread_id: str, updated_at: str) -> None:
        if self._settings.redis_url:
            client = _redis_client(self._settings.redis_url)
            index_key = _owner_index_key(owner_id)
            await client.zadd(index_key, {thread_id: _score(updated_at)})
            await client.expire(index_key, self._ttl_seconds)
            return

        _MEMORY_OWNER_INDEX.setdefault(owner_id, set()).add(thread_id)

    async def _prune_owner_threads(self, owner_id: str) -> None:
        max_threads = self._settings.ai_thread_max_threads
        if self._settings.redis_url:
            client = _redis_client(self._settings.redis_url)
            index_key = _owner_index_key(owner_id)
            total = await client.zcard(index_key)
            overage = int(total) - max_threads
            if overage <= 0:
                return
            old_ids = await client.zrange(index_key, 0, overage - 1)
            if old_ids:
                await client.delete(*[_thread_key(str(thread_id)) for thread_id in old_ids])
                await client.zrem(index_key, *old_ids)
            return

        records = await self._list_records(owner_id)
        for record in records[max_threads:]:
            _MEMORY_THREADS.pop(record["id"], None)
            _MEMORY_OWNER_INDEX.get(owner_id, set()).discard(record["id"])


def _redis_client(redis_url: str) -> redis.Redis:
    client = _REDIS_CLIENTS.get(redis_url)
    if client is None:
        client = redis.from_url(redis_url, decode_responses=True)
        _REDIS_CLIENTS[redis_url] = client
    return client


def _thread_key(thread_id: str) -> str:
    return f"ai:thread:{thread_id}"


def _owner_index_key(owner_id: str) -> str:
    return f"ai:threads:{owner_id}"


def _score(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _message_record(
    role: str,
    content: str,
    created_at: str,
    status: Optional[str] = None,
    sources: Optional[list[dict[str, Any]]] = None,
    tool_calls: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    return {
        "id": str(uuid4()),
        "role": role,
        "content": content,
        "status": status,
        "sources": sources or [],
        "toolCalls": tool_calls or [],
        "createdAt": created_at,
    }


def _summary_from_record(record: dict[str, Any]) -> ThreadSummary:
    messages = record.get("messages") if isinstance(record.get("messages"), list) else []
    return ThreadSummary(
        id=str(record.get("id") or ""),
        title=str(record.get("title") or "New chat"),
        created_at=str(record.get("createdAt") or ""),
        updated_at=str(record.get("updatedAt") or ""),
        message_count=len(messages),
    )


def _message_from_record(record: dict[str, Any]) -> ThreadMessage:
    return ThreadMessage(
        id=str(record.get("id") or ""),
        role=record.get("role") if record.get("role") in {"user", "assistant"} else "assistant",
        content=str(record.get("content") or ""),
        status=record.get("status") if isinstance(record.get("status"), str) else None,
        sources=[
            SourceRef.model_validate(source)
            for source in _list_of_dicts(record.get("sources"))
        ],
        tool_calls=[
            ToolCallRef.model_validate(tool)
            for tool in _list_of_dicts(record.get("toolCalls") or record.get("tool_calls"))
        ],
        created_at=str(record.get("createdAt") or ""),
    )


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _clean_title(value: Optional[str]) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())[:120]


def _title_from_message(message: str) -> str:
    title = _clean_title(message)
    if len(title) > 64:
        return f"{title[:61]}..."
    return title or "New chat"

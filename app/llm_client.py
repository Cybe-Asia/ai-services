import json
from collections.abc import AsyncIterator
from typing import Optional

import httpx

from app.config import Settings


class LlmClient:
    def __init__(self, settings: Settings):
        self._settings = settings

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
        response_format: Optional[dict[str, str]] = None,
    ) -> Optional[str]:
        if self._settings.ai_provider_base_url is None:
            return None

        base_url = str(self._settings.ai_provider_base_url).rstrip("/")
        headers = {"content-type": "application/json"}
        if self._settings.ai_provider_api_key:
            headers["authorization"] = f"Bearer {self._settings.ai_provider_api_key}"

        payload = {
            "model": self._settings.ai_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens or self._settings.ai_max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        timeout = timeout_seconds or self._settings.request_timeout_seconds
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            body = response.json()

        choices = body.get("choices") or []
        if not choices:
            return None
        message = choices[0].get("message") or {}
        content = message.get("content")
        return content if isinstance(content, str) and content.strip() else None

    async def stream_complete(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
        response_format: Optional[dict[str, str]] = None,
    ) -> AsyncIterator[str]:
        if self._settings.ai_provider_base_url is None:
            return

        base_url = str(self._settings.ai_provider_base_url).rstrip("/")
        headers = {"content-type": "application/json"}
        if self._settings.ai_provider_api_key:
            headers["authorization"] = f"Bearer {self._settings.ai_provider_api_key}"

        payload = {
            "model": self._settings.ai_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens or self._settings.ai_max_tokens,
            "stream": True,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        timeout = timeout_seconds or self._settings.request_timeout_seconds
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                f"{base_url}/chat/completions",
                json=payload,
                headers=headers,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line.removeprefix("data:").strip()
                    if not raw or raw == "[DONE]":
                        if raw == "[DONE]":
                            break
                        continue
                    try:
                        body = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    choices = body.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        yield content

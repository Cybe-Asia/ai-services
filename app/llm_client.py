import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
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
        headers = _provider_headers(self._settings)
        native_base_url = _ollama_native_base_url(self._settings.ai_model, base_url)
        if native_base_url is not None:
            return await self._complete_ollama_native(
                native_base_url,
                headers,
                system_prompt,
                user_message,
                temperature,
                max_tokens,
                timeout_seconds,
                response_format,
            )

        payload = {
            "model": self._settings.ai_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": _apply_model_prompt_controls(
                        self._settings.ai_model,
                        user_message,
                    ),
                },
            ],
            "temperature": temperature,
            "max_tokens": max_tokens or self._settings.ai_max_tokens,
        }
        if _is_qwen3_model(self._settings.ai_model):
            payload["think"] = False
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
        if not isinstance(content, str):
            return None

        cleaned = _strip_thinking_content(content).strip()
        return cleaned if cleaned else None

    async def complete_multimodal(
        self,
        *,
        model: str,
        system_prompt: str,
        user_content: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 768,
        timeout_seconds: Optional[float] = None,
        response_format: Optional[dict[str, str]] = None,
    ) -> Optional[str]:
        """Complete a provider-neutral OpenAI-compatible multimodal request."""
        if self._settings.ai_provider_base_url is None:
            return None
        base_url = str(self._settings.ai_provider_base_url).rstrip("/")
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        timeout = timeout_seconds or self._settings.request_timeout_seconds
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                f"{base_url}/chat/completions",
                json=payload,
                headers=_provider_headers(self._settings),
            ) as response:
                response.raise_for_status()
                if int(response.headers.get("content-length", "0")) > 64 * 1024:
                    raise ValueError("provider response is too large")
                raw = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(raw) + len(chunk) > 64 * 1024:
                        raise ValueError("provider response is too large")
                    raw.extend(chunk)
                body = json.loads(raw)
        choices = body.get("choices") or []
        content = (choices[0].get("message") or {}).get("content") if choices else None
        return content.strip() if isinstance(content, str) and content.strip() else None

    async def _complete_ollama_native(
        self,
        base_url: str,
        headers: dict[str, str],
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: Optional[int],
        timeout_seconds: Optional[float],
        response_format: Optional[dict[str, str]],
    ) -> Optional[str]:
        payload = _ollama_chat_payload(
            self._settings.ai_model,
            system_prompt,
            user_message,
            temperature,
            max_tokens or self._settings.ai_max_tokens,
            stream=False,
            response_format=response_format,
        )

        timeout = timeout_seconds or self._settings.request_timeout_seconds
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{base_url}/api/chat",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            body = response.json()

        message = body.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            return None

        cleaned = _strip_thinking_content(content).strip()
        return cleaned if cleaned else None

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
        headers = _provider_headers(self._settings)
        native_base_url = _ollama_native_base_url(self._settings.ai_model, base_url)
        if native_base_url is not None:
            async for chunk in self._stream_ollama_native(
                native_base_url,
                headers,
                system_prompt,
                user_message,
                temperature,
                max_tokens,
                timeout_seconds,
                response_format,
            ):
                yield chunk
            return

        payload = {
            "model": self._settings.ai_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": _apply_model_prompt_controls(
                        self._settings.ai_model,
                        user_message,
                    ),
                },
            ],
            "temperature": temperature,
            "max_tokens": max_tokens or self._settings.ai_max_tokens,
            "stream": True,
        }
        if _is_qwen3_model(self._settings.ai_model):
            payload["think"] = False
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
                thinking_filter = _ThinkingContentFilter()
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
                        cleaned = thinking_filter.feed(content)
                        if cleaned:
                            yield cleaned

                cleaned = thinking_filter.flush()
                if cleaned:
                    yield cleaned

    async def _stream_ollama_native(
        self,
        base_url: str,
        headers: dict[str, str],
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: Optional[int],
        timeout_seconds: Optional[float],
        response_format: Optional[dict[str, str]],
    ) -> AsyncIterator[str]:
        payload = _ollama_chat_payload(
            self._settings.ai_model,
            system_prompt,
            user_message,
            temperature,
            max_tokens or self._settings.ai_max_tokens,
            stream=True,
            response_format=response_format,
        )

        timeout = timeout_seconds or self._settings.request_timeout_seconds
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                f"{base_url}/api/chat",
                json=payload,
                headers=headers,
            ) as response:
                response.raise_for_status()
                thinking_filter = _ThinkingContentFilter()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        body = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    message = body.get("message") or {}
                    content = message.get("content")
                    if isinstance(content, str) and content:
                        cleaned = thinking_filter.feed(content)
                        if cleaned:
                            yield cleaned
                    if body.get("done") is True:
                        break

                cleaned = thinking_filter.flush()
                if cleaned:
                    yield cleaned


def _provider_headers(settings: Settings) -> dict[str, str]:
    headers = {"content-type": "application/json"}
    if settings.ai_provider_api_key:
        headers["authorization"] = f"Bearer {settings.ai_provider_api_key}"
    return headers


def _ollama_chat_payload(
    model: str,
    system_prompt: str,
    user_message: str,
    temperature: float,
    max_tokens: int,
    stream: bool,
    response_format: Optional[dict[str, str]],
) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _apply_model_prompt_controls(model, user_message)},
        ],
        "stream": stream,
        "think": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    if response_format and response_format.get("type") == "json_object":
        payload["format"] = "json"
    return payload


def _ollama_native_base_url(model: str, base_url: str) -> Optional[str]:
    if not _is_qwen3_model(model):
        return None
    if "ollama" not in base_url.casefold() and ":11434" not in base_url:
        return None
    return base_url[:-3] if base_url.endswith("/v1") else base_url


def _apply_model_prompt_controls(model: str, user_message: str) -> str:
    if not _is_qwen3_model(model):
        return user_message
    if "/no_think" in user_message or "/think" in user_message:
        return user_message
    return f"{user_message.rstrip()} /no_think"


def _is_qwen3_model(model: str) -> bool:
    return "qwen3" in model.casefold()


def _strip_thinking_content(text: str) -> str:
    thinking_filter = _ThinkingContentFilter()
    return f"{thinking_filter.feed(text)}{thinking_filter.flush()}"


def _longest_suffix_prefix(text: str, prefix: str) -> int:
    max_length = min(len(text), len(prefix) - 1)
    for length in range(max_length, 0, -1):
        if text[-length:] == prefix[:length]:
            return length
    return 0


@dataclass
class _ThinkingContentFilter:
    buffer: str = ""
    in_thinking: bool = False

    def feed(self, chunk: str) -> str:
        self.buffer += chunk
        output: list[str] = []

        while self.buffer:
            lowered = self.buffer.casefold()
            if self.in_thinking:
                close_index = lowered.find("</think>")
                if close_index < 0:
                    keep = _longest_suffix_prefix(lowered, "</think>")
                    self.buffer = self.buffer[-keep:] if keep else ""
                    return "".join(output)
                self.buffer = self.buffer[close_index + len("</think>") :]
                self.in_thinking = False
                continue

            open_index = lowered.find("<think")
            if open_index < 0:
                keep = _longest_suffix_prefix(lowered, "<think")
                if keep:
                    output.append(self.buffer[:-keep])
                    self.buffer = self.buffer[-keep:]
                else:
                    output.append(self.buffer)
                    self.buffer = ""
                return "".join(output)

            output.append(self.buffer[:open_index])
            tag_end = self.buffer.find(">", open_index)
            if tag_end < 0:
                self.buffer = self.buffer[open_index:]
                return "".join(output)

            self.buffer = self.buffer[tag_end + 1 :]
            self.in_thinking = True

        return "".join(output)

    def flush(self) -> str:
        if self.in_thinking:
            self.buffer = ""
            return ""
        output = self.buffer
        self.buffer = ""
        return output

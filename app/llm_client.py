import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Optional

import httpx

from app.config import Settings


async def _bounded_provider_json(
    client: httpx.AsyncClient, url: str, *, payload: dict, headers: dict[str, str]
) -> dict:
    async with client.stream(
        "POST", url, json=payload, headers={**headers, "Accept-Encoding": "identity"}
    ) as response:
        response.raise_for_status()
        if response.headers.get("content-encoding", "identity").strip().casefold() != "identity":
            raise ValueError("compressed provider responses are not supported")
        if int(response.headers.get("content-length", "0")) > 65536:
            raise ValueError("provider response is too large")
        raw = bytearray()
        async for chunk in response.aiter_bytes(chunk_size=8192):
            if len(raw) + len(chunk) > 65536:
                raise ValueError("provider response is too large")
            raw.extend(chunk)
        body = json.loads(raw)
        if not isinstance(body, dict):
            raise ValueError("invalid provider response")
        return body


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
        require_complete: bool = False,
    ) -> Optional[str]:
        if self._settings.ai_provider_base_url is None:
            return None

        _validate_provider_messages(self._settings, system_prompt, user_message)
        base_url = str(self._settings.ai_provider_base_url).rstrip("/")
        if self._settings.ai_provider_protocol == "anthropic":
            return await self._complete_anthropic(
                base_url,
                system_prompt,
                user_message,
                temperature,
                max_tokens,
                timeout_seconds,
            )
        headers = _provider_headers(self._settings)
        if self._settings.ai_provider_protocol == "openai":
            return await self._complete_openai(
                base_url,
                headers,
                system_prompt,
                user_message,
                max_tokens,
                timeout_seconds,
                response_format,
            )
        native_base_url = _ollama_native_base_url(
            self._settings.ai_model, base_url, self._settings.ai_provider_protocol
        )
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
                require_complete,
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
            body = await asyncio.wait_for(
                _bounded_provider_json(
                    client,
                    f"{base_url}/chat/completions",
                    payload=payload,
                    headers=headers,
                ),
                timeout=timeout,
            )

        choices = body.get("choices") or []
        if not choices:
            return None
        if require_complete and choices[0].get("finish_reason") != "stop":
            return None
        if choices[0].get("finish_reason") not in (None, "stop"):
            return None
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            return None

        cleaned = _strip_thinking_content(content).strip()
        return cleaned if cleaned else None

    async def _complete_openai(
        self,
        base_url: str,
        headers: dict[str, str],
        system_prompt: str,
        user_message: str,
        max_tokens: Optional[int],
        timeout_seconds: Optional[float],
        response_format: Optional[dict[str, str]],
    ) -> Optional[str]:
        """OpenAI Chat Completions for its reasoning models: the output limit is
        `max_completion_tokens` and sampling controls are not sent. Only a finished
        `stop` answer is released."""
        payload: dict = {
            "model": self._settings.ai_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "max_completion_tokens": max_tokens or self._settings.ai_max_tokens,
        }
        if self._settings.ai_reasoning_effort is not None:
            payload["reasoning_effort"] = self._settings.ai_reasoning_effort
        if response_format is not None:
            payload["response_format"] = response_format
        timeout = timeout_seconds or self._settings.request_timeout_seconds
        async with httpx.AsyncClient(timeout=timeout) as client:
            body = await asyncio.wait_for(
                _bounded_provider_json(
                    client, f"{base_url}/chat/completions", payload=payload, headers=headers
                ),
                timeout=timeout,
            )
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return None
        if choices[0].get("finish_reason") != "stop":
            return None
        content = (choices[0].get("message") or {}).get("content")
        if not isinstance(content, str):
            return None
        cleaned = _without_code_fence(content)
        return cleaned if cleaned else None

    async def _complete_anthropic(
        self,
        base_url: str,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: Optional[int],
        timeout_seconds: Optional[float],
    ) -> Optional[str]:
        """Anthropic Messages API; only a finished `end_turn` answer is released."""
        payload = {
            "model": self._settings.ai_model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}],
            "temperature": temperature,
            "max_tokens": max_tokens or self._settings.ai_max_tokens,
        }
        headers = {
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": self._settings.ai_provider_api_key or "",
        }
        timeout = timeout_seconds or self._settings.request_timeout_seconds
        async with httpx.AsyncClient(timeout=timeout) as client:
            body = await asyncio.wait_for(
                _bounded_provider_json(
                    client, f"{base_url}/v1/messages", payload=payload, headers=headers
                ),
                timeout=timeout,
            )
        if body.get("stop_reason") != "end_turn":
            return None
        blocks = body.get("content")
        if not isinstance(blocks, list):
            return None
        texts = [b.get("text") for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
        if not texts or not all(isinstance(t, str) for t in texts):
            return None
        cleaned = _without_code_fence("".join(texts))
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

    async def complete_gateway_vision(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        image_base64: str,
        json_schema: dict,
        temperature: float = 0.0,
        max_tokens: int = 768,
        timeout_seconds: Optional[float] = None,
    ) -> Optional[str]:
        """Call CYBE Gateway's bounded vision adapter without leaking provider details."""
        if self._settings.ai_provider_base_url is None:
            return None
        base_url = str(self._settings.ai_provider_base_url).rstrip("/")
        timeout = timeout_seconds or self._settings.request_timeout_seconds
        payload = {
            "model": model,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "image_base64": image_base64,
            "json_schema": json_schema,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                f"{base_url}/vision/extract",
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
        content = body.get("content")
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
        require_complete: bool = False,
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
            body = await asyncio.wait_for(
                _bounded_provider_json(
                    client,
                    f"{base_url}/api/chat",
                    payload=payload,
                    headers=headers,
                ),
                timeout=timeout,
            )

        if require_complete and (body.get("done") is not True or body.get("done_reason") != "stop"):
            return None
        if body.get("done_reason") not in (None, "stop"):
            return None
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

        _validate_provider_messages(self._settings, system_prompt, user_message)
        base_url = str(self._settings.ai_provider_base_url).rstrip("/")
        headers = _provider_headers(self._settings)
        native_base_url = _ollama_native_base_url(
            self._settings.ai_model, base_url, self._settings.ai_provider_protocol
        )
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


def _without_code_fence(text: str) -> str:
    """A JSON answer wrapped in one Markdown code fence, as Claude sometimes returns it."""
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```") and stripped.count("```") == 2:
        body = stripped[3:-3]
        return body.split("\n", 1)[1] if body.startswith("json\n") else body
    return stripped


def _provider_headers(settings: Settings) -> dict[str, str]:
    headers = {"content-type": "application/json"}
    if settings.ai_provider_api_key:
        headers["authorization"] = f"Bearer {settings.ai_provider_api_key}"
    return headers


def _validate_provider_messages(settings: Settings, system_prompt: str, user_message: str) -> None:
    limit = settings.ai_provider_message_max_chars
    if limit is not None and (
        len(system_prompt) > limit
        or len(_apply_model_prompt_controls(settings.ai_model, user_message)) > limit
    ):
        raise ValueError("provider message exceeds configured limit")


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


def _ollama_native_base_url(
    model: str, base_url: str, protocol: str = "auto"
) -> Optional[str]:
    if protocol == "openai_compatible":
        return None
    if protocol == "ollama_native":
        return base_url[:-3] if base_url.endswith("/v1") else base_url
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

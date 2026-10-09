import asyncio

import httpx
import pytest

from app import llm_client
from app.config import Settings
from app.llm_client import LlmClient


@pytest.mark.parametrize(
    "model,path,body",
    [
        (
            "gemma2:2b",
            "/v1/chat/completions",
            {"choices": [{"message": {"content": "Explanation"}}]},
        ),
        ("qwen3:4b", "/api/chat", {"message": {"content": "Explanation"}}),
    ],
)
def test_bounded_complete_preserves_openai_and_ollama_contracts(monkeypatch, model, path, body):
    original = httpx.AsyncClient

    def response(request):
        assert request.url.path == path
        return httpx.Response(200, json=body)

    monkeypatch.setattr(
        llm_client.httpx,
        "AsyncClient",
        lambda **kwargs: original(**kwargs, transport=httpx.MockTransport(response)),
    )
    settings = Settings(
        _env_file=None, ai_model=model, ai_provider_base_url="http://localhost:11434/v1"
    )
    assert asyncio.run(LlmClient(settings).complete("Reading", "Question")) == "Explanation"


@pytest.mark.parametrize("model", ["gemma2:2b", "qwen3:4b"])
def test_completion_stops_and_closes_upstream_before_buffering_oversized_json(monkeypatch, model):
    original = httpx.AsyncClient

    class LargeStream(httpx.AsyncByteStream):
        def __init__(self):
            self.chunks = 0
            self.closed = False

        async def __aiter__(self):
            for _ in range(100):
                self.chunks += 1
                yield b"x" * 8192

        async def aclose(self):
            self.closed = True

    stream = LargeStream()
    transport = httpx.MockTransport(lambda request: httpx.Response(200, stream=stream))
    monkeypatch.setattr(
        llm_client.httpx, "AsyncClient", lambda **kwargs: original(**kwargs, transport=transport)
    )
    settings = Settings(
        _env_file=None, ai_model=model, ai_provider_base_url="http://localhost:11434/v1"
    )
    with pytest.raises(ValueError, match="too large"):
        asyncio.run(LlmClient(settings).complete("Reading", "Question"))
    assert stream.chunks == 9
    assert stream.closed


@pytest.mark.parametrize(
    "model,body",
    [
        (
            "gemma2:2b",
            {"choices": [{"finish_reason": "length", "message": {"content": "Cut off"}}]},
        ),
        (
            "gemma2:2b",
            {"choices": [{"finish_reason": "content_filter", "message": {"content": "Cut off"}}]},
        ),
        ("qwen3:4b", {"done_reason": "length", "message": {"content": "Cut off"}}),
    ],
)
def test_incomplete_provider_generation_is_not_an_answer(monkeypatch, model, body):
    original = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=body))
    monkeypatch.setattr(
        llm_client.httpx, "AsyncClient", lambda **kwargs: original(**kwargs, transport=transport)
    )
    settings = Settings(
        _env_file=None, ai_model=model, ai_provider_base_url="http://localhost:11434/v1"
    )
    assert asyncio.run(LlmClient(settings).complete("Reading", "Question")) is None


@pytest.mark.parametrize("encoding", ["gzip", "br", "identity, gzip"])
def test_compressed_provider_body_is_rejected_before_decoding(monkeypatch, encoding):
    original = httpx.AsyncClient

    class EncodedStream(httpx.AsyncByteStream):
        read = False
        closed = False

        async def __aiter__(self):
            self.read = True
            yield b"must never decode this body"

        async def aclose(self):
            self.closed = True

    stream = EncodedStream()

    def response(request):
        assert request.headers["accept-encoding"] == "identity"
        return httpx.Response(200, headers={"Content-Encoding": encoding}, stream=stream)

    monkeypatch.setattr(
        llm_client.httpx,
        "AsyncClient",
        lambda **kwargs: original(**kwargs, transport=httpx.MockTransport(response)),
    )
    settings = Settings(_env_file=None, ai_provider_base_url="http://localhost:3254/v1")
    with pytest.raises(ValueError, match="compressed"):
        asyncio.run(LlmClient(settings).complete("Reading", "Question"))
    assert not stream.read
    assert stream.closed


@pytest.mark.parametrize(
    "protocol,model,path,native",
    [
        ("openai_compatible", "qwen3:4b", "/v1/chat/completions", False),
        ("openai_compatible", "gemma2:2b", "/v1/chat/completions", False),
        ("ollama_native", "gemma2:2b", "/api/chat", True),
        ("auto", "qwen3:4b", "/api/chat", True),
    ],
)
def test_explicit_protocol_routes_independently_of_model_family(
    monkeypatch, protocol, model, path, native
):
    import json

    original = httpx.AsyncClient

    def response(request):
        assert request.url.path == path
        assert request.headers["authorization"] == "Bearer test-provider-key"
        assert request.headers["accept-encoding"] == "identity"
        payload = json.loads(request.content)
        assert payload["model"] == model
        if native:
            assert payload["format"] == "json"
            assert payload["options"]["num_predict"] == 512
            return httpx.Response(
                200, json={"done": True, "done_reason": "stop", "message": {"content": "Answer"}}
            )
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["max_tokens"] == 512
        return httpx.Response(
            200, json={"choices": [{"finish_reason": "stop", "message": {"content": "Answer"}}]}
        )

    monkeypatch.setattr(
        llm_client.httpx,
        "AsyncClient",
        lambda **kwargs: original(**kwargs, transport=httpx.MockTransport(response)),
    )
    settings = Settings(
        _env_file=None,
        ai_model=model,
        ai_provider_base_url="http://localhost:11434/v1",
        ai_provider_protocol=protocol,
        ai_provider_api_key="test-provider-key",
    )
    assert asyncio.run(
        LlmClient(settings).complete(
            "Published reading",
            "Question",
            max_tokens=512,
            response_format={"type": "json_object"},
            require_complete=True,
        )
    ) == "Answer"


@pytest.mark.parametrize("protocol", ["openai_compatible", "ollama_native"])
def test_absolute_completion_deadline_closes_trickling_body(monkeypatch, protocol):
    original = httpx.AsyncClient

    class Trickle(httpx.AsyncByteStream):
        chunks = 0
        closed = False

        async def __aiter__(self):
            for _ in range(100):
                await asyncio.sleep(0.005)
                self.chunks += 1
                yield b" "

        async def aclose(self):
            self.closed = True

    stream = Trickle()
    monkeypatch.setattr(
        llm_client.httpx,
        "AsyncClient",
        lambda **kwargs: original(
            **kwargs, transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=stream))
        ),
    )
    settings = Settings(
        _env_file=None,
        ai_provider_base_url="http://localhost:3254/v1",
        ai_provider_protocol=protocol,
    )
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(LlmClient(settings).complete("Reading", "Question", timeout_seconds=0.02))
    assert stream.chunks < 100
    assert stream.closed


@pytest.mark.parametrize("model,text", [("gemma2:2b", "x" * 21), ("qwen3:4b", "x" * 15)])
def test_gateway_message_limit_rejects_whole_source_before_network(monkeypatch, model, text):
    def unexpected_client(**kwargs):
        raise AssertionError("oversized source must never be sent")

    monkeypatch.setattr(llm_client.httpx, "AsyncClient", unexpected_client)
    settings = Settings(
        _env_file=None,
        ai_model=model,
        ai_provider_base_url="http://localhost:3254/v1",
        ai_provider_protocol="openai_compatible",
        ai_provider_message_max_chars=20,
    )
    with pytest.raises(ValueError, match="configured limit"):
        asyncio.run(LlmClient(settings).complete("Reading", text))

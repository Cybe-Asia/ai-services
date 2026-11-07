from app.llm_client import (
    _apply_model_prompt_controls,
    _ollama_chat_payload,
    _ollama_native_base_url,
    _strip_thinking_content,
    _ThinkingContentFilter,
)


def test_qwen3_prompt_uses_no_think_control() -> None:
    assert _apply_model_prompt_controls("qwen3:14b", "berapa EOI?") == "berapa EOI? /no_think"
    assert _apply_model_prompt_controls("gemma2:2b", "berapa EOI?") == "berapa EOI?"
    assert _apply_model_prompt_controls("qwen3:14b", "/think\nanalisa") == "/think\nanalisa"


def test_qwen3_uses_ollama_native_base_url() -> None:
    assert _ollama_native_base_url("qwen3:14b", "http://ollama:11434/v1") == (
        "http://ollama:11434"
    )
    assert _ollama_native_base_url("gemma2:2b", "http://ollama:11434/v1") is None
    assert _ollama_native_base_url("qwen3:14b", "https://api.example.com/v1") is None


def test_ollama_payload_disables_thinking_and_uses_json_format() -> None:
    payload = _ollama_chat_payload(
        "qwen3:14b",
        "system",
        "berapa EOI?",
        temperature=0.0,
        max_tokens=32,
        stream=False,
        response_format={"type": "json_object"},
    )

    assert payload["think"] is False
    assert payload["format"] == "json"
    assert payload["options"] == {"temperature": 0.0, "num_predict": 32}
    assert payload["messages"][1]["content"] == "berapa EOI? /no_think"


def test_strip_thinking_content_from_complete_response() -> None:
    cleaned = _strip_thinking_content("<think>hidden reasoning</think>Ada 1 EOI.")
    assert cleaned == "Ada 1 EOI."


def test_streaming_filter_hides_split_thinking_chunks() -> None:
    thinking_filter = _ThinkingContentFilter()
    chunks = ["<thi", "nk>hidden", " reasoning</thi", "nk>Ada", " 1 EOI."]
    visible = "".join(thinking_filter.feed(chunk) for chunk in chunks)
    visible += thinking_filter.flush()

    assert visible == "Ada 1 EOI."


def test_assistant_data_tier_reaches_the_gateway_header_and_others_send_none():
    from app.config import Settings
    from app.llm_client import _provider_headers

    settings = Settings(
        assistant_provider_base_url="http://ai-gateway.ai-platform:8080/v1",
        assistant_provider_api_key="school-token",
        assistant_model="digital-school",
        assistant_data_tier="standard",
    )
    headers = _provider_headers(settings.for_feature("assistant"))
    assert headers["x-ai-data-tier"] == "standard"
    assert headers["authorization"] == "Bearer school-token"
    assert "x-ai-data-tier" not in _provider_headers(settings)


def test_every_feature_block_carries_its_own_data_tier():
    from app.config import FEATURE_PROVIDERS, Settings
    from app.llm_client import _provider_headers

    for feature in FEATURE_PROVIDERS:
        settings = Settings(
            **{
                f"{feature}_provider_base_url": "http://ai-gateway.ai-platform:8080/v1",
                f"{feature}_provider_api_key": "gateway-token",
                f"{feature}_model": "digital-school-chat",
                f"{feature}_data_tier": "standard",
            }
        )
        assert _provider_headers(settings.for_feature(feature))["x-ai-data-tier"] == "standard"
        for other in FEATURE_PROVIDERS:
            if other != feature:
                assert "x-ai-data-tier" not in _provider_headers(settings.for_feature(other))


def openai_stream(monkeypatch, events: str):
    import asyncio

    import httpx

    from app import llm_client
    from app.config import Settings

    real = httpx.AsyncClient
    sent = []

    def handler(request):
        import json

        sent.append(json.loads(request.content))
        return httpx.Response(200, text=events)

    monkeypatch.setattr(
        llm_client.httpx,
        "AsyncClient",
        lambda **kw: real(transport=httpx.MockTransport(handler), **kw),
    )
    client = llm_client.LlmClient(
        Settings(
            _env_file=None,
            ai_provider_base_url="http://gateway/v1",
            ai_provider_protocol="openai",
            ai_model="digital-school-chat",
        )
    )

    async def run():
        return [
            chunk
            async for chunk in client.stream_complete(
                system_prompt="s", user_message="u", require_complete=True
            )
        ]

    return asyncio.run(run()), sent


def test_openai_stream_yields_deltas_and_sends_reasoning_model_fields(monkeypatch) -> None:
    chunks, sent = openai_stream(
        monkeypatch,
        'data: {"choices":[{"delta":{"content":"Ha"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}\n\n'
        'data: {"choices":[],"usage":{"prompt_tokens":1}}\n\ndata: [DONE]\n\n',
    )

    assert chunks == ["Ha", "lo"]
    assert sent[0]["stream"] is True and "max_completion_tokens" in sent[0]
    assert "temperature" not in sent[0]


def test_openai_stream_that_is_cut_off_raises_when_completeness_is_required(monkeypatch) -> None:
    import pytest

    with pytest.raises(ValueError):
        openai_stream(
            monkeypatch,
            'data: {"choices":[{"delta":{"content":"Ha"},"finish_reason":"length"}]}\n\n',
        )

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

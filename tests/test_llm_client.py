from app.llm_client import (
    _apply_model_prompt_controls,
    _strip_thinking_content,
    _ThinkingContentFilter,
)


def test_qwen3_prompt_uses_no_think_control() -> None:
    assert _apply_model_prompt_controls("qwen3:14b", "berapa EOI?") == "berapa EOI? /no_think"
    assert _apply_model_prompt_controls("llama3.2:1b", "berapa EOI?") == "berapa EOI?"
    assert _apply_model_prompt_controls("qwen3:14b", "/think\nanalisa") == "/think\nanalisa"


def test_strip_thinking_content_from_complete_response() -> None:
    cleaned = _strip_thinking_content("<think>hidden reasoning</think>Ada 1 EOI.")
    assert cleaned == "Ada 1 EOI."


def test_streaming_filter_hides_split_thinking_chunks() -> None:
    thinking_filter = _ThinkingContentFilter()
    chunks = ["<thi", "nk>hidden", " reasoning</thi", "nk>Ada", " 1 EOI."]
    visible = "".join(thinking_filter.feed(chunk) for chunk in chunks)
    visible += thinking_filter.flush()

    assert visible == "Ada 1 EOI."

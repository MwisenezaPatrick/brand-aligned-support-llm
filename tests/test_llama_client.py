"""Mocked tests for the llama.cpp client. No model server or network is used.

Every client here runs on httpx.MockTransport, so pytest never contacts a server. These tests
check response format and shape. They do not check business claims, and they do not check that
the system prompt was sent; tests/test_prompting.py covers that.
"""

import dataclasses
import json

import httpx
import pytest

from training.inference.llama_client import (
    ChatResult,
    EmptyAnswerError,
    GenerationSettings,
    LlamaClientError,
    LlamaServerClient,
    ReasoningLeakError,
    TruncatedResponseError,
)

MESSAGES = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Does it have a lid?"},
]


def make_client(handler, base_url="http://127.0.0.1:8080"):
    return LlamaServerClient(base_url=base_url, transport=httpx.MockTransport(handler))


def reply(content="Yes, it has a lid.", finish="stop", reasoning=None):
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {
        "choices": [{"message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 7},
        "timings": {"predicted_per_second": 27.0},
    }


# Successful replies


def test_returns_answer_usage_and_timings():
    result = make_client(lambda r: httpx.Response(200, json=reply())).chat(MESSAGES)
    assert result.content == "Yes, it has a lid."
    assert result.finish_reason == "stop"
    assert (result.prompt_tokens, result.completion_tokens) == (12, 7)
    assert result.timings["predicted_per_second"] == 27.0


def test_chat_result_type_has_no_reasoning_field():
    fields = {field.name for field in dataclasses.fields(ChatResult)}
    assert "reasoning" not in fields
    assert "reasoning_content" not in fields


# Request settings


def test_request_payload_disables_thinking_and_uses_greedy_decoding():
    sent = {}

    def handler(request):
        sent.update(json.loads(request.content))
        return httpx.Response(200, json=reply())

    make_client(handler).chat(MESSAGES)
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}
    assert sent["temperature"] == 0.0
    assert sent["stream"] is False


def test_thinking_switch_is_omitted_when_thinking_is_enabled():
    sent = {}

    def handler(request):
        sent.update(json.loads(request.content))
        return httpx.Response(200, json=reply(reasoning="working it out"))

    result = make_client(handler).chat(MESSAGES, GenerationSettings(disable_thinking=False))
    assert "chat_template_kwargs" not in sent
    assert result.content == "Yes, it has a lid."
    assert "working it out" not in result.content


# Reasoning is rejected while thinking is disabled


def test_reasoning_with_length_finish_is_rejected_as_leak():
    client = make_client(
        lambda r: httpx.Response(200, json=reply(content="", finish="length", reasoning="step 1"))
    )
    with pytest.raises(ReasoningLeakError):
        client.chat(MESSAGES)


def test_reasoning_only_reply_with_stop_is_a_leak_not_an_empty_answer():
    client = make_client(
        lambda r: httpx.Response(200, json=reply(content="", finish="stop", reasoning="step 1"))
    )
    with pytest.raises(ReasoningLeakError):
        client.chat(MESSAGES)


def test_thinking_tags_in_the_answer_are_rejected():
    client = make_client(lambda r: httpx.Response(200, json=reply(content="<think>x</think>Answer")))
    with pytest.raises(ReasoningLeakError):
        client.chat(MESSAGES)


# Empty and truncated replies are never returned


def test_stop_with_empty_answer_is_rejected():
    client = make_client(lambda r: httpx.Response(200, json=reply(content="", finish="stop")))
    with pytest.raises(EmptyAnswerError):
        client.chat(MESSAGES)


def test_stop_with_whitespace_only_answer_is_rejected():
    client = make_client(lambda r: httpx.Response(200, json=reply(content="   \n\t", finish="stop")))
    with pytest.raises(EmptyAnswerError):
        client.chat(MESSAGES)


def test_null_content_with_stop_is_rejected_as_empty():
    client = make_client(lambda r: httpx.Response(200, json=reply(content=None, finish="stop")))
    with pytest.raises(EmptyAnswerError):
        client.chat(MESSAGES)


def test_length_with_empty_answer_is_rejected_as_truncated():
    client = make_client(lambda r: httpx.Response(200, json=reply(content="", finish="length")))
    with pytest.raises(TruncatedResponseError):
        client.chat(MESSAGES)


def test_length_with_partial_answer_is_rejected_as_truncated():
    client = make_client(lambda r: httpx.Response(200, json=reply(content="Partial answer", finish="length")))
    with pytest.raises(TruncatedResponseError):
        client.chat(MESSAGES)


@pytest.mark.parametrize("finish", ["tool_calls", None])
def test_unexpected_finish_reason_is_rejected(finish):
    client = make_client(lambda r: httpx.Response(200, json=reply(finish=finish)))
    with pytest.raises(LlamaClientError, match="unexpected finish reason"):
        client.chat(MESSAGES)


# Malformed responses


def test_empty_choices_list_is_malformed():
    client = make_client(lambda r: httpx.Response(200, json={"choices": [], "usage": {}}))
    with pytest.raises(LlamaClientError, match="malformed"):
        client.chat(MESSAGES)


def test_non_json_success_response_is_reported():
    client = make_client(lambda r: httpx.Response(200, text="<html>ok</html>"))
    with pytest.raises(LlamaClientError, match="not valid JSON"):
        client.chat(MESSAGES)


def test_unexpected_json_shape_is_malformed():
    client = make_client(lambda r: httpx.Response(200, json={"unexpected": True}))
    with pytest.raises(LlamaClientError, match="malformed"):
        client.chat(MESSAGES)


# Input validation: refused before any request is sent


@pytest.mark.parametrize(
    "messages",
    [[], [{"role": "bot", "content": "hi"}], [{"role": "user", "content": "   "}], [{"role": "user"}]],
)
def test_invalid_messages_are_refused_before_any_request(messages):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=reply())

    with pytest.raises(ValueError):
        make_client(handler).chat(messages)
    assert calls == []


def test_history_with_extra_fields_is_refused_before_sending():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=reply())

    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "reasoning_content": "partial thoughts"},
    ]
    with pytest.raises(ValueError, match="not allowed"):
        make_client(handler).chat(history)
    assert calls == []


# Addresses


@pytest.mark.parametrize(
    "base_url",
    ["https://example.com:8080", "http://10.0.0.5:8080", "http://localhost.example.com:8080"],
)
def test_non_local_addresses_are_refused(base_url):
    with pytest.raises(ValueError, match="localhost"):
        LlamaServerClient(base_url=base_url)


@pytest.mark.parametrize("base_url", ["http://127.0.0.1:8080", "http://localhost:8080"])
def test_local_addresses_are_accepted(base_url):
    client = LlamaServerClient(base_url=base_url)
    client.close()


# Transport errors


def test_http_error_is_reported_with_status_and_a_truncated_body():
    client = make_client(lambda r: httpx.Response(500, text="x" * 1000))
    with pytest.raises(LlamaClientError) as info:
        client.chat(MESSAGES)
    assert info.value.status_code == 500
    assert len(str(info.value)) < 400


def test_unreachable_server_is_reported():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(LlamaClientError, match="cannot reach"):
        make_client(handler).chat(MESSAGES)


def test_timeout_is_reported():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(LlamaClientError, match="timed out"):
        make_client(handler).chat(MESSAGES)


def test_invalid_generation_settings_are_refused():
    with pytest.raises(ValueError):
        GenerationSettings(max_tokens=0)
    with pytest.raises(ValueError):
        GenerationSettings(temperature=-1.0)

"""Client for a local llama.cpp server with an OpenAI-compatible chat endpoint.

What this client checks: response format and shape only.
- It accepts only localhost addresses (127.0.0.1 or localhost), so requests stay on this machine.
- It disables thinking on every request by default, with chat_template_kwargs enable_thinking false.
- It uses greedy decoding (temperature 0) and does not stream.
- It returns a reply only if the reply has final answer text, finished normally (finish reason
  "stop"), and contains no reasoning output. Reasoning is detected from the reasoning_content
  field and from <think> and </think> tags. Reasoning written as plain text without tags is
  not detected.
- It rejects messages that contain fields other than role and content, so reasoning output
  cannot be sent back in conversation history.
- ChatResult has no reasoning field.

What this client does NOT check: business claims. It does not know the brand profile, does
not add the system prompt, and cannot tell whether an answer is grounded, correctly routed,
or policy-compliant. Send conversations through training.inference.prompting.PromptedChat,
which adds the approved system prompt.

The client writes no files and reads no credentials.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

ALLOWED_HOSTS = ("127.0.0.1", "localhost")
CHAT_PATH = "/v1/chat/completions"
# llama-server answers with the model it has loaded. This value is informational only.
MODEL_ALIAS = "qwen3.5-0.8b"
VALID_ROLES = ("system", "user", "assistant")
ALLOWED_MESSAGE_KEYS = frozenset({"role", "content"})
THINK_MARKERS = ("<think>", "</think>")
ACCEPTED_FINISH_REASON = "stop"
MAX_ERROR_BODY_CHARS = 300


class LlamaClientError(RuntimeError):
    """Base class for every failure the client reports."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class TruncatedResponseError(LlamaClientError):
    """The model reached the token limit. Any partial text is not returned."""


class EmptyAnswerError(LlamaClientError):
    """The reply has no final answer text."""


class ReasoningLeakError(LlamaClientError):
    """Reasoning output or thinking tags appeared while thinking was disabled."""


@dataclass(frozen=True)
class GenerationSettings:
    temperature: float = 0.0
    max_tokens: int = 200
    disable_thinking: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if not 1 <= self.max_tokens <= 4096:
            raise ValueError("max_tokens must be between 1 and 4096")


@dataclass(frozen=True)
class ChatResult:
    content: str
    finish_reason: str
    prompt_tokens: int | None
    completion_tokens: int | None
    timings: Any
    wall_seconds: float


@dataclass(frozen=True)
class _RawReply:
    content: str
    reasoning: str
    finish_reason: Any
    usage: dict[str, Any]
    timings: Any


def _check_base_url(base_url: str) -> None:
    url = httpx.URL(base_url)
    if url.scheme != "http" or url.host not in ALLOWED_HOSTS:
        raise ValueError("base_url must use http with localhost or 127.0.0.1")


def _validate_messages(messages: Any) -> None:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"message {index} must be an object")
        extra = set(message) - ALLOWED_MESSAGE_KEYS
        if extra:
            raise ValueError(f"message {index} has fields that are not allowed: {sorted(map(str, extra))}")
        if message.get("role") not in VALID_ROLES:
            raise ValueError(f"message {index} must have a role of system, user, or assistant")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"message {index} must have non-empty text")


def _parse_reply(response: httpx.Response) -> _RawReply:
    try:
        data = response.json()
    except ValueError as exc:
        raise LlamaClientError("response is not valid JSON") from exc
    try:
        choice = data["choices"][0]
        message = choice["message"]
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
        finish_reason = choice.get("finish_reason")
        usage = data.get("usage") or {}
        timings = data.get("timings")
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise LlamaClientError("malformed response from the llama.cpp server") from exc
    if not isinstance(content, str) or not isinstance(reasoning, str) or not isinstance(usage, dict):
        raise LlamaClientError("malformed response from the llama.cpp server")
    return _RawReply(content, reasoning, finish_reason, usage, timings)


class LlamaServerClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        _check_base_url(base_url)
        self._client = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "LlamaServerClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def chat(
        self,
        messages: list[dict[str, str]],
        settings: GenerationSettings | None = None,
    ) -> ChatResult:
        settings = settings or GenerationSettings()
        _validate_messages(messages)
        payload: dict[str, Any] = {
            "model": MODEL_ALIAS,
            "messages": messages,
            "temperature": settings.temperature,
            "max_tokens": settings.max_tokens,
            "stream": False,
        }
        if settings.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        start = time.perf_counter()
        try:
            response = self._client.post(CHAT_PATH, json=payload)
        except httpx.TimeoutException as exc:
            raise LlamaClientError("request timed out") from exc
        except httpx.HTTPError as exc:
            raise LlamaClientError(f"cannot reach the llama.cpp server ({type(exc).__name__})") from exc
        wall_seconds = time.perf_counter() - start

        if response.status_code != 200:
            raise LlamaClientError(
                f"server returned HTTP {response.status_code}: {response.text[:MAX_ERROR_BODY_CHARS]}",
                response.status_code,
            )

        reply = _parse_reply(response)

        if settings.disable_thinking:
            if reply.reasoning.strip():
                raise ReasoningLeakError("reasoning output appeared while thinking is disabled")
            if any(marker in reply.content for marker in THINK_MARKERS):
                raise ReasoningLeakError("thinking tags appeared in the answer while thinking is disabled")
        if reply.finish_reason == "length":
            raise TruncatedResponseError("reply reached the token limit and was not returned")
        if reply.finish_reason != ACCEPTED_FINISH_REASON:
            raise LlamaClientError(f"unexpected finish reason: {reply.finish_reason!r}")
        if not reply.content.strip():
            raise EmptyAnswerError("reply has no final answer text")

        return ChatResult(
            content=reply.content,
            finish_reason=reply.finish_reason,
            prompt_tokens=reply.usage.get("prompt_tokens"),
            completion_tokens=reply.usage.get("completion_tokens"),
            timings=reply.timings,
            wall_seconds=wall_seconds,
        )

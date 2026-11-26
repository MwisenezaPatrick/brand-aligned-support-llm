"""Render the approved system prompt from the brand profile, and send chats through it.

The template file has a two-key header, one line that is exactly '---', and a body with the
five placeholders {{CATALOGUE}}, {{ESCALATION_TABLE}}, {{ROUTING_TABLE}},
{{THREE_TEAM_EXAMPLE}}, and {{UNKNOWN_FACTS}}. Each placeholder must appear exactly once, and
each is filled from the profile before anything is sent.

Hashes are SHA-256 over UTF-8 text, after converting line endings to LF and stripping
surrounding whitespace:
- profile_file_sha256: the raw bytes of the profile file
- template_sha256: the template body before rendering
- prompt_sha256: the rendered prompt that the model receives

PromptedChat is the intended path to the model. Its ask() method builds the messages with
build_messages(), which places the rendered system prompt first, and returns the reply with
the prompt record that a run manifest must store.

What this module does NOT do:
- It does not guarantee that the model follows the prompt.
- It does not prevent unsupported business claims, wrong routing, or wrong escalation. Those
  depend on the model, and only controlled live tests can show how it behaves.
- It cannot stop code that imports LlamaServerClient directly. A source scan in
  tests/test_prompting.py checks that no training, evaluation, app, or script module calls the
  raw client outside this wrapper.

This module reads two local files and makes no network calls.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.inference.brand import (
    DEFAULT_PROFILE_PATH,
    Profile,
    ProfileError,
    load_profile,
    render_catalogue,
    render_escalation_table,
    render_routing_table,
    render_three_team_example,
    render_unknown_facts,
)
from training.inference.llama_client import ChatResult, GenerationSettings, LlamaServerClient

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE_PATH = ROOT / "prompts" / "support_system_v1.md"
SEPARATOR = "\n---\n"
REQUIRED_HEADER_KEYS = ("prompt_version", "profile_version")
PLACEHOLDERS = ("CATALOGUE", "ESCALATION_TABLE", "ROUTING_TABLE", "THREE_TEAM_EXAMPLE", "UNKNOWN_FACTS")
PLACEHOLDER_PATTERN = re.compile(r"\{\{([A-Z_]+)\}\}")
VERSION_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
REQUIRED_PHRASES = (
    "PRIORITY RULE",
    "exactly one sentence",
    "overrides every other rule in this prompt",
    "Write nothing before that sentence and nothing after it",
    "MULTIPLE TEAMS RULE",
    "NEVER:",
)
ALLOWED_CONVERSATION_ROLES = ("user", "assistant")
ALLOWED_MESSAGE_KEYS = frozenset({"role", "content"})
MAX_PROMPT_CHARS = 8000  # a size budget for a small model, not a measure of quality


class PromptError(ValueError):
    """The template is malformed or does not match the profile, or the conversation is invalid."""


@dataclass(frozen=True)
class SystemPrompt:
    """A rendered system prompt and the record that identifies it."""

    body: str
    prompt_version: str
    profile_version: str
    profile_file_sha256: str
    template_sha256: str
    prompt_sha256: str

    def record(self) -> dict[str, str]:
        return {
            "prompt_version": self.prompt_version,
            "profile_version": self.profile_version,
            "profile_file_sha256": self.profile_file_sha256,
            "template_sha256": self.template_sha256,
            "prompt_sha256": self.prompt_sha256,
        }


@dataclass(frozen=True)
class PromptedReply:
    """A reply together with the prompt record that a manifest must store."""

    chat: ChatResult
    prompt: dict[str, str]

    @property
    def content(self) -> str:
        return self.chat.content


def _normalise(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_template(text: str) -> tuple[dict[str, str], str]:
    text = _normalise(text)
    if text.count(SEPARATOR) != 1:
        raise PromptError("the template needs exactly one '---' line between its header and body")
    header_text, body = text.split(SEPARATOR, 1)
    header: dict[str, str] = {}
    for line in header_text.splitlines():
        if not line.strip():
            continue
        if ": " not in line:
            raise PromptError(f"malformed header line: {line!r}")
        key, value = line.split(": ", 1)
        key = key.strip()
        if key in header:
            raise PromptError(f"duplicate header key: {key}")
        header[key] = value.strip()
    unexpected = sorted(set(header) - set(REQUIRED_HEADER_KEYS))
    if unexpected:
        raise PromptError(f"unexpected header keys: {unexpected}")
    missing = [key for key in REQUIRED_HEADER_KEYS if key not in header]
    if missing:
        raise PromptError(f"missing header keys: {missing}")
    for key in REQUIRED_HEADER_KEYS:
        if not VERSION_PATTERN.fullmatch(header[key]):
            raise PromptError(f"{key} must use lowercase letters, digits, and single hyphens")
    body = body.strip()
    if not body:
        raise PromptError("the template body is empty")
    return header, body


def _check_placeholders(body: str) -> None:
    found = PLACEHOLDER_PATTERN.findall(body)
    unknown = sorted(set(found) - set(PLACEHOLDERS))
    if unknown:
        raise PromptError(f"unknown placeholders in the template: {unknown}")
    for name in PLACEHOLDERS:
        count = found.count(name)
        if count != 1:
            raise PromptError(f"placeholder {{{{{name}}}}} must appear exactly once, found {count}")
    if re.search(r"[{}]", PLACEHOLDER_PATTERN.sub("", body)):
        raise PromptError("the template contains braces outside the placeholders")


def _render(body: str, profile: Profile) -> str:
    try:
        values = {
            "CATALOGUE": render_catalogue(profile),
            "ESCALATION_TABLE": render_escalation_table(profile),
            "ROUTING_TABLE": render_routing_table(profile),
            "THREE_TEAM_EXAMPLE": render_three_team_example(profile),
            "UNKNOWN_FACTS": render_unknown_facts(profile),
        }
    except ValueError as exc:
        raise PromptError(f"cannot render the prompt from the profile: {exc}") from exc
    rendered = PLACEHOLDER_PATTERN.sub(lambda match: values[match.group(1)], body).strip()
    if re.search(r"[{}]", rendered):
        raise PromptError("the rendered prompt contains braces")
    missing = [phrase for phrase in REQUIRED_PHRASES if phrase not in rendered]
    if missing:
        raise PromptError(f"the rendered prompt lacks required rule text: {missing}")
    if len(rendered) > MAX_PROMPT_CHARS:
        raise PromptError(
            f"the rendered prompt is {len(rendered)} characters, over the {MAX_PROMPT_CHARS} budget"
        )
    return rendered


def load_system_prompt(
    template_path: Path = DEFAULT_TEMPLATE_PATH,
    profile_path: Path = DEFAULT_PROFILE_PATH,
) -> SystemPrompt:
    """Read the template and profile, render the prompt, and return it with its record."""
    try:
        template_text = Path(template_path).read_bytes().decode("utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise PromptError(f"cannot read the template: {exc}") from exc
    try:
        profile = load_profile(profile_path)
    except ProfileError as exc:
        raise PromptError(f"the profile is invalid: {exc}") from exc

    header, body = _parse_template(template_text)
    if header["profile_version"] != profile.profile_version:
        raise PromptError(
            f"template expects profile {header['profile_version']!r}, "
            f"but the profile is {profile.profile_version!r}"
        )
    _check_placeholders(body)
    rendered = _render(body, profile)
    return SystemPrompt(
        body=rendered,
        prompt_version=header["prompt_version"],
        profile_version=profile.profile_version,
        profile_file_sha256=profile.file_sha256,
        template_sha256=_sha256(body),
        prompt_sha256=_sha256(rendered),
    )


def build_messages(system: SystemPrompt, conversation: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Return the rendered system prompt followed by the conversation. The input is not modified.

    The conversation may hold only user and assistant messages, each with exactly role and
    content fields. It must end with a user message. Text in the system role comes only from
    the prompt.
    """
    if not isinstance(system, SystemPrompt):
        raise PromptError("system must come from load_system_prompt()")
    if not isinstance(conversation, list) or not conversation:
        raise PromptError("the conversation must be a non-empty list")
    cleaned: list[dict[str, str]] = []
    for index, message in enumerate(conversation):
        if not isinstance(message, dict):
            raise PromptError(f"message {index} must be an object")
        extra = set(message) - ALLOWED_MESSAGE_KEYS
        if extra:
            raise PromptError(f"message {index} has fields that are not allowed: {sorted(map(str, extra))}")
        role = message.get("role")
        if role not in ALLOWED_CONVERSATION_ROLES:
            raise PromptError(
                f"message {index} must have role user or assistant; system text comes only from the prompt"
            )
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise PromptError(f"message {index} must have non-empty text")
        cleaned.append({"role": role, "content": content})
    if cleaned[-1]["role"] != "user":
        raise PromptError("the conversation must end with a user message")
    return [{"role": "system", "content": system.body}, *cleaned]


class PromptedChat:
    """The intended path to the model. ask() always places the rendered system prompt first."""

    def __init__(self, system: SystemPrompt, client: LlamaServerClient) -> None:
        if not isinstance(system, SystemPrompt):
            raise TypeError("system must come from load_system_prompt()")
        if not isinstance(client, LlamaServerClient):
            raise TypeError("client must be a LlamaServerClient")
        self._system = system
        self._client = client

    def prompt_record(self) -> dict[str, str]:
        return self._system.record()

    def ask(
        self,
        conversation: list[dict[str, Any]],
        settings: GenerationSettings | None = None,
    ) -> PromptedReply:
        messages = build_messages(self._system, conversation)
        chat = self._client.chat(messages, settings or GenerationSettings())
        return PromptedReply(chat=chat, prompt=self._system.record())

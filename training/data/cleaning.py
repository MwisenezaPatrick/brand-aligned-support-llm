"""Clean loaded customer-support conversations.

Input is the canonical conversation list produced by the loaders in
training/data/ingest.py. Output is a new list containing only conversations
that are complete and unique, plus a report of every change and exclusion.
Excluded conversations are also returned, already normalized, so the
redaction step can cover them as well.

This module never reads or writes files and never modifies its input. Cleaning
is deterministic: the same input always gives the same output and report.

Steps, applied to each conversation in input order:

1. Normalize each message's text (see normalize_text).
2. Remove messages whose text is empty after normalization.
3. Collapse consecutive messages with the same role and identical normalized
   text into one message. Repeats separated by other messages are kept.
4. Exclude the conversation if no messages remain.
5. Exclude the conversation if its conversation_id was already seen, or if its
   customer_id and normalized messages match an earlier record. The first
   occurrence is kept.
6. Exclude and flag conversations with no agent reply or no customer message.
   They are kept out of the cleaned set so training never treats them as
   successful support conversations.

Steps 5 and 6 compare every record that reaches them, including incomplete
ones, so a later copy of an incomplete record is reported as a duplicate.

Not done here: PII redaction, categorization, and instruction-response
conversion. Those happen in later steps.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

CLEANING_VERSION = "1"

REASON_NO_MESSAGES = "no_messages_after_cleaning"
REASON_DUPLICATE_ID = "duplicate_conversation_id"
REASON_DUPLICATE_CONTENT = "duplicate_content"
REASON_INCOMPLETE_NO_AGENT = "incomplete_no_agent_reply"
REASON_INCOMPLETE_NO_CUSTOMER = "incomplete_no_customer_message"

ACTION_REMOVED_EMPTY = "removed_empty"
ACTION_COLLAPSED_DUPLICATE = "collapsed_duplicate"

# Byte-order marks and soft hyphens carry no meaning in support text.
# Zero-width spaces are left alone: removing them could join two words.
_INVISIBLE_CHARACTERS = dict.fromkeys(map(ord, "\ufeff\u00ad"))
_SPACE_RUN = re.compile(r"[^\S\n]+")
_BLANK_LINE_RUN = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Return the cleaned form of one message's text.

    Changes: Unicode NFC form, invisible characters removed, line endings
    unified to LF, runs of spaces and tabs (including non-breaking spaces)
    collapsed to one space, spaces at line ends removed, runs of three or more
    newlines reduced to a blank line, and surrounding whitespace removed.

    Unchanged: letters and their case, punctuation, digits, emoji, and the
    order of words and lines.
    """
    text = unicodedata.normalize("NFC", text).translate(_INVISIBLE_CHARACTERS)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_SPACE_RUN.sub(" ", line).strip(" ") for line in text.split("\n")]
    text = _BLANK_LINE_RUN.sub("\n\n", "\n".join(lines))
    return text.strip()


@dataclass(frozen=True)
class MessageAction:
    """A change made to one message. message_index is its position as loaded, from 0."""

    conversation_id: str
    message_index: int
    action: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "message_index": self.message_index,
            "action": self.action,
        }


@dataclass(frozen=True)
class Exclusion:
    """A conversation left out of the cleaned set.

    original_conversation_id names the earlier record for duplicate reasons.
    """

    conversation_id: str
    reason: str
    original_conversation_id: str | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "reason": self.reason,
            "original_conversation_id": self.original_conversation_id,
            "detail": self.detail,
        }


@dataclass
class CleaningReport:
    """Counts before and after cleaning, plus every change and exclusion."""

    cleaning_version: str = CLEANING_VERSION
    input_conversations: int = 0
    input_messages: int = 0
    output_conversations: int = 0
    output_messages: int = 0
    message_actions: list[MessageAction] = field(default_factory=list)
    exclusions: list[Exclusion] = field(default_factory=list)

    def counts_by_reason(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for exclusion in self.exclusions:
            counts[exclusion.reason] = counts.get(exclusion.reason, 0) + 1
        return dict(sorted(counts.items()))

    def count_actions(self, action: str) -> int:
        return sum(1 for item in self.message_actions if item.action == action)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cleaning_version": self.cleaning_version,
            "before": {"conversations": self.input_conversations, "messages": self.input_messages},
            "after": {"conversations": self.output_conversations, "messages": self.output_messages},
            "messages_removed_empty": self.count_actions(ACTION_REMOVED_EMPTY),
            "messages_collapsed_duplicate": self.count_actions(ACTION_COLLAPSED_DUPLICATE),
            "excluded_by_reason": self.counts_by_reason(),
            "exclusions": [exclusion.to_dict() for exclusion in self.exclusions],
            "message_actions": [item.to_dict() for item in self.message_actions],
        }


@dataclass
class CleaningResult:
    """conversations: the kept set. excluded: excluded conversations, normalized
    but not filtered, in input order. report: what changed and why."""

    conversations: list[dict[str, Any]]
    report: CleaningReport
    excluded: list[dict[str, Any]] = field(default_factory=list)


def clean_conversations(conversations: list[dict[str, Any]]) -> CleaningResult:
    """Clean canonical conversations. The input must already pass validate_conversation.

    The input is not modified. Excluded conversations are returned in `excluded`,
    already normalized, so redaction can cover them. They never appear in
    `conversations`.
    """
    report = CleaningReport(
        input_conversations=len(conversations),
        input_messages=sum(len(conversation["messages"]) for conversation in conversations),
    )
    cleaned_out: list[dict[str, Any]] = []
    excluded_out: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_content: dict[tuple, str] = {}

    def exclude(
        conversation_id: str,
        customer_id: str,
        messages: list[dict[str, str]],
        exclusion: Exclusion,
    ) -> None:
        report.exclusions.append(exclusion)
        excluded_out.append({
            "conversation_id": conversation_id,
            "customer_id": customer_id,
            "messages": messages,
        })

    for conversation in conversations:
        conversation_id = conversation["conversation_id"]
        customer_id = conversation["customer_id"]
        messages = _clean_messages(conversation_id, conversation["messages"], report)

        if not messages:
            exclude(conversation_id, customer_id, messages, Exclusion(
                conversation_id,
                REASON_NO_MESSAGES,
                detail="every message was empty after normalization",
            ))
            continue

        if conversation_id in seen_ids:
            exclude(conversation_id, customer_id, messages, Exclusion(
                conversation_id,
                REASON_DUPLICATE_ID,
                conversation_id,
                "an earlier record already uses this conversation_id",
            ))
            continue

        content_key = (customer_id, tuple((m["role"], m["text"]) for m in messages))
        if content_key in seen_content:
            original = seen_content[content_key]
            exclude(conversation_id, customer_id, messages, Exclusion(
                conversation_id,
                REASON_DUPLICATE_CONTENT,
                original,
                f"same customer and messages as earlier record {original!r}",
            ))
            continue

        seen_ids.add(conversation_id)
        seen_content[content_key] = conversation_id

        roles = {message["role"] for message in messages}
        if "agent" not in roles:
            exclude(conversation_id, customer_id, messages, Exclusion(
                conversation_id,
                REASON_INCOMPLETE_NO_AGENT,
                detail="no agent reply; flagged as incomplete",
            ))
            continue
        if "customer" not in roles:
            exclude(conversation_id, customer_id, messages, Exclusion(
                conversation_id,
                REASON_INCOMPLETE_NO_CUSTOMER,
                detail="no customer message; flagged as incomplete",
            ))
            continue

        cleaned_out.append({
            "conversation_id": conversation_id,
            "customer_id": customer_id,
            "messages": messages,
        })

    report.output_conversations = len(cleaned_out)
    report.output_messages = sum(len(conversation["messages"]) for conversation in cleaned_out)
    return CleaningResult(conversations=cleaned_out, report=report, excluded=excluded_out)


def _clean_messages(
    conversation_id: str,
    messages: list[dict[str, Any]],
    report: CleaningReport,
) -> list[dict[str, str]]:
    cleaned: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        text = normalize_text(message["text"])
        if not text:
            report.message_actions.append(
                MessageAction(conversation_id, index, ACTION_REMOVED_EMPTY)
            )
            continue
        previous = cleaned[-1] if cleaned else None
        if previous is not None and previous["role"] == message["role"] and previous["text"] == text:
            report.message_actions.append(
                MessageAction(conversation_id, index, ACTION_COLLAPSED_DUPLICATE)
            )
            continue
        cleaned.append({"role": message["role"], "text": text})
    return cleaned

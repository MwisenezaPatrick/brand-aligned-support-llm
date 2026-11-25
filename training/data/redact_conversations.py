"""Apply PII redaction to cleaned conversations and build a redaction report.

Input is the cleaned conversation list from training/data/cleaning.py. Redaction
runs after cleaning, so duplicate detection sees the original text, and before
dataset splitting, so the splits only contain redacted text.

This module never reads or writes files and never modifies its input. The
report stores counts and locations only. It never stores the removed values,
so it is safe to log or commit.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from training.data.redaction import DEMO_FIRST_NAMES, DEMO_LAST_NAMES, PLACEHOLDERS, redact_text

REDACTION_VERSION = "1"


@dataclass(frozen=True)
class Location:
    """Where a redaction happened: a conversation and a message position (from 0)."""

    conversation_id: str
    message_index: int

    def to_dict(self) -> dict[str, Any]:
        return {"conversation_id": self.conversation_id, "message_index": self.message_index}


@dataclass
class RedactionReport:
    """Counts and locations per category. Holds no removed values."""

    redaction_version: str = REDACTION_VERSION
    conversations: int = 0
    messages: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    locations: dict[str, list[Location]] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def to_dict(self) -> dict[str, Any]:
        categories = sorted(PLACEHOLDERS)
        return {
            "redaction_version": self.redaction_version,
            "conversations": self.conversations,
            "messages": self.messages,
            "total_redactions": self.total,
            "counts": {kind: self.counts.get(kind, 0) for kind in categories},
            "locations": {
                kind: [location.to_dict() for location in self.locations.get(kind, [])]
                for kind in categories
            },
        }


@dataclass
class RedactionResult:
    conversations: list[dict[str, Any]]
    report: RedactionReport


def redact_conversations(
    conversations: Sequence[dict[str, Any]],
    first_names: Sequence[str] = DEMO_FIRST_NAMES,
    last_names: Sequence[str] = DEMO_LAST_NAMES,
) -> RedactionResult:
    """Redact every message. Conversation IDs, customer IDs, roles, and order are kept."""
    report = RedactionReport(
        conversations=len(conversations),
        messages=sum(len(conversation["messages"]) for conversation in conversations),
    )
    redacted: list[dict[str, Any]] = []

    for conversation in conversations:
        conversation_id = conversation["conversation_id"]
        new_messages: list[dict[str, Any]] = []
        for index, message in enumerate(conversation["messages"]):
            result = redact_text(message["text"], first_names, last_names)
            new_messages.append({"role": message["role"], "text": result.text})
            for kind, count in result.counts.items():
                if count:
                    report.counts[kind] = report.counts.get(kind, 0) + count
                    report.locations.setdefault(kind, []).append(
                        Location(conversation_id, index)
                    )
        redacted.append({
            "conversation_id": conversation_id,
            "customer_id": conversation["customer_id"],
            "messages": new_messages,
        })

    return RedactionResult(conversations=redacted, report=report)


def redact_conversation_copy(conversation: dict[str, Any]) -> dict[str, Any]:
    """Return a redacted deep copy of one conversation. Useful for previews."""
    return redact_conversations([copy.deepcopy(conversation)]).conversations[0]
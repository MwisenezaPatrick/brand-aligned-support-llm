"""Content rules for customer-support conversations.

This module checks the canonical conversation shape that the loaders in
training/data/ingest.py produce. It never reads files.

Canonical conversation:

    {
        "conversation_id": "conv-0001",
        "customer_id": "cust-0001",
        "messages": [
            {"role": "customer", "text": "Does the kettle come with a lid?"},
            {"role": "agent", "text": "Yes, it does."}
        ]
    }

Rules enforced here:
- conversation_id and customer_id are non-empty and use only letters, digits,
  '.', '_' and '-' (up to 100 characters).
- messages is a non-empty list.
- every message has a role of exactly "customer" or "agent".
- every message has non-empty text.

Rules deliberately not enforced here:
- A conversation with no agent reply is valid. The cleaning step flags it as incomplete.
- Duplicates and PII are handled in later steps.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

VALID_ROLES = ("customer", "agent")
ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}")


@dataclass(frozen=True)
class SchemaError:
    """One rule a record breaks.

    `field` names the offending key, such as "messages[2].role".
    `message_index` is set when the problem is inside one message, so the
    loader can map it back to the source line.
    """

    field: str
    reason: str
    message_index: int | None = None

    def __str__(self) -> str:
        return f"{self.field} {self.reason}"


def _check_id(value: Any, field_name: str) -> SchemaError | None:
    if not isinstance(value, str) or not value:
        return SchemaError(field_name, "must be a non-empty string")
    if not ID_PATTERN.fullmatch(value):
        return SchemaError(
            field_name,
            "may use only letters, digits, '.', '_' and '-' (1 to 100 characters)",
        )
    return None


def validate_message(message: Any, index: int) -> list[SchemaError]:
    """Check one message at position `index` within its conversation."""
    prefix = f"messages[{index}]"
    if not isinstance(message, Mapping):
        return [SchemaError(prefix, "must be an object", index)]

    errors: list[SchemaError] = []

    if "role" not in message:
        errors.append(SchemaError(f"{prefix}.role", "is missing", index))
    elif not isinstance(message["role"], str) or message["role"] not in VALID_ROLES:
        errors.append(
            SchemaError(
                f"{prefix}.role",
                f"must be 'customer' or 'agent', got {message['role']!r}",
                index,
            )
        )

    if "text" not in message:
        errors.append(SchemaError(f"{prefix}.text", "is missing", index))
    elif not isinstance(message["text"], str):
        errors.append(SchemaError(f"{prefix}.text", "must be a string", index))
    elif not message["text"].strip():
        errors.append(SchemaError(f"{prefix}.text", "must not be empty", index))

    return errors


def validate_conversation(record: Any) -> list[SchemaError]:
    """Return every rule the record breaks. An empty list means the record is valid."""
    if not isinstance(record, Mapping):
        return [SchemaError("record", "must be an object")]

    errors: list[SchemaError] = []

    for field_name in ("conversation_id", "customer_id"):
        if field_name not in record:
            errors.append(SchemaError(field_name, "is missing"))
        else:
            error = _check_id(record[field_name], field_name)
            if error is not None:
                errors.append(error)

    if "messages" not in record:
        errors.append(SchemaError("messages", "is missing"))
        return errors

    messages = record["messages"]
    if not isinstance(messages, list):
        errors.append(SchemaError("messages", "must be a list"))
    elif not messages:
        errors.append(SchemaError("messages", "must contain at least one message"))
    else:
        for index, message in enumerate(messages):
            errors.extend(validate_message(message, index))

    return errors
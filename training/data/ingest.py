"""Load customer-support conversations from CSV, JSON, and JSONL files.

This module reads files and returns canonical conversations (the shape
described in training/data/schema.py). It does not clean text, remove
duplicates, or redact PII; those belong to later steps. Files are opened
read-only and are never modified.

CSV format: one row per message
-------------------------------

Required columns. Names are case-insensitive, and surrounding spaces are ignored:

    conversation_id   Identifier shared by every message in one conversation.
    customer_id       Identifier of the customer. Must be the same on every row
                      of one conversation.
    role              Exactly "customer" or "agent" (case-sensitive, after
                      surrounding spaces are removed).
    text              The message text. Quote any field that contains a comma
                      or a line break, as usual for CSV.

Optional ordering columns:

    message_order     A whole number. Lower values come first.
    timestamp         An ISO 8601 date and time, such as 2026-11-25T09:30:00Z.
                      Values without a timezone are read as UTC.

Example:

    conversation_id,customer_id,role,text,message_order,timestamp
    conv-0001,cust-0001,customer,Does the kettle come with a lid?,1,2026-11-25T09:00:00Z
    conv-0001,cust-0001,agent,"Yes, it includes a leak-resistant lid.",2,2026-11-25T09:02:00Z

Ordering rules
--------------

- Rows with the same conversation_id are merged, even when other rows sit between them.
- If message_order exists, messages are sorted by it. The timestamp column is not used.
- Otherwise, if timestamp exists, messages are sorted by it.
- Otherwise, messages keep their order in the file.
- Rows with equal sort values keep their order in the file.

Error rules
-----------

- A malformed row is reported with its line number and skipped. Every other row
  still loads. A row is malformed when:
  - it has a different number of fields than the header (often an unquoted comma in the text);
  - conversation_id, customer_id, role, or text is empty or invalid;
  - the ordering value is missing or invalid;
  - its customer_id differs from the customer_id already recorded for that
    conversation. The first row in the file wins.
- Blank lines and lines where every cell is empty are skipped silently.
- A file-level problem stops loading and returns no conversations: the file is
  missing, is not UTF-8, is empty, lacks a required column, or has a duplicated
  column name.
- Text is kept exactly as written. Only identifiers, role, and ordering values
  have surrounding spaces removed.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from training.data.schema import validate_conversation

REQUIRED_COLUMNS = ("conversation_id", "customer_id", "role", "text")
MESSAGE_ORDER_COLUMN = "message_order"
TIMESTAMP_COLUMN = "timestamp"
FILE_ORDER = "file order"


@dataclass(frozen=True)
class IngestError:
    """One problem found while loading.

    line_number is the physical line in the CSV file, or None for a problem
    that affects the whole file.
    """

    line_number: int | None
    message: str

    def __str__(self) -> str:
        if self.line_number is None:
            return self.message
        return f"line {self.line_number}: {self.message}"


@dataclass
class IngestResult:
    """What a load produced: canonical conversations plus everything rejected."""

    source: str
    conversations: list[dict[str, Any]] = field(default_factory=list)
    errors: list[IngestError] = field(default_factory=list)
    rows_read: int = 0
    rows_accepted: int = 0
    order_source: str = FILE_ORDER

    @property
    def rows_rejected(self) -> int:
        return self.rows_read - self.rows_accepted


@dataclass(frozen=True)
class _Layout:
    """Column positions and ordering mode, taken from the header row."""

    width: int
    index: dict[str, int]
    order_mode: str


@dataclass
class _Group:
    customer_id: str
    rows: list[tuple[tuple, str, str]] = field(default_factory=list)


def load_csv(path: str | Path) -> IngestResult:
    """Load one CSV file. The file is opened for reading only."""
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return parse_csv(handle, source=str(path))
    except FileNotFoundError:
        return _file_error(path, f"file not found: {path}")
    except UnicodeDecodeError as exc:
        return _file_error(path, f"file is not valid UTF-8 (problem at byte {exc.start})")
    except OSError as exc:
        return _file_error(path, f"cannot read file: {exc}")


def parse_csv(stream: TextIO, source: str = "<stream>") -> IngestResult:
    """Parse CSV text from an open text stream. See the module docstring for the format."""
    result = IngestResult(source=source)
    reader = csv.reader(stream)

    try:
        header = next(reader, None)
    except csv.Error as exc:
        result.errors.append(IngestError(1, f"cannot read header: {exc}"))
        return result

    if header is None:
        result.errors.append(IngestError(None, "file is empty"))
        return result

    layout = _read_layout(header, result)
    if layout is None:
        return result

    groups: dict[str, _Group] = {}
    previous_line = reader.line_num

    while True:
        try:
            cells = next(reader)
        except StopIteration:
            break
        except csv.Error as exc:
            result.errors.append(
                IngestError(
                    reader.line_num,
                    f"cannot parse this row; rows after it were not loaded: {exc}",
                )
            )
            break

        # A quoted field can span several physical lines, so compute the
        # row's first line from where the previous row ended.
        line_number = previous_line + 1
        previous_line = reader.line_num

        if _is_blank(cells):
            continue

        result.rows_read += 1
        fields, problems = _check_row(cells, layout, line_number)
        if fields is None:
            result.errors.append(IngestError(line_number, "; ".join(problems)))
            continue

        conversation_id = fields["conversation_id"]
        customer_id = fields["customer_id"]
        group = groups.get(conversation_id)
        if group is None:
            group = groups[conversation_id] = _Group(customer_id=customer_id)
        elif group.customer_id != customer_id:
            result.errors.append(
                IngestError(
                    line_number,
                    f"customer_id {customer_id!r} does not match {group.customer_id!r}, "
                    f"the customer already recorded for conversation {conversation_id!r}",
                )
            )
            continue

        group.rows.append((fields["sort_key"], fields["role"], fields["text"]))
        result.rows_accepted += 1

    for conversation_id, group in groups.items():
        ordered = sorted(group.rows, key=lambda row: row[0])
        result.conversations.append({
            "conversation_id": conversation_id,
            "customer_id": group.customer_id,
            "messages": [{"role": role, "text": text} for _, role, text in ordered],
        })
    return result


def _file_error(path: Path, message: str) -> IngestResult:
    result = IngestResult(source=str(path))
    result.errors.append(IngestError(None, message))
    return result


def _is_blank(cells: list[str]) -> bool:
    return all(not cell.strip() for cell in cells)


def _read_layout(header: list[str], result: IngestResult) -> _Layout | None:
    names = [name.strip().lower() for name in header]
    missing = [name for name in REQUIRED_COLUMNS if name not in names]
    duplicates = sorted({name for name in names if name and names.count(name) > 1})

    if missing:
        result.errors.append(
            IngestError(1, f"missing required column(s): {', '.join(missing)}")
        )
    if duplicates:
        result.errors.append(
            IngestError(1, f"duplicate column name(s): {', '.join(duplicates)}")
        )
    if result.errors:
        return None

    index = {name: position for position, name in enumerate(names)}
    if MESSAGE_ORDER_COLUMN in index:
        order_mode = MESSAGE_ORDER_COLUMN
    elif TIMESTAMP_COLUMN in index:
        order_mode = TIMESTAMP_COLUMN
    else:
        order_mode = FILE_ORDER
    result.order_source = order_mode
    return _Layout(width=len(names), index=index, order_mode=order_mode)


def _check_row(
    cells: list[str], layout: _Layout, line_number: int
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate one row.

    Returns (fields, []) when the row is valid, or (None, problems) when it is malformed.
    """
    if len(cells) != layout.width:
        hint = " (check for an unquoted comma in the text)" if len(cells) > layout.width else ""
        return None, [f"expected {layout.width} fields but found {len(cells)}{hint}"]

    index = layout.index
    conversation_id = cells[index["conversation_id"]].strip()
    customer_id = cells[index["customer_id"]].strip()
    role = cells[index["role"]].strip()
    text = cells[index["text"]]

    # Reuse the schema validator for IDs, role, and text. Its messages start
    # with "messages[0]." because the row is checked as a one-message conversation.
    problems = [
        str(error).replace("messages[0].", "", 1)
        for error in validate_conversation({
            "conversation_id": conversation_id,
            "customer_id": customer_id,
            "messages": [{"role": role, "text": text}],
        })
    ]

    sort_value: Any = None
    if layout.order_mode == MESSAGE_ORDER_COLUMN:
        raw = cells[index[MESSAGE_ORDER_COLUMN]].strip()
        try:
            sort_value = int(raw)
        except ValueError:
            problems.append(f"message_order must be a whole number, got {raw!r}")
    elif layout.order_mode == TIMESTAMP_COLUMN:
        raw = cells[index[TIMESTAMP_COLUMN]].strip()
        try:
            moment = datetime.fromisoformat(raw)
        except ValueError:
            problems.append(f"timestamp must be an ISO 8601 date and time, got {raw!r}")
        else:
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=timezone.utc)
            sort_value = moment

    if problems:
        return None, problems

    if layout.order_mode == FILE_ORDER:
        sort_key: tuple = (line_number,)
    else:
        sort_key = (sort_value, line_number)

    return {
        "conversation_id": conversation_id,
        "customer_id": customer_id,
        "role": role,
        "text": text,
        "sort_key": sort_key,
    }, []


# ---------------------------------------------------------------------------
# JSON arrays and JSONL files
# ---------------------------------------------------------------------------
#
# JSON file: one top-level array. Each element is a conversation in the
# canonical shape:
#
#     [
#       {
#         "conversation_id": "conv-0001",
#         "customer_id": "cust-0001",
#         "messages": [
#           {"role": "customer", "text": "Does the kettle come with a lid?"},
#           {"role": "agent", "text": "Yes, it does."}
#         ]
#       }
#     ]
#
# JSONL file: one conversation object per line, in the same shape.
#
# Rules for both formats:
# - Every record is checked with validate_conversation(). A failing record is
#   reported and skipped; the other records still load.
# - Only conversation_id, customer_id, messages, and each message's role and
#   text are copied to the output. Other fields in the source are ignored.
#   The source file is never modified.
# - Messages keep the order they have in the file.
# - Two records with the same conversation_id both load. Detecting duplicate
#   records is the cleaning step's job.
# - A UTF-8 byte order mark is accepted.
# - Blank lines in JSONL are skipped silently.
#
# Error reporting:
# - JSONL errors carry the physical line number.
# - A JSON array has no line numbers for its elements, so element errors are
#   reported by record number: the 1-based position in the array.
# - If the JSON text cannot be parsed at all, the error carries the line
#   number from the parser and no conversations are returned.

_BYTE_ORDER_MARK = "\ufeff"


def load_json(path: str | Path) -> IngestResult:
    """Load one JSON file containing an array of conversations. The file is read only."""
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return parse_json(handle.read(), source=str(path))
    except FileNotFoundError:
        return _file_error(path, f"file not found: {path}")
    except UnicodeDecodeError as exc:
        return _file_error(path, f"file is not valid UTF-8 (problem at byte {exc.start})")
    except OSError as exc:
        return _file_error(path, f"cannot read file: {exc}")


def parse_json(text: str, source: str = "<string>") -> IngestResult:
    """Parse JSON text whose top-level value is an array of conversations."""
    result = IngestResult(source=source)
    text = text.lstrip(_BYTE_ORDER_MARK)

    if not text.strip():
        result.errors.append(IngestError(None, "file is empty"))
        return result

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        result.errors.append(
            IngestError(exc.lineno, f"invalid JSON: {exc.msg} (column {exc.colno})")
        )
        return result

    if not isinstance(data, list):
        result.errors.append(
            IngestError(None, f"top-level value must be a JSON array, got {_json_type(data)}")
        )
        return result

    for record_number, record in enumerate(data, start=1):
        result.rows_read += 1
        errors = validate_conversation(record)
        if errors:
            problems = "; ".join(str(error) for error in errors)
            result.errors.append(
                IngestError(None, f"record {record_number}{_id_hint(record)}: {problems}")
            )
            continue
        result.conversations.append(_canonical(record))
        result.rows_accepted += 1

    return result


def load_jsonl(path: str | Path) -> IngestResult:
    """Load one JSONL file with one conversation per line. The file is read only."""
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return parse_jsonl(handle, source=str(path))
    except FileNotFoundError:
        return _file_error(path, f"file not found: {path}")
    except UnicodeDecodeError as exc:
        return _file_error(path, f"file is not valid UTF-8 (problem at byte {exc.start})")
    except OSError as exc:
        return _file_error(path, f"cannot read file: {exc}")


def parse_jsonl(stream: TextIO, source: str = "<stream>") -> IngestResult:
    """Parse JSONL text from an open text stream, one conversation per line."""
    result = IngestResult(source=source)
    saw_content = False

    for line_number, raw_line in enumerate(stream, start=1):
        if line_number == 1:
            raw_line = raw_line.lstrip(_BYTE_ORDER_MARK)
        line = raw_line.strip()
        if not line:
            continue
        saw_content = True
        result.rows_read += 1

        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            result.errors.append(
                IngestError(line_number, f"invalid JSON: {exc.msg} (column {exc.colno})")
            )
            continue

        errors = validate_conversation(record)
        if errors:
            problems = "; ".join(str(error) for error in errors)
            result.errors.append(IngestError(line_number, problems))
            continue

        result.conversations.append(_canonical(record))
        result.rows_accepted += 1

    if not saw_content:
        result.errors.append(IngestError(None, "file is empty"))
    return result


def _canonical(record: dict[str, Any]) -> dict[str, Any]:
    """Copy only the canonical fields, in the same shape the CSV loader returns."""
    return {
        "conversation_id": record["conversation_id"],
        "customer_id": record["customer_id"],
        "messages": [
            {"role": message["role"], "text": message["text"]}
            for message in record["messages"]
        ],
    }


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, (int, float)):
        return "a number"
    if isinstance(value, str):
        return "a string"
    return "an object"


def _id_hint(record: Any) -> str:
    """Name the conversation in an error message when its ID is readable."""
    if isinstance(record, dict) and isinstance(record.get("conversation_id"), str):
        return f" (conversation_id {record['conversation_id']!r})"
    return ""
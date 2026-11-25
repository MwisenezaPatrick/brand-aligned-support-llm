"""Write the redacted demo outputs to data/demo/.

Run from the project root:
    python -m training.data.write_redacted_demo

Three files are written:

- conversations_redacted.jsonl: retained conversations. This is the only file
  that training, validation, and test splits may read.
- excluded_conversations_redacted.jsonl: redacted duplicate and incomplete
  conversations, kept for review. Each record carries its exclusion_reason and
  original_conversation_id.
- redaction_report.json: counts, locations, excluded IDs, and the file policy.

The source file is only read. Real or client data must never be written to
data/demo/; it belongs in data/processed/, which is ignored by Git.
"""

import json
from pathlib import Path
from typing import Any

from training.data.cleaning import clean_conversations
from training.data.ingest import load_jsonl
from training.data.redact_conversations import redact_conversations

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "demo" / "conversations.jsonl"
REDACTED_OUT = ROOT / "data" / "demo" / "conversations_redacted.jsonl"
EXCLUDED_OUT = ROOT / "data" / "demo" / "excluded_conversations_redacted.jsonl"
REPORT_OUT = ROOT / "data" / "demo" / "redaction_report.json"

RETAINED_PATH = "data/demo/conversations_redacted.jsonl"
EXCLUDED_PATH = "data/demo/excluded_conversations_redacted.jsonl"
REPORT_PATH = "data/demo/redaction_report.json"


def _jsonl(records: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records)


def build_outputs() -> tuple[str, str, str]:
    """Return (retained JSONL, excluded JSONL, report JSON) without writing files."""
    loaded = load_jsonl(SOURCE)
    cleaned = clean_conversations(loaded.conversations)
    kept_count = len(cleaned.conversations)
    exclusions = cleaned.report.exclusions

    if len(exclusions) != len(cleaned.excluded):
        raise RuntimeError("exclusion records and excluded conversations do not match")

    # Kept conversations come first, so splitting by position is exact.
    redacted = redact_conversations(cleaned.conversations + cleaned.excluded)
    retained = redacted.conversations[:kept_count]
    excluded_redacted = redacted.conversations[kept_count:]

    excluded_records: list[dict[str, Any]] = []
    for record, exclusion in zip(excluded_redacted, exclusions):
        if record["conversation_id"] != exclusion.conversation_id:
            raise RuntimeError("excluded records are out of step with their exclusion reasons")
        excluded_records.append({
            "conversation_id": record["conversation_id"],
            "customer_id": record["customer_id"],
            "exclusion_reason": exclusion.reason,
            "original_conversation_id": exclusion.original_conversation_id,
            "messages": record["messages"],
        })

    report = redacted.report.to_dict()
    report["source_file"] = "data/demo/conversations.jsonl"
    report["scope"] = "every conversation that passed the loader, retained and excluded"
    report["cleaned_conversations"] = kept_count
    report["excluded_conversations"] = len(excluded_records)
    report["excluded_conversation_ids"] = [e.conversation_id for e in exclusions]
    report["outputs"] = {
        "retained_conversations": RETAINED_PATH,
        "excluded_conversations": EXCLUDED_PATH,
        "report": REPORT_PATH,
    }
    report["training_input_file"] = RETAINED_PATH
    report["excluded_file_is_training_input"] = False
    report_text = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    return _jsonl(retained), _jsonl(excluded_records), report_text


def main() -> None:
    retained_text, excluded_text, report_text = build_outputs()
    REDACTED_OUT.write_text(retained_text, encoding="utf-8", newline="\n")
    EXCLUDED_OUT.write_text(excluded_text, encoding="utf-8", newline="\n")
    REPORT_OUT.write_text(report_text, encoding="utf-8", newline="\n")
    print(f"Wrote {REDACTED_OUT}")
    print(f"Wrote {EXCLUDED_OUT}")
    print(f"Wrote {REPORT_OUT}")


if __name__ == "__main__":
    main()

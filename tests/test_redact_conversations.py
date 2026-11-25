"""Tests for conversation-level redaction, the redaction report, and the output files.

Redaction covers every conversation the cleaner saw. Three files are written:

- conversations_redacted.jsonl: retained conversations. This is the only file
  that the training, validation, and test splits may read.
- excluded_conversations_redacted.jsonl: redacted duplicate and incomplete
  conversations, for review only.
- redaction_report.json: counts, locations, excluded IDs, and the file policy.

The mandatory checks compare every output with the planted values in
planted_issues.json. A planted value found in any output fails the build.
"""

import json
from pathlib import Path

import pytest

from training.data import write_redacted_demo as writer
from training.data.cleaning import clean_conversations
from training.data.ingest import load_jsonl
from training.data.redact_conversations import (
    Location,
    redact_conversation_copy,
    redact_conversations,
)
from training.data.redaction import PLACEHOLDERS
from training.data.write_redacted_demo import (
    EXCLUDED_OUT,
    REDACTED_OUT,
    REPORT_OUT,
    build_outputs,
)

ROOT = Path(__file__).resolve().parents[1]
DEMO_FILE = ROOT / "data" / "demo" / "conversations.jsonl"
MANIFEST_FILE = ROOT / "data" / "demo" / "planted_issues.json"

PII_ISSUE_TO_KIND = {
    "pii_email": "EMAIL",
    "pii_phone": "PHONE",
    "pii_name": "NAME",
    "pii_address": "ADDRESS",
    "pii_order": "ORDER_REF",
}

EXPECTED_EXCLUDED_IDS = ["conv-0050", "conv-0051", "conv-0052", "conv-0058"]
EXCLUDED_RECORD_KEYS = {
    "conversation_id",
    "customer_id",
    "exclusion_reason",
    "original_conversation_id",
    "messages",
}


def conversation(conversation_id="conv-0001", customer_id="cust-0001", messages=None):
    if messages is None:
        messages = [("customer", "Hi"), ("agent", "Hello")]
    return {
        "conversation_id": conversation_id,
        "customer_id": customer_id,
        "messages": [{"role": role, "text": text} for role, text in messages],
    }


def parse_lines(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines()]


def core(record: dict) -> tuple:
    return (record["conversation_id"], record["customer_id"], record["messages"])


@pytest.fixture(scope="module")
def demo():
    loaded = load_jsonl(DEMO_FILE)
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    cleaned = clean_conversations(loaded.conversations)
    kept_count = len(cleaned.conversations)
    # Kept conversations come first, then excluded ones, in input order.
    everything = redact_conversations(cleaned.conversations + cleaned.excluded)
    planted = [issue for issue in manifest["planted_issues"] if issue["issue"] in PII_ISSUE_TO_KIND]
    return cleaned, everything, kept_count, planted


# Structure and content


def test_ids_roles_and_order_are_preserved():
    original = [conversation(messages=[
        ("customer", "Call 0712 345 678"),
        ("agent", "Thanks"),
        ("customer", "Bye"),
    ])]
    result = redact_conversations(original)
    kept = result.conversations[0]
    assert kept["conversation_id"] == "conv-0001"
    assert kept["customer_id"] == "cust-0001"
    assert [m["role"] for m in kept["messages"]] == ["customer", "agent", "customer"]
    assert [m["text"] for m in kept["messages"]] == ["Call [PHONE]", "Thanks", "Bye"]


def test_non_personal_text_is_unchanged():
    original = [conversation(messages=[("customer", "Does the kettle have a lid?")])]
    assert redact_conversations(original).conversations == original


def test_input_is_not_modified():
    original = [conversation(messages=[("customer", "Email amina@example.com")])]
    snapshot = json.dumps(original, sort_keys=True)
    redact_conversations(original)
    assert json.dumps(original, sort_keys=True) == snapshot


def test_redact_conversation_copy_does_not_touch_its_argument():
    original = conversation(messages=[("customer", "Email amina@example.com")])
    snapshot = json.dumps(original, sort_keys=True)
    preview = redact_conversation_copy(original)
    assert preview["messages"][0]["text"] == "Email [EMAIL]"
    assert json.dumps(original, sort_keys=True) == snapshot


def test_empty_input_gives_empty_output():
    result = redact_conversations([])
    assert result.conversations == []
    assert result.report.conversations == 0
    assert result.report.total == 0


# Report content


def test_report_counts_and_locations_by_category():
    result = redact_conversations([
        conversation("conv-0001", messages=[
            ("customer", "Email amina@example.com"),
            ("agent", "Sure"),
        ]),
        conversation("conv-0002", messages=[
            ("customer", "Hi"),
            ("customer", "Call 0712 345 678 or 0700 000 123"),
        ]),
    ])
    report = result.report
    assert report.counts == {"EMAIL": 1, "PHONE": 2}
    assert report.locations["EMAIL"] == [Location("conv-0001", 0)]
    assert report.locations["PHONE"] == [Location("conv-0002", 1)]
    assert report.total == 3
    assert (report.conversations, report.messages) == (2, 4)


def test_report_dict_lists_every_category_even_when_zero():
    data = redact_conversations([conversation()]).report.to_dict()
    assert data["counts"] == {"ADDRESS": 0, "EMAIL": 0, "NAME": 0, "ORDER_REF": 0, "PHONE": 0}
    assert data["locations"] == {"ADDRESS": [], "EMAIL": [], "NAME": [], "ORDER_REF": [], "PHONE": []}
    assert data["total_redactions"] == 0


def test_report_never_contains_removed_values():
    secrets = ["amina@example.com", "0712 345 678", "ORD-123456", "Grace Kamau", "12 Ngong Road"]
    result = redact_conversations([conversation(messages=[
        ("customer", f"Email {secrets[0]} or call {secrets[1]}."),
        ("customer", f"{secrets[2]}, I'm {secrets[3]}, delivered to {secrets[4]}."),
    ])])
    report_text = json.dumps(result.report.to_dict())
    for secret in secrets:
        assert secret not in report_text
    assert secrets[0] not in repr(result.report)


def test_report_locations_are_in_input_order():
    result = redact_conversations([
        conversation("conv-0001", messages=[("customer", "amina@example.com")]),
        conversation("conv-0002", messages=[("customer", "grace@example.com")]),
    ])
    assert [loc.conversation_id for loc in result.report.locations["EMAIL"]] == ["conv-0001", "conv-0002"]


# Repeatability


def test_identical_inputs_give_identical_outputs():
    original = [conversation(messages=[("customer", "Call 0712 345 678, Grace Kamau")])]
    first = redact_conversations(original)
    second = redact_conversations(original)
    assert first.conversations == second.conversations
    assert first.report.to_dict() == second.report.to_dict()


def test_name_list_changes_output_in_the_expected_way():
    original = [conversation(messages=[("customer", "My name is Zawadi Mutua.")])]
    with_list = redact_conversations(original, ("Zawadi",), ("Mutua",))
    without_list = redact_conversations(original, (), ())
    assert with_list.conversations[0]["messages"][0]["text"] == "My name is [NAME]."
    assert without_list.conversations[0]["messages"][0]["text"] == "My name is Zawadi Mutua."


def test_written_files_are_byte_identical_across_runs():
    first = build_outputs()
    second = build_outputs()
    assert first == second


# Excluded conversations in memory


def test_cleaner_returns_excluded_conversations_in_normalized_form():
    result = clean_conversations([
        conversation("conv-0001", messages=[("customer", "Hello\u00a0there  "), ("agent", "Hi")]),
        conversation("conv-0002", "cust-0002", messages=[("customer", "Hello?  ")]),
    ])
    assert [c["conversation_id"] for c in result.conversations] == ["conv-0001"]
    assert [c["conversation_id"] for c in result.excluded] == ["conv-0002"]
    assert result.excluded[0]["messages"] == [{"role": "customer", "text": "Hello?"}]


def test_excluded_conversations_do_not_appear_in_kept_output():
    result = clean_conversations([
        conversation("conv-0001"),
        conversation("conv-0002", "cust-0002", messages=[("customer", "Hello?")]),
    ])
    assert [c["conversation_id"] for c in result.conversations] == ["conv-0001"]
    assert "conv-0002" not in json.dumps(result.conversations)


def test_excluded_conversations_are_redacted_and_reported():
    cleaned = clean_conversations([
        conversation("conv-0001", messages=[("customer", "Hi"), ("agent", "Hello")]),
        conversation("conv-0002", "cust-0002", messages=[
            ("customer", "Email amina@example.com about my order"),
        ]),
    ])
    result = redact_conversations(cleaned.conversations + cleaned.excluded)
    assert result.conversations[1]["messages"][0]["text"] == "Email [EMAIL] about my order"
    assert result.report.counts == {"EMAIL": 1}
    assert result.report.locations["EMAIL"] == [Location("conv-0002", 0)]


# Output files


def test_three_output_paths_are_distinct_and_named_as_agreed():
    assert len({REDACTED_OUT, EXCLUDED_OUT, REPORT_OUT}) == 3
    assert REDACTED_OUT.name == "conversations_redacted.jsonl"
    assert EXCLUDED_OUT.name == "excluded_conversations_redacted.jsonl"
    assert REPORT_OUT.name == "redaction_report.json"
    assert EXCLUDED_OUT.parent == ROOT / "data" / "demo"


def test_main_writes_three_files_matching_build_outputs(tmp_path, monkeypatch, capsys):
    retained_path = tmp_path / "retained.jsonl"
    excluded_path = tmp_path / "excluded.jsonl"
    report_path = tmp_path / "report.json"
    monkeypatch.setattr(writer, "REDACTED_OUT", retained_path)
    monkeypatch.setattr(writer, "EXCLUDED_OUT", excluded_path)
    monkeypatch.setattr(writer, "REPORT_OUT", report_path)

    writer.main()

    retained, excluded, report = build_outputs()
    assert retained_path.read_text(encoding="utf-8") == retained
    assert excluded_path.read_text(encoding="utf-8") == excluded
    assert report_path.read_text(encoding="utf-8") == report
    printed = capsys.readouterr().out.splitlines()
    assert printed == [
        f"Wrote {retained_path}",
        f"Wrote {excluded_path}",
        f"Wrote {report_path}",
    ]


def test_excluded_file_holds_exactly_the_excluded_conversations():
    _, excluded_text, _ = build_outputs()
    assert [r["conversation_id"] for r in parse_lines(excluded_text)] == EXPECTED_EXCLUDED_IDS


def test_excluded_records_carry_reason_and_original_id():
    _, excluded_text, _ = build_outputs()
    by_id = {r["conversation_id"]: r for r in parse_lines(excluded_text)}
    assert by_id["conv-0050"]["exclusion_reason"] == "incomplete_no_agent_reply"
    assert by_id["conv-0050"]["original_conversation_id"] is None
    assert by_id["conv-0058"]["exclusion_reason"] == "duplicate_content"
    assert by_id["conv-0058"]["original_conversation_id"] == "conv-0003"


def test_excluded_records_have_the_expected_keys():
    _, excluded_text, _ = build_outputs()
    for record in parse_lines(excluded_text):
        assert set(record) == EXCLUDED_RECORD_KEYS


def test_excluded_file_has_no_overlap_with_retained_file():
    retained_text, excluded_text, _ = build_outputs()
    retained_ids = {r["conversation_id"] for r in parse_lines(retained_text)}
    excluded_ids = {r["conversation_id"] for r in parse_lines(excluded_text)}
    assert len(retained_ids) == 54
    assert len(excluded_ids) == 4
    assert retained_ids.isdisjoint(excluded_ids)


def test_excluded_file_messages_are_redacted_before_writing():
    _, excluded_text, _ = build_outputs()
    record = next(r for r in parse_lines(excluded_text) if r["conversation_id"] == "conv-0052")
    text = record["messages"][0]["text"]
    assert "[ADDRESS]" in text
    assert "197 Mombasa Road" not in text


def test_report_names_the_retained_file_as_the_only_training_input():
    _, _, report_text = build_outputs()
    report = json.loads(report_text)
    assert report["training_input_file"] == "data/demo/conversations_redacted.jsonl"
    assert report["training_input_file"] != report["outputs"]["excluded_conversations"]
    assert report["excluded_file_is_training_input"] is False


def test_report_output_names_match_the_files_written():
    _, _, report_text = build_outputs()
    outputs = json.loads(report_text)["outputs"]
    assert Path(outputs["retained_conversations"]).name == REDACTED_OUT.name
    assert Path(outputs["excluded_conversations"]).name == EXCLUDED_OUT.name
    assert Path(outputs["report"]).name == REPORT_OUT.name


# Demo data: mandatory checks against the manifest


def test_redacted_demo_contains_no_planted_value(demo):
    _, everything, _, planted = demo
    text = json.dumps(everything.conversations, ensure_ascii=False)
    leaked = [issue for issue in planted if issue["value"] in text]
    assert not leaked, f"planted values still present: {leaked}"


def test_report_contains_no_planted_value(demo):
    """Mandatory: the build fails if any planted PII value appears in the report."""
    _, everything, _, planted = demo
    report_text = json.dumps(everything.report.to_dict(), ensure_ascii=False)
    leaked = [issue for issue in planted if issue["value"] in report_text]
    assert not leaked, f"planted values found in the report: {leaked}"


def test_written_report_file_contains_no_planted_value(demo):
    _, _, _, planted = demo
    _, _, report_text = build_outputs()
    leaked = [issue for issue in planted if issue["value"] in report_text]
    assert not leaked, f"planted values found in the report file: {leaked}"


def test_retained_file_contains_no_planted_value(demo):
    _, _, _, planted = demo
    retained_text, _, _ = build_outputs()
    leaked = [issue for issue in planted if issue["value"] in retained_text]
    assert not leaked, f"planted values found in the retained file: {leaked}"


def test_excluded_file_contains_no_planted_value(demo):
    _, _, _, planted = demo
    _, excluded_text, _ = build_outputs()
    leaked = [issue for issue in planted if issue["value"] in excluded_text]
    assert not leaked, f"planted values found in the excluded file: {leaked}"


def test_redaction_counts_match_manifest_by_category(demo):
    _, everything, _, planted = demo
    expected: dict[str, int] = {kind: 0 for kind in PLACEHOLDERS}
    for issue in planted:
        expected[PII_ISSUE_TO_KIND[issue["issue"]]] += 1
    assert everything.report.counts == {k: v for k, v in expected.items() if v}
    assert everything.report.to_dict()["counts"] == expected


def test_each_planted_value_has_a_location_in_the_report(demo):
    _, everything, _, planted = demo
    located = {
        (loc.conversation_id, kind)
        for kind, locations in everything.report.locations.items()
        for loc in locations
    }
    for issue in planted:
        assert (issue["conversation_id"], PII_ISSUE_TO_KIND[issue["issue"]]) in located, issue


def test_demo_conversations_keep_their_ids_and_order(demo):
    cleaned, everything, kept_count, _ = demo
    kept = everything.conversations[:kept_count]
    assert [c["conversation_id"] for c in kept] == [c["conversation_id"] for c in cleaned.conversations]
    for before, after in zip(cleaned.conversations, kept):
        assert before["customer_id"] == after["customer_id"]
        assert [m["role"] for m in before["messages"]] == [m["role"] for m in after["messages"]]


def test_excluded_conversations_keep_their_ids_and_order_after_redaction(demo):
    cleaned, everything, kept_count, _ = demo
    excluded = everything.conversations[kept_count:]
    assert [c["conversation_id"] for c in excluded] == [c["conversation_id"] for c in cleaned.excluded]
    for before, after in zip(cleaned.excluded, excluded):
        assert before["customer_id"] == after["customer_id"]
        assert [m["role"] for m in before["messages"]] == [m["role"] for m in after["messages"]]


def test_training_file_contains_only_kept_conversations(demo):
    cleaned, _, _, _ = demo
    retained_text, _, _ = build_outputs()
    written_ids = [r["conversation_id"] for r in parse_lines(retained_text)]
    assert written_ids == [c["conversation_id"] for c in cleaned.conversations]
    assert len(written_ids) == 54
    assert set(written_ids).isdisjoint(EXPECTED_EXCLUDED_IDS)


def test_report_lists_excluded_conversation_ids(demo):
    cleaned, _, _, _ = demo
    _, _, report_text = build_outputs()
    report = json.loads(report_text)
    assert report["excluded_conversation_ids"] == EXPECTED_EXCLUDED_IDS
    assert [c["conversation_id"] for c in cleaned.excluded] == EXPECTED_EXCLUDED_IDS
    assert report["excluded_conversations"] == 4
    assert report["cleaned_conversations"] == 54
    assert report["conversations"] == 58


def test_excluded_file_matches_in_memory_redaction(demo):
    _, everything, kept_count, _ = demo
    retained_text, excluded_text, _ = build_outputs()
    assert [core(r) for r in parse_lines(retained_text)] == [
        core(c) for c in everything.conversations[:kept_count]
    ]
    assert [core(r) for r in parse_lines(excluded_text)] == [
        core(c) for c in everything.conversations[kept_count:]
    ]


def test_cleaned_input_is_not_modified_by_redaction(demo):
    cleaned, _, _, _ = demo
    before = json.dumps(cleaned.conversations, sort_keys=True)
    redact_conversations(cleaned.conversations)
    assert json.dumps(cleaned.conversations, sort_keys=True) == before


def test_source_file_is_not_modified():
    before = DEMO_FILE.read_bytes()
    build_outputs()
    assert DEMO_FILE.read_bytes() == before


def test_demo_outputs_match_a_fresh_build():
    """The committed demo outputs must exist and match what the writer produces today."""
    retained_text, excluded_text, report_text = build_outputs()
    assert REDACTED_OUT.exists() and EXCLUDED_OUT.exists() and REPORT_OUT.exists()
    assert REDACTED_OUT.read_text(encoding="utf-8") == retained_text
    assert EXCLUDED_OUT.read_text(encoding="utf-8") == excluded_text
    assert REPORT_OUT.read_text(encoding="utf-8") == report_text

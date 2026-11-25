import copy
import json
from pathlib import Path

import pytest

from training.data.cleaning import (
    ACTION_COLLAPSED_DUPLICATE,
    ACTION_REMOVED_EMPTY,
    CLEANING_VERSION,
    REASON_DUPLICATE_CONTENT,
    REASON_DUPLICATE_ID,
    REASON_INCOMPLETE_NO_AGENT,
    REASON_INCOMPLETE_NO_CUSTOMER,
    REASON_NO_MESSAGES,
    MessageAction,
    clean_conversations,
    normalize_text,
)
from training.data.ingest import load_jsonl

ROOT = Path(__file__).resolve().parents[1]
DEMO_FILE = ROOT / "data" / "demo" / "conversations.jsonl"
MANIFEST_FILE = ROOT / "data" / "demo" / "planted_issues.json"


def conversation(conversation_id="conv-0001", customer_id="cust-0001", messages=None):
    if messages is None:
        messages = [
            ("customer", "Does the kettle have a lid?"),
            ("agent", "Yes, it does."),
        ]
    return {
        "conversation_id": conversation_id,
        "customer_id": customer_id,
        "messages": [{"role": role, "text": text} for role, text in messages],
    }


def ids_of(conversations):
    return [c["conversation_id"] for c in conversations]


def exclusions_of(result):
    return [(e.conversation_id, e.reason, e.original_conversation_id) for e in result.report.exclusions]


# Text normalization


@pytest.mark.parametrize("raw, expected", [
    ("  Hello   there \t friend  ", "Hello there friend"),
    ("Hello\u00a0there", "Hello there"),
    ("Line one  \nLine two", "Line one\nLine two"),
    ("Line one\r\nLine two\rLine three", "Line one\nLine two\nLine three"),
    ("Line one\n\n\n\nLine two", "Line one\n\nLine two"),
    ("\ufeffHello", "Hello"),
    ("co\u00adoperate", "cooperate"),
    ("Cafe\u0301", "Caf\u00e9"),
    ("Thanks \U0001F64F!", "Thanks \U0001F64F!"),
    ("Nothing to change.", "Nothing to change."),
])
def test_normalize_text(raw, expected):
    assert normalize_text(raw) == expected


@pytest.mark.parametrize("raw", [
    "  Hello   there ",
    "co\u00adoperate",
    "Line one\r\n\r\n\r\n\r\nNext",
    "Cafe\u0301 \u00a0ok",
])
def test_normalize_text_is_idempotent(raw):
    once = normalize_text(raw)
    assert normalize_text(once) == once


def test_normalize_text_keeps_case_punctuation_and_symbols():
    text = "Is order #A-42 late? I'm UPSET!!"
    assert normalize_text(text) == text


# Message-level rules


def test_empty_after_normalization_is_removed_and_recorded():
    result = clean_conversations([conversation(messages=[
        ("customer", "Hello"),
        ("customer", "\ufeff"),
        ("agent", "Hi there"),
    ])])
    assert [m["text"] for m in result.conversations[0]["messages"]] == ["Hello", "Hi there"]
    assert result.report.message_actions == [
        MessageAction("conv-0001", 1, ACTION_REMOVED_EMPTY),
    ]


def test_consecutive_identical_messages_from_same_role_are_collapsed():
    result = clean_conversations([conversation(messages=[
        ("customer", "Hi"),
        ("customer", "Hi"),
        ("agent", "Hello"),
    ])])
    assert result.conversations[0]["messages"] == [
        {"role": "customer", "text": "Hi"},
        {"role": "agent", "text": "Hello"},
    ]
    assert result.report.message_actions == [
        MessageAction("conv-0001", 1, ACTION_COLLAPSED_DUPLICATE),
    ]


def test_repeated_message_that_is_not_consecutive_is_kept():
    result = clean_conversations([conversation(messages=[
        ("customer", "Hi"),
        ("agent", "Hello"),
        ("customer", "Hi"),
    ])])
    assert len(result.conversations[0]["messages"]) == 3
    assert result.report.message_actions == []


def test_same_text_from_different_roles_is_kept():
    result = clean_conversations([conversation(messages=[
        ("customer", "Yes"),
        ("agent", "Yes"),
    ])])
    assert len(result.conversations[0]["messages"]) == 2
    assert result.report.message_actions == []


def test_collapse_applies_after_an_empty_message_is_removed():
    result = clean_conversations([conversation(messages=[
        ("customer", "Hi"),
        ("customer", "\ufeff"),
        ("customer", "Hi"),
        ("agent", "Hello"),
    ])])
    assert result.conversations[0]["messages"] == [
        {"role": "customer", "text": "Hi"},
        {"role": "agent", "text": "Hello"},
    ]
    assert result.report.message_actions == [
        MessageAction("conv-0001", 1, ACTION_REMOVED_EMPTY),
        MessageAction("conv-0001", 2, ACTION_COLLAPSED_DUPLICATE),
    ]


def test_messages_that_differ_only_in_spacing_are_collapsed():
    result = clean_conversations([conversation(messages=[
        ("customer", "Hi there"),
        ("customer", "Hi\u00a0there "),
        ("agent", "Hello!"),
    ])])
    assert result.conversations[0]["messages"] == [
        {"role": "customer", "text": "Hi there"},
        {"role": "agent", "text": "Hello!"},
    ]
    assert result.report.message_actions == [
        MessageAction("conv-0001", 1, ACTION_COLLAPSED_DUPLICATE),
    ]


# Conversation-level rules


def test_complete_conversation_is_kept_unchanged():
    result = clean_conversations([conversation()])
    assert result.conversations == [conversation()]
    assert result.report.exclusions == []


def test_customer_only_conversation_is_flagged_incomplete():
    result = clean_conversations([conversation(messages=[("customer", "Hello?")])])
    assert result.conversations == []
    assert exclusions_of(result) == [("conv-0001", REASON_INCOMPLETE_NO_AGENT, None)]


def test_agent_only_conversation_is_flagged_incomplete():
    result = clean_conversations([conversation(messages=[("agent", "Hello, how can I help?")])])
    assert result.conversations == []
    assert exclusions_of(result) == [("conv-0001", REASON_INCOMPLETE_NO_CUSTOMER, None)]


def test_conversation_with_no_messages_after_cleaning_is_excluded():
    result = clean_conversations([conversation(messages=[
        ("customer", "\ufeff"),
        ("agent", "\u00ad"),
    ])])
    assert result.conversations == []
    assert exclusions_of(result) == [("conv-0001", REASON_NO_MESSAGES, None)]


# Duplicates: the first occurrence is kept


def test_later_copy_with_same_customer_and_messages_is_excluded():
    original = conversation("conv-0003")
    copy_of_it = conversation("conv-0058")
    result = clean_conversations([original, copy_of_it])
    assert ids_of(result.conversations) == ["conv-0003"]
    assert exclusions_of(result) == [("conv-0058", REASON_DUPLICATE_CONTENT, "conv-0003")]


def test_duplicate_detection_uses_normalized_text():
    first = conversation("conv-0001", messages=[
        ("customer", "Does the kettle have a lid?"),
        ("agent", "Yes, it does."),
    ])
    second = conversation("conv-0002", messages=[
        ("customer", "Does the\u00a0kettle have a lid?  "),
        ("agent", "Yes, it does."),
    ])
    result = clean_conversations([first, second])
    assert ids_of(result.conversations) == ["conv-0001"]
    assert exclusions_of(result) == [("conv-0002", REASON_DUPLICATE_CONTENT, "conv-0001")]


def test_same_messages_from_different_customers_are_not_duplicates():
    result = clean_conversations([
        conversation("conv-0001", "cust-0001"),
        conversation("conv-0002", "cust-0002"),
    ])
    assert ids_of(result.conversations) == ["conv-0001", "conv-0002"]
    assert result.report.exclusions == []


def test_reused_conversation_id_with_different_content_is_excluded():
    first = conversation("conv-0001", "cust-0001", messages=[
        ("customer", "First question"),
        ("agent", "First answer"),
    ])
    second = conversation("conv-0001", "cust-0002", messages=[
        ("customer", "Other question"),
        ("agent", "Other answer"),
    ])
    result = clean_conversations([first, second])
    assert ids_of(result.conversations) == ["conv-0001"]
    assert result.conversations[0]["customer_id"] == "cust-0001"
    assert exclusions_of(result) == [("conv-0001", REASON_DUPLICATE_ID, "conv-0001")]


def test_incomplete_copy_is_reported_as_duplicate_not_incomplete():
    first = conversation("conv-0001", messages=[("customer", "Hello?")])
    second = conversation("conv-0002", messages=[("customer", "Hello?")])
    result = clean_conversations([first, second])
    assert result.conversations == []
    assert exclusions_of(result) == [
        ("conv-0001", REASON_INCOMPLETE_NO_AGENT, None),
        ("conv-0002", REASON_DUPLICATE_CONTENT, "conv-0001"),
    ]


# Ordering, counts, and safety


def test_kept_conversations_keep_input_order():
    result = clean_conversations([
        conversation("conv-0003", "cust-0003"),
        conversation("conv-0001", "cust-0001"),
        conversation("conv-0002", "cust-0002"),
    ])
    assert ids_of(result.conversations) == ["conv-0003", "conv-0001", "conv-0002"]


def test_counts_before_and_after_cleaning():
    result = clean_conversations([
        conversation("conv-0001", "cust-0001"),
        conversation("conv-0002", "cust-0002", messages=[("customer", "Hello?")]),
        conversation("conv-0003", "cust-0001"),
        conversation("conv-0004", "cust-0004", messages=[
            ("customer", "Hi"),
            ("customer", "Hi"),
            ("agent", "Hello"),
        ]),
        conversation("conv-0005", "cust-0005", messages=[("customer", "\ufeff")]),
    ])
    report = result.report
    assert (report.input_conversations, report.input_messages) == (5, 9)
    assert (report.output_conversations, report.output_messages) == (2, 4)
    assert report.counts_by_reason() == {
        REASON_INCOMPLETE_NO_AGENT: 1,
        REASON_DUPLICATE_CONTENT: 1,
        REASON_NO_MESSAGES: 1,
    }
    assert report.input_conversations == report.output_conversations + len(report.exclusions)
    summary = report.to_dict()
    assert summary["messages_removed_empty"] == 1
    assert summary["messages_collapsed_duplicate"] == 1


def test_cleaning_does_not_modify_its_input():
    original = [
        conversation("conv-0001", messages=[
            ("customer", "Hi"),
            ("customer", "Hi"),
            ("agent", "\ufeffHello"),
        ]),
        conversation("conv-0002", messages=[("customer", "Hello?")]),
    ]
    snapshot = copy.deepcopy(original)
    clean_conversations(original)
    assert original == snapshot


def test_cleaning_is_deterministic():
    original = [
        conversation("conv-0001", "cust-0001"),
        conversation("conv-0002", "cust-0001"),
        conversation("conv-0003", messages=[("customer", "Hello?")]),
    ]
    first = clean_conversations(original)
    second = clean_conversations(original)
    assert first.report.to_dict() == second.report.to_dict()
    assert first.conversations == second.conversations


def test_report_serializes_to_json():
    result = clean_conversations([conversation(messages=[("customer", "Hi")])])
    data = json.loads(json.dumps(result.report.to_dict()))
    assert data["cleaning_version"] == CLEANING_VERSION
    assert data["before"] == {"conversations": 1, "messages": 1}
    assert data["after"] == {"conversations": 0, "messages": 0}
    assert data["excluded_by_reason"] == {REASON_INCOMPLETE_NO_AGENT: 1}
    assert data["exclusions"][0]["conversation_id"] == "conv-0001"


def test_pii_is_left_for_the_redaction_step():
    result = clean_conversations([conversation(messages=[
        ("customer", "My email is jane@example.com."),
        ("agent", "Thanks, Jane."),
    ])])
    assert result.conversations[0]["messages"][0]["text"] == "My email is jane@example.com."


# Validation against the synthetic demo and its manifest


@pytest.fixture(scope="module")
def demo():
    loaded = load_jsonl(DEMO_FILE)
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    cleaned = clean_conversations(loaded.conversations)
    return loaded, manifest, cleaned


def planted(manifest, issue_type):
    return [issue for issue in manifest["planted_issues"] if issue["issue"] == issue_type]


def test_loader_rejections_match_manifest(demo):
    loaded, manifest, _ = demo
    loader_issues = ["invalid_role", "empty_customer_message", "missing_messages_field", "invalid_json"]
    planted_count = sum(len(planted(manifest, issue)) for issue in loader_issues)
    assert planted_count == len(loaded.errors) == 4
    # Record N sits on line N of the demo file, so invalid_role (conv-0021) is line 21.
    assert sorted(e.line_number for e in loaded.errors) == [21, 55, 61, 62]
    assert loaded.rows_accepted == 58


def test_incomplete_flags_match_manifest(demo):
    _, manifest, cleaned = demo
    expected = {issue["conversation_id"] for issue in planted(manifest, "incomplete_no_agent_reply")}
    flagged = {
        e.conversation_id for e in cleaned.report.exclusions
        if e.reason == REASON_INCOMPLETE_NO_AGENT
    }
    assert flagged == expected == {"conv-0050", "conv-0051", "conv-0052"}


def test_duplicates_match_manifest(demo):
    _, manifest, cleaned = demo
    expected = {
        issue["conversation_id"]: issue["duplicate_of"]
        for issue in planted(manifest, "duplicate_of")
    }
    flagged = {
        e.conversation_id: e.original_conversation_id
        for e in cleaned.report.exclusions
        if e.reason in (REASON_DUPLICATE_CONTENT, REASON_DUPLICATE_ID)
    }
    assert flagged == expected, f"cleaner flagged {flagged}, manifest plants {expected}"


def test_cleaned_output_excludes_planted_problem_records(demo):
    _, manifest, cleaned = demo
    kept = set(ids_of(cleaned.conversations))
    problem_ids = {
        issue["conversation_id"] for issue in manifest["planted_issues"]
        if issue["issue"] in ("incomplete_no_agent_reply", "duplicate_of")
    }
    assert kept.isdisjoint(problem_ids)
    assert "conv-0003" in kept


def test_planted_pii_is_left_for_the_redaction_step(demo):
    _, manifest, cleaned = demo
    kept = {c["conversation_id"]: c for c in cleaned.conversations}
    checked = 0
    for issue in manifest["planted_issues"]:
        if issue["issue"].startswith("pii_") and issue["conversation_id"] in kept:
            text = " ".join(m["text"] for m in kept[issue["conversation_id"]]["messages"])
            assert issue["value"] in text, issue
            checked += 1
    assert checked > 0


def test_demo_counts_are_consistent(demo):
    loaded, _, cleaned = demo
    report = cleaned.report
    assert report.input_conversations == len(loaded.conversations) == 58
    assert report.input_messages == 113
    assert report.input_conversations == report.output_conversations + len(report.exclusions)
    assert report.output_messages == sum(len(c["messages"]) for c in cleaned.conversations)


def test_demo_has_no_message_level_changes(demo):
    _, _, cleaned = demo
    assert cleaned.report.message_actions == []
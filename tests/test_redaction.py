"""Tests for the redaction patterns in training/data/redaction.py.

The demo checks use the synthetic demo file and its manifest. They confirm that
every planted value is replaced, that nothing else changes, and that the counts
match the manifest.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from training.data.generate_synthetic import FIRST_NAMES, LAST_NAMES
from training.data.ingest import load_jsonl
from training.data.redaction import (
    DEMO_FIRST_NAMES,
    DEMO_LAST_NAMES,
    PLACEHOLDERS,
    redact_text,
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

MIXED = (
    "I'm Amina Otieno. Reach me on 0712 345 678 or amina@example.com. "
    "My order is ORD-123456, delivered to 12 Ngong Road, Nairobi."
)
MIXED_REDACTED = (
    "I'm [NAME]. Reach me on [PHONE] or [EMAIL]. "
    "My order is [ORDER_REF], delivered to [ADDRESS]."
)


def redacted(text: str) -> str:
    return redact_text(text).text


# Placeholders


def test_placeholders_are_exactly_the_agreed_strings():
    assert PLACEHOLDERS == {
        "EMAIL": "[EMAIL]",
        "PHONE": "[PHONE]",
        "NAME": "[NAME]",
        "ADDRESS": "[ADDRESS]",
        "ORDER_REF": "[ORDER_REF]",
    }


# Email


@pytest.mark.parametrize("address", [
    "jane.doe@example.com",
    "anne_99+shop@mail.example.co.ke",
    "AMINA@EXAMPLE.COM",
])
def test_email_is_replaced(address):
    assert redacted(f"Write to {address} please.") == "Write to [EMAIL] please."


def test_email_with_digits_in_local_part_is_replaced_whole():
    # The local part looks like a Kenyan number. The longer email match must win.
    assert redacted("Write to 0712345678@example.com please.") == "Write to [EMAIL] please."


@pytest.mark.parametrize("sentence", [
    "Follow @DemoBrand for offers.",
    "Email us through the website contact form.",
])
def test_text_without_an_email_is_unchanged(sentence):
    assert redacted(sentence) == sentence


# Phone numbers


@pytest.mark.parametrize("number", [
    "0712345678",
    "0712 345 678",
    "0700-000-123",
    "0112345678",
    "011 234 5678",
    "+254712345678",
    "+254 712 345 678",
    "+254-712-345-678",
    "+254112345678",
    "+254 112 345 678",
    "+254 0712 345 678",
    "+254 700 000 123",
])
def test_kenyan_phone_formats_are_replaced(number):
    assert redacted(f"Call me on {number} tomorrow.") == "Call me on [PHONE] tomorrow."


def test_mpesa_number_is_treated_as_phone():
    assert redacted("Send the refund to my M-Pesa number 0712 345 678.") == (
        "Send the refund to my M-Pesa number [PHONE]."
    )


@pytest.mark.parametrize("sentence", [
    "Only 0712 345 is shown on the label.",
    "Model 12345678 is out of stock.",
    "It costs KES 1,200 with free delivery.",
])
def test_phone_like_text_that_is_not_a_phone_is_unchanged(sentence):
    assert redacted(sentence) == sentence


# Order references


@pytest.mark.parametrize("ref", ["ORD-123456", "ord-987654", "ORD-1234567890"])
def test_order_reference_is_replaced(ref):
    assert redacted(f"My order reference is {ref}.") == "My order reference is [ORDER_REF]."


def test_order_reference_wins_over_phone_like_digits():
    assert redacted("Ref ORD-0712345678.") == "Ref [ORDER_REF]."


@pytest.mark.parametrize("sentence", [
    "The word-123456 pattern is unrelated.",
    "Order ORD-12 is too short to match.",
    "Model KT-1000 is in stock.",
])
def test_order_like_text_that_is_not_a_reference_is_unchanged(sentence):
    assert redacted(sentence) == sentence


# Names


@pytest.mark.parametrize("sentence, expected", [
    ("My name is Amina Otieno.", "My name is [NAME]."),
    ("Thanks, Grace Kamau.", "Thanks, [NAME]."),
    ("I'm Brian, and I need help.", "I'm [NAME], and I need help."),
    ("I\u2019m Irene.", "I\u2019m [NAME]."),
    ("My name is Otieno.", "My name is [NAME]."),
    ("This is Esther from support.", "This is [NAME] from support."),
])
def test_known_names_are_replaced(sentence, expected):
    assert redacted(sentence) == expected


@pytest.mark.parametrize("sentence", [
    "Grace period is 30 days.",
    "John Deere tractor parts are in stock.",
    "I'm grateful for the help.",
    "The Kiptoo Road is closed this week.",
    "Do you deliver to Kisumu?",
    "Ask Amina Otienoo for the file.",
])
def test_names_inside_ordinary_text_are_unchanged(sentence):
    assert redacted(sentence) == sentence


# Addresses


@pytest.mark.parametrize("address", [
    "12 Riverside Drive, Nairobi",
    "45 Ngong Road",
    "200 Mombasa Road, Kisumu",
    "7 Kenyatta Avenue, Nakuru",
])
def test_street_address_is_replaced(address):
    assert redacted(f"Please deliver to {address}.") == "Please deliver to [ADDRESS]."


@pytest.mark.parametrize("sentence", [
    "Ngong Road is closed this week.",
    "We have 2 Road bikes in stock.",
    "Delivery to Kisumu takes two days.",
])
def test_address_like_text_that_is_not_an_address_is_unchanged(sentence):
    assert redacted(sentence) == sentence


# Ordinary product text must not be touched


@pytest.mark.parametrize("sentence", [
    "Does the 1-litre steel bottle come with a lid?",
    "Model KT-1000 costs KES 2,500.",
    "Is the insulated lunch box dishwasher safe?",
    "Our store opens at 8 am on weekdays.",
])
def test_ordinary_product_text_is_unchanged(sentence):
    assert redacted(sentence) == sentence


# Combined behaviour


def test_several_kinds_in_one_message_are_each_replaced():
    result = redact_text(MIXED)
    assert result.text == MIXED_REDACTED
    assert result.counts == {"EMAIL": 1, "PHONE": 1, "NAME": 1, "ADDRESS": 1, "ORDER_REF": 1}


def test_redaction_is_idempotent():
    once = redact_text(MIXED).text
    again = redact_text(once)
    assert again.text == once
    assert again.total == 0


def test_counts_are_zero_for_every_kind_when_nothing_is_found():
    assert redact_text("Does the kettle have a lid?").counts == {kind: 0 for kind in PLACEHOLDERS}


def test_result_holds_counts_not_removed_values():
    result = redact_text("Email amina@example.com or call 0712345678.")
    assert "amina@example.com" not in repr(result)
    assert "0712345678" not in repr(result)
    assert result.counts["EMAIL"] == 1
    assert result.counts["PHONE"] == 1


def test_name_list_can_be_supplied_for_another_brand():
    result = redact_text(
        "My name is Zawadi Mutua.",
        first_names=("Zawadi",),
        last_names=("Mutua",),
    )
    assert result.text == "My name is [NAME]."


def test_default_list_only_redacts_names_it_contains():
    # Documents the known-list limit: a name outside the demo list is not redacted.
    assert redacted("My name is Zawadi Mutua.") == "My name is Zawadi Mutua."


def test_empty_name_lists_disable_name_matching():
    assert redact_text("Amina Otieno", first_names=(), last_names=()).text == "Amina Otieno"


# Demo data: planted personal data, checked at text level


@pytest.fixture(scope="module")
def demo():
    conversations = load_jsonl(DEMO_FILE).conversations
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    return conversations, manifest


def planted_pii(manifest):
    return [issue for issue in manifest["planted_issues"] if issue["issue"] in PII_ISSUE_TO_KIND]


def test_demo_name_list_covers_every_name_the_generator_uses():
    assert set(FIRST_NAMES) <= set(DEMO_FIRST_NAMES)
    assert set(LAST_NAMES) <= set(DEMO_LAST_NAMES)


def test_every_planted_pii_value_is_in_the_demo_file(demo):
    conversations, manifest = demo
    issues = planted_pii(manifest)
    assert issues, "the manifest should record planted PII"
    by_id = {c["conversation_id"]: c for c in conversations}
    for issue in issues:
        assert issue["conversation_id"] in by_id, issue
        messages = by_id[issue["conversation_id"]]["messages"]
        assert any(issue["value"] in m["text"] for m in messages), issue


def test_planted_values_are_replaced_and_nothing_else_changes(demo):
    conversations, manifest = demo
    issues_by_conv = defaultdict(list)
    for issue in planted_pii(manifest):
        issues_by_conv[issue["conversation_id"]].append(issue)

    for conversation in conversations:
        conv_id = conversation["conversation_id"]
        for message in conversation["messages"]:
            expected = message["text"]
            for issue in issues_by_conv.get(conv_id, []):
                placeholder = PLACEHOLDERS[PII_ISSUE_TO_KIND[issue["issue"]]]
                expected = expected.replace(issue["value"], placeholder)
            assert redact_text(message["text"]).text == expected, conv_id


def test_no_planted_value_survives_redaction(demo):
    conversations, manifest = demo
    messages_by_conv = {c["conversation_id"]: c["messages"] for c in conversations}
    for issue in planted_pii(manifest):
        for message in messages_by_conv[issue["conversation_id"]]:
            assert issue["value"] not in redacted(message["text"]), issue


def test_demo_redaction_counts_match_the_manifest_by_kind(demo):
    conversations, manifest = demo
    actual = Counter()
    for conversation in conversations:
        for message in conversation["messages"]:
            actual.update(redact_text(message["text"]).counts)
    expected = Counter(PII_ISSUE_TO_KIND[issue["issue"]] for issue in planted_pii(manifest))
    assert {k: actual[k] for k in PLACEHOLDERS} == {k: expected[k] for k in PLACEHOLDERS}
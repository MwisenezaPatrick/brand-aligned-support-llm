"""Tests for the customer-level training, validation, and test split.

The split step must read only data/demo/conversations_redacted.jsonl. The
excluded conversations file must never be opened by the split step. A spy on
file opens checks that. The other tests check the written outputs against the
retained file, read directly.
"""

import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import pytest

from training.data import write_split_demo as writer
from training.data.ingest import load_jsonl
from training.data.schema import validate_conversation
from training.data.split import (
    DEFAULT_RATIOS,
    DEFAULT_SEED,
    SPLIT_NAMES,
    assign_customers,
    split_by_customer,
)

ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "data" / "demo"
RETAINED_FILE = DEMO_DIR / "conversations_redacted.jsonl"
EXCLUDED_FILE = DEMO_DIR / "excluded_conversations_redacted.jsonl"
EXPECTED_EXCLUDED_IDS = {"conv-0050", "conv-0051", "conv-0052", "conv-0058"}
CANONICAL_KEYS = {"conversation_id", "customer_id", "messages"}
MESSAGE_KEYS = {"role", "text"}


def conversation(conversation_id="conv-0001", customer_id="cust-0001"):
    return {
        "conversation_id": conversation_id,
        "customer_id": customer_id,
        "messages": [
            {"role": "customer", "text": "Hi"},
            {"role": "agent", "text": "Hello"},
        ],
    }


def synthetic(count):
    return [conversation(f"conv-{i:04d}", f"cust-{i:04d}") for i in range(count)]


def parse_text(text):
    return [json.loads(line) for line in text.splitlines()]


@pytest.fixture(scope="module")
def retained():
    loaded = load_jsonl(RETAINED_FILE)
    assert loaded.errors == []
    return loaded.conversations


@pytest.fixture(scope="module")
def demo_result(retained):
    return split_by_customer(retained)


@pytest.fixture(scope="module")
def outputs():
    return writer.build_outputs()


# Approved settings


def test_approved_ratios_seed_and_names():
    assert DEFAULT_RATIOS == (0.70, 0.15, 0.15)
    assert DEFAULT_SEED == 42
    assert SPLIT_NAMES == ("training", "validation", "test")
    assert writer.FILE_NAMES == {
        "training": "training.jsonl",
        "validation": "validation.jsonl",
        "test": "test.jsonl",
    }


# Verification items 1 to 6 on the written outputs


def test_every_customer_appears_in_exactly_one_split(outputs, retained):
    texts, _ = outputs
    customers_by_split = {
        name: {record["customer_id"] for record in parse_text(texts[name])}
        for name in SPLIT_NAMES
    }
    everyone = {record["customer_id"] for record in retained}
    assert set().union(*customers_by_split.values()) == everyone
    for first, second in combinations(SPLIT_NAMES, 2):
        assert customers_by_split[first].isdisjoint(customers_by_split[second]), (first, second)


def test_each_multi_conversation_customer_stays_in_one_split(outputs, retained):
    """Verification item 2, applied to every customer that has more than one conversation."""
    texts, _ = outputs
    split_of = {
        record["conversation_id"]: name
        for name in SPLIT_NAMES
        for record in parse_text(texts[name])
    }
    conversations_by_customer: dict[str, list[str]] = {}
    for record in retained:
        conversations_by_customer.setdefault(record["customer_id"], []).append(record["conversation_id"])

    multi = {customer: ids for customer, ids in conversations_by_customer.items() if len(ids) > 1}
    assert multi, "the demo data should contain customers with several conversations"
    for customer, ids in multi.items():
        splits = {split_of[cid] for cid in ids}
        assert len(splits) == 1, (customer, {cid: split_of[cid] for cid in ids})


def test_cust_0003_has_its_conversations_in_one_split(outputs, retained):
    texts, _ = outputs
    split_of = {
        record["conversation_id"]: name
        for name in SPLIT_NAMES
        for record in parse_text(texts[name])
    }
    owned = [record["conversation_id"] for record in retained if record["customer_id"] == "cust-0003"]
    assert "conv-0036" in owned
    assert len({split_of[cid] for cid in owned}) == 1


def test_conv_0003_and_conv_0036_belong_to_different_customers(retained):
    """Documents the demo data: the pair named in review is not one customer's conversations."""
    customer_of = {record["conversation_id"]: record["customer_id"] for record in retained}
    assert customer_of["conv-0003"] == "cust-0005"
    assert customer_of["conv-0036"] == "cust-0003"


def test_no_conversation_appears_twice(outputs):
    texts, _ = outputs
    ids = [record["conversation_id"] for name in SPLIT_NAMES for record in parse_text(texts[name])]
    assert [cid for cid, count in Counter(ids).items() if count > 1] == []


def test_all_retained_conversations_appear_exactly_once(outputs, retained):
    texts, _ = outputs
    assert len(retained) == 54
    written = sorted(
        record["conversation_id"] for name in SPLIT_NAMES for record in parse_text(texts[name])
    )
    assert written == sorted(record["conversation_id"] for record in retained)


def test_no_excluded_conversation_appears_in_any_output(outputs):
    texts, _ = outputs
    excluded_in_file = {
        json.loads(line)["conversation_id"]
        for line in EXCLUDED_FILE.read_text(encoding="utf-8").splitlines()
    }
    assert excluded_in_file == EXPECTED_EXCLUDED_IDS
    written = {record["conversation_id"] for name in SPLIT_NAMES for record in parse_text(texts[name])}
    assert written.isdisjoint(excluded_in_file)


def test_split_step_never_opens_the_excluded_file(monkeypatch):
    records = load_jsonl(RETAINED_FILE).conversations  # loaded before the spy starts
    opened: list[Path] = []
    original_open = Path.open

    def spy(self, *args, **kwargs):
        opened.append(Path(self).resolve())
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy)

    split_by_customer(records)
    assert opened == [], "the split logic itself must not open any file"

    writer.build_outputs()
    data_files = [path for path in opened if path.is_relative_to(ROOT / "data")]
    assert EXCLUDED_FILE.resolve() not in data_files
    assert data_files and set(data_files) == {RETAINED_FILE.resolve()}


# Verification items 7 and 8


def test_repeated_runs_produce_identical_files(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    writer.main(first)
    writer.main(second)
    for filename in writer.FILE_NAMES.values():
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_outputs_preserve_canonical_schema(outputs):
    texts, _ = outputs
    for name in SPLIT_NAMES:
        for record in parse_text(texts[name]):
            assert validate_conversation(record) == [], record["conversation_id"]
            assert set(record) == CANONICAL_KEYS
            for message in record["messages"]:
                assert set(message) == MESSAGE_KEYS


# Assignment behaviour


def test_assignment_is_independent_of_input_order(retained, demo_result):
    reversed_result = split_by_customer(list(reversed(retained)))
    assert reversed_result.customer_assignment == demo_result.customer_assignment
    assert assign_customers(retained) == demo_result.customer_assignment


def test_splits_keep_input_order(retained, demo_result):
    for name in SPLIT_NAMES:
        expected = [
            record["conversation_id"]
            for record in retained
            if demo_result.customer_assignment[record["customer_id"]] == name
        ]
        assert [record["conversation_id"] for record in demo_result.splits[name]] == expected


def test_different_seed_changes_assignment():
    records = synthetic(200)
    assert assign_customers(records, seed=1) != assign_customers(records, seed=2)


def test_large_input_splits_close_to_target_ratios():
    result = split_by_customer(synthetic(1000))
    for name, target in zip(SPLIT_NAMES, DEFAULT_RATIOS):
        actual = result.stats[name].conversations / result.total_conversations
        assert abs(actual - target) <= 0.01, (name, actual)


# Reporting and non-empty splits


def test_demo_split_reports_actual_counts(retained, demo_result):
    customers_in_data = {record["customer_id"] for record in retained}
    assert demo_result.total_conversations == len(retained) == 54
    assert demo_result.total_customers == len(customers_in_data)
    assert sum(stats.customers for stats in demo_result.stats.values()) == len(customers_in_data)
    for name in SPLIT_NAMES:
        members = demo_result.splits[name]
        stats = demo_result.stats[name]
        assert stats.conversations == len(members)
        assert stats.customers == len({record["customer_id"] for record in members})
        assert stats.messages == sum(len(record["messages"]) for record in members)
        assert stats.percent_of_conversations == pytest.approx(100 * len(members) / 54)
    assert sum(stats.percent_of_conversations for stats in demo_result.stats.values()) == pytest.approx(100)


def test_demo_splits_are_all_non_empty(demo_result):
    assert all(demo_result.splits[name] for name in SPLIT_NAMES)


# Refusals


def test_too_few_customers_is_refused():
    with pytest.raises(ValueError, match="would be empty"):
        split_by_customer([conversation("conv-0001", "cust-0001")])


@pytest.mark.parametrize("ratios", [(0.5, 0.5), (0.7, 0.2, 0.2), (0.8, 0.3, -0.1), (0.5, 0.5, 0.5)])
def test_invalid_ratios_are_refused(ratios):
    with pytest.raises(ValueError):
        split_by_customer(synthetic(10), ratios=ratios)


def test_empty_input_is_refused():
    with pytest.raises(ValueError, match="no conversations"):
        split_by_customer([])


def test_duplicate_conversation_ids_are_refused():
    records = [conversation("conv-0001", "cust-0001"), conversation("conv-0001", "cust-0002")]
    with pytest.raises(ValueError, match="unique"):
        split_by_customer(records)


def test_invalid_record_is_refused():
    broken = conversation("conv-0001", "cust-0001")
    del broken["customer_id"]
    with pytest.raises(ValueError, match="fails the schema"):
        split_by_customer([broken])


def test_writer_refuses_a_source_with_loader_errors(tmp_path):
    source = tmp_path / "bad.jsonl"
    source.write_text(json.dumps(conversation()) + "\n{not json}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no splits were built"):
        writer.build_outputs(source=source)


def test_main_writes_the_three_files_and_prints_counts(tmp_path, capsys):
    result = writer.main(tmp_path)
    for filename in writer.FILE_NAMES.values():
        assert (tmp_path / filename).exists()
    printed = capsys.readouterr().out
    assert printed.count("Wrote ") == 3
    assert f"Customers: {result.total_customers}" in printed
    assert f"Conversations: {result.total_conversations}" in printed

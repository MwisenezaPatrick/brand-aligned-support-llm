"""Regression tests: duplicate detection must hold for every generator seed.

The synthetic generator plants one duplicate (conv-0058, an exact copy of
conv-0003). Apart from that copy, no conversation may be flagged as a
duplicate, whatever the random seed. These tests run the generator with many
seeds, load the output through the same JSONL parser the pipeline uses, clean
it, and compare the result with the planted issues.
"""

import io
import json
import random

import pytest

from training.data.cleaning import (
    REASON_DUPLICATE_CONTENT,
    REASON_DUPLICATE_ID,
    REASON_INCOMPLETE_NO_AGENT,
    clean_conversations,
)
from training.data.generate_synthetic import (
    DUPLICATE_ID,
    DUPLICATE_OF,
    EMPTY_MESSAGE_ID,
    INCOMPLETE_IDS,
    INVALID_ROLE_ID,
    build_records,
    content_key,
)
from training.data.ingest import parse_jsonl

SEEDS = range(25)


def conv_id(number: int) -> str:
    return f"conv-{number:04d}"


def load_for_seed(seed: int):
    records, _, _ = build_records(random.Random(seed))
    text = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    return parse_jsonl(io.StringIO(text), source=f"seed-{seed}")


def test_same_seed_gives_identical_output():
    assert build_records(random.Random(42)) == build_records(random.Random(42))


@pytest.mark.parametrize("seed", SEEDS)
def test_planted_copy_matches_its_original_for_each_seed(seed):
    records, _, _ = build_records(random.Random(seed))
    by_id = {record["conversation_id"]: record for record in records}
    assert content_key(by_id[conv_id(DUPLICATE_ID)]) == content_key(by_id[conv_id(DUPLICATE_OF)])


@pytest.mark.parametrize("seed", SEEDS)
def test_generator_creates_no_unplanted_duplicates_for_each_seed(seed):
    records, _, _ = build_records(random.Random(seed))
    first_seen: dict[tuple, str] = {}
    for record in records:
        cid = record["conversation_id"]
        if cid == conv_id(DUPLICATE_ID):
            continue  # the planted copy, checked above
        key = content_key(record)
        assert key not in first_seen, f"seed {seed}: {cid} repeats {first_seen.get(key)}"
        first_seen[key] = cid


@pytest.mark.parametrize("seed", SEEDS)
def test_cleaner_flags_only_planted_problems_for_each_seed(seed):
    loaded = load_for_seed(seed)
    assert sorted(e.line_number for e in loaded.errors) == sorted([INVALID_ROLE_ID, EMPTY_MESSAGE_ID])

    cleaned = clean_conversations(loaded.conversations)
    exclusions = {e.conversation_id: e for e in cleaned.report.exclusions}

    duplicates = {
        cid: e.original_conversation_id
        for cid, e in exclusions.items()
        if e.reason in (REASON_DUPLICATE_CONTENT, REASON_DUPLICATE_ID)
    }
    assert duplicates == {conv_id(DUPLICATE_ID): conv_id(DUPLICATE_OF)}

    incomplete = {cid for cid, e in exclusions.items() if e.reason == REASON_INCOMPLETE_NO_AGENT}
    assert incomplete == {conv_id(n) for n in INCOMPLETE_IDS}

    kept = {c["conversation_id"] for c in cleaned.conversations}
    assert conv_id(DUPLICATE_OF) in kept
    assert conv_id(DUPLICATE_ID) not in kept
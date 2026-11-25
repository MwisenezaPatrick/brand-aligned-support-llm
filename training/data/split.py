"""Split redacted conversations into training, validation, and test sets.

Input is the retained conversation list produced by the redaction step. Excluded
(duplicate and incomplete) conversations are never passed to this module, so
they cannot appear in any split.

Every conversation belonging to a customer goes into the same split, so no
customer is seen in more than one split.

Assignment is deterministic:

1. Customer IDs are sorted, then shuffled with random.Random(seed). Input order
   has no effect on the assignment.
2. Each customer, in that shuffled order, goes to the split that is furthest
   below its target share of conversations. Ties go to the earlier split in
   SPLIT_NAMES.

The default seed is DEFAULT_SEED (42). The default ratios are 70% training,
15% validation, and 15% test. Because whole customers move together, actual
shares can differ from the targets; SplitResult reports the actual counts.

Within each split, conversations keep the order they had in the input.

This module never reads or writes files. The writer script
training/data/write_split_demo.py reads the redacted retained file and writes
the three split files.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from training.data.schema import validate_conversation

SPLIT_NAMES: tuple[str, ...] = ("training", "validation", "test")
DEFAULT_RATIOS: tuple[float, float, float] = (0.70, 0.15, 0.15)
DEFAULT_SEED: int = 42


@dataclass(frozen=True)
class SplitStats:
    """Actual counts for one split."""

    customers: int
    conversations: int
    messages: int
    percent_of_conversations: float


@dataclass
class SplitResult:
    """The three splits, the customer assignment, and the actual counts."""

    seed: int
    ratios: tuple[float, ...]
    splits: dict[str, list[dict[str, Any]]]
    customer_assignment: dict[str, str]
    stats: dict[str, SplitStats]

    @property
    def total_customers(self) -> int:
        return len(self.customer_assignment)

    @property
    def total_conversations(self) -> int:
        return sum(len(conversations) for conversations in self.splits.values())


def check_ratios(ratios: Sequence[float]) -> None:
    """Raise ValueError unless ratios are three positive numbers that sum to 1."""
    if len(ratios) != len(SPLIT_NAMES):
        raise ValueError(f"ratios must have {len(SPLIT_NAMES)} values, got {len(ratios)}")
    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("every ratio must be greater than zero")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1, got {sum(ratios):.6f}")


def _canonical(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "conversation_id": record["conversation_id"],
        "customer_id": record["customer_id"],
        "messages": [{"role": m["role"], "text": m["text"]} for m in record["messages"]],
    }


def assign_customers(
    conversations: Sequence[dict[str, Any]],
    seed: int = DEFAULT_SEED,
    ratios: Sequence[float] = DEFAULT_RATIOS,
) -> dict[str, str]:
    """Map each customer_id to one split name.

    The result depends only on the customer IDs, how many conversations each
    customer has, the seed, and the ratios. It does not depend on input order.
    """
    check_ratios(ratios)
    sizes: dict[str, int] = {}
    for record in conversations:
        sizes[record["customer_id"]] = sizes.get(record["customer_id"], 0) + 1

    order = sorted(sizes)
    random.Random(seed).shuffle(order)

    total = sum(sizes.values())
    targets = {name: ratio * total for name, ratio in zip(SPLIT_NAMES, ratios)}
    counts = {name: 0 for name in SPLIT_NAMES}
    assignment: dict[str, str] = {}
    for customer_id in order:
        name = max(
            SPLIT_NAMES,
            key=lambda candidate: (targets[candidate] - counts[candidate], -SPLIT_NAMES.index(candidate)),
        )
        assignment[customer_id] = name
        counts[name] += sizes[customer_id]
    return assignment


def split_by_customer(
    conversations: Sequence[dict[str, Any]],
    seed: int = DEFAULT_SEED,
    ratios: Sequence[float] = DEFAULT_RATIOS,
) -> SplitResult:
    """Split conversations by customer. Raises ValueError if the split cannot be made."""
    check_ratios(ratios)
    if not conversations:
        raise ValueError("no conversations to split")

    for index, record in enumerate(conversations, start=1):
        errors = validate_conversation(record)
        if errors:
            raise ValueError(f"record {index} fails the schema: {errors[0]}")

    ids = [record["conversation_id"] for record in conversations]
    if len(set(ids)) != len(ids):
        raise ValueError("conversation_id values must be unique before splitting")

    assignment = assign_customers(conversations, seed=seed, ratios=ratios)
    splits: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    for record in conversations:
        splits[assignment[record["customer_id"]]].append(_canonical(record))

    empty = [name for name in SPLIT_NAMES if not splits[name]]
    if empty:
        raise ValueError(
            f"split(s) {', '.join(empty)} would be empty: there are too few customers for these ratios"
        )

    total = len(conversations)
    stats = {
        name: SplitStats(
            customers=sum(1 for target in assignment.values() if target == name),
            conversations=len(splits[name]),
            messages=sum(len(record["messages"]) for record in splits[name]),
            percent_of_conversations=100.0 * len(splits[name]) / total,
        )
        for name in SPLIT_NAMES
    }
    return SplitResult(
        seed=seed,
        ratios=tuple(ratios),
        splits=splits,
        customer_assignment=assignment,
        stats=stats,
    )

"""Write the training, validation, and test splits for the redacted demo data.

Run from the project root:
    python -m training.data.write_split_demo

Reads only data/demo/conversations_redacted.jsonl. The excluded conversations
file is never opened here. Writes data/demo/training.jsonl,
data/demo/validation.jsonl, and data/demo/test.jsonl, and prints the actual
customer and conversation counts for each split.

Real or client data must never be written to data/demo/. It belongs in
data/processed/, which Git ignores.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from training.data.ingest import load_jsonl
from training.data.split import (
    DEFAULT_RATIOS,
    DEFAULT_SEED,
    SPLIT_NAMES,
    SplitResult,
    split_by_customer,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "demo" / "conversations_redacted.jsonl"
OUT_DIR = ROOT / "data" / "demo"
FILE_NAMES: dict[str, str] = {
    "training": "training.jsonl",
    "validation": "validation.jsonl",
    "test": "test.jsonl",
}


def _jsonl(records: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records)


def build_outputs(
    source: Path = SOURCE,
    seed: int = DEFAULT_SEED,
    ratios: tuple[float, ...] = DEFAULT_RATIOS,
) -> tuple[dict[str, str], SplitResult]:
    """Return the three split file texts and the split result, without writing files."""
    loaded = load_jsonl(source)
    if loaded.errors:
        problems = "; ".join(str(error) for error in loaded.errors)
        raise RuntimeError(f"{source} has loader errors, so no splits were built: {problems}")
    result = split_by_customer(loaded.conversations, seed=seed, ratios=ratios)
    texts = {name: _jsonl(result.splits[name]) for name in SPLIT_NAMES}
    return texts, result


def summary_lines(result: SplitResult) -> list[str]:
    ratio_text = " / ".join(f"{ratio * 100:.0f}%" for ratio in result.ratios)
    lines = [
        f"Seed {result.seed}; target split {ratio_text} (training / validation / test)",
        f"Customers: {result.total_customers}   Conversations: {result.total_conversations}",
    ]
    for name in SPLIT_NAMES:
        stats = result.stats[name]
        lines.append(
            f"  {name:<11} customers {stats.customers:>3}   "
            f"conversations {stats.conversations:>3}   "
            f"{stats.percent_of_conversations:5.1f}% of conversations   "
            f"messages {stats.messages}"
        )
    return lines


def main(out_dir: Path = OUT_DIR) -> SplitResult:
    texts, result = build_outputs()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_NAMES:
        path = out_dir / FILE_NAMES[name]
        path.write_text(texts[name], encoding="utf-8", newline="\n")
        print(f"Wrote {path}")
    for line in summary_lines(result):
        print(line)
    return result


if __name__ == "__main__":
    main()

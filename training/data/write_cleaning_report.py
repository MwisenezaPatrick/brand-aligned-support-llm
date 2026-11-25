"""Write the cleaning report for the synthetic demo data to data/demo/.

Run from the project root:
    python training/data/write_cleaning_report.py

The report contains counts, exclusion reasons, and conversation IDs. It does not
contain message text, so it is safe to commit. The source file is only read.
"""

import json
from pathlib import Path

from training.data.cleaning import clean_conversations
from training.data.ingest import load_jsonl

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "demo" / "conversations.jsonl"
OUTPUT = ROOT / "data" / "demo" / "cleaning_report.json"


def main() -> None:
    loaded = load_jsonl(SOURCE)
    if loaded.errors:
        print("Loader reported errors; the report covers only the accepted records:")
        for error in loaded.errors:
            print(f"  {error}")

    report = clean_conversations(loaded.conversations).report.to_dict()
    report["source_file"] = "data/demo/conversations.jsonl"
    report["loader"] = {
        "rows_read": loaded.rows_read,
        "rows_accepted": loaded.rows_accepted,
        "rows_rejected": loaded.rows_rejected,
        "rejections": [str(error) for error in loaded.errors],
    }

    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
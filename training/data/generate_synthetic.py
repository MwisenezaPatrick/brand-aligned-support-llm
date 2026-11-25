"""Generate clearly labeled synthetic customer-support transcripts for demos and tests.

Everything here is fictional: the brand, customers, email addresses, phone numbers,
street addresses, and order references. Output goes to data/demo/, which Git tracks.
Real transcripts must never be written to that folder; they belong in data/raw/,
which .gitignore excludes.

The script plants known problems and records each one in planted_issues.json, so
later tests can check that the cleaning pipeline finds them.

Uniqueness guarantee: apart from the planted duplicate, no two generated
conversations share a customer and identical messages. When a random draw would
repeat an earlier conversation by chance, the draw is repeated. Two kinds of
record are exempt: the planted duplicate, which is a deliberate copy, and the
records the loader rejects on purpose (invalid role, empty message), which never
reach the cleaner.

Run from the project root:
    python training/data/generate_synthetic.py
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "demo"
SEED = 42
BRAND = "DemoBrand"
NUM_CONVERSATIONS = 60
MAX_REDRAWS = 100

INCOMPLETE_IDS = {50, 51, 52}
EMPTY_MESSAGE_ID = 55
DUPLICATE_ID, DUPLICATE_OF = 58, 3
INVALID_ROLE_ID = 21
PII_KINDS = ["email", "phone", "name", "address", "order"]

# Conversations exempt from the uniqueness redraw (see the module docstring).
_UNCHECKED_IDS = {INVALID_ROLE_ID, EMPTY_MESSAGE_ID, DUPLICATE_ID}

CUSTOMER_IDS = [f"cust-{i:04d}" for i in range(1, 31)]
FIRST_NAMES = ["Amina", "Brian", "Cynthia", "David", "Esther", "Felix", "Grace", "Hassan", "Irene", "John"]
LAST_NAMES = ["Otieno", "Mwangi", "Kamau", "Wekesa", "Njeri", "Odhiambo", "Achieng", "Kiptoo"]
STREETS = ["Riverside Drive", "Ngong Road", "Mombasa Road", "Kenyatta Avenue"]
GREETINGS = ["Hi there. ", "Hello, ", "Good morning. ", ""]

# (category, customer message, approved agent reply)
TEMPLATES = [
    ("product_inquiry", "Does the 1-litre steel bottle come with a lid?",
     "Yes, the 1-litre steel bottle includes a leak-resistant lid. Is there anything else you'd like to know about it?"),
    ("product_inquiry", "Is the insulated lunch box dishwasher safe?",
     "Please hand-wash the insulated lunch box with warm water and mild soap. Let me know if you need care tips for other items."),
    ("pricing", "How much is the premium kettle right now?",
     "I'm not able to confirm live prices in this chat. I can connect you with our sales team, who can give you the current price."),
    ("pricing", "Do you offer a discount for buying two items?",
     "I don't have information about bundle discounts. I can pass your question to our sales team so they can confirm what's available."),
    ("delivery", "Where is my order? I placed it last week.",
     "I'm sorry for the wait. Could you share your order reference so I can check the status with our delivery team?"),
    ("delivery", "Can you deliver to Kisumu on Saturday?",
     "I can't confirm delivery dates without checking first. I'll ask our logistics team and follow up with you."),
    ("returns", "The lid on my flask arrived cracked. Can I return it?",
     "I'm sorry the flask arrived damaged. Please send a photo of the damage and your order reference so we can review your return."),
    ("returns", "I changed my mind about the bag. Can I send it back?",
     "Returns depend on the item's condition and timing. Could you tell me when you received it so I can check our return policy?"),
    ("refunds", "When will I get my refund for the cancelled order?",
     "I'm sorry for the delay. I'll check the refund status for you. Please share your order reference."),
    ("refunds", "Can I get a full refund if the product doesn't fit?",
     "I'm not able to promise a refund for that case. I'll escalate your request to our support team for a decision."),
    ("warranty", "How long is the warranty on the blender?",
     "I'd like to confirm that for you accurately. Could you tell me the model name so I can check the warranty terms?"),
    ("warranty", "Does the warranty cover accidental damage?",
     "I don't want to guess about coverage. I'll pass your question to our support team, who can review the warranty terms with you."),
    ("troubleshooting", "My kettle won't turn on after I plugged it in.",
     "Let's try a few steps. Check that the plug is firmly in the socket and the base is seated correctly. If it still doesn't turn on, let me know."),
    ("troubleshooting", "The blender makes a loud grinding noise. What should I do?",
     "Please switch it off and unplug it. Check for food stuck under the blade, then try again. If the noise continues, I can help you arrange a check."),
    ("complaints", "This is the third time my order arrived incomplete. I'm really frustrated.",
     "I'm sorry this has happened again. I understand how frustrating that is. I'm escalating this to a senior team member who will contact you."),
    ("complaints", "The staff member at your shop was rude to me.",
     "I'm sorry to hear about that experience. I've recorded your feedback and passed it to our customer care manager."),
    ("general_support", "What are your opening hours?",
     "Our support team is available during business hours. Please check our website for the current schedule."),
    ("general_support", "Can I change the email address on my account?",
     "Yes, I can help with that. Please confirm your identity with the details you registered with, and I'll guide you through the change."),
]


def make_pii(rng: random.Random, kind: str) -> tuple[str, str]:
    """Return a customer sentence containing one fake PII value, and that value."""
    first, last = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
    if kind == "email":
        value = f"{first.lower()}.{last.lower()}{rng.randint(1, 99)}@example.com"
        return f"My email is {value}.", value
    if kind == "phone":
        value = f"+254 700 000 {rng.randint(100, 999)}"
        return f"You can call me on {value}.", value
    if kind == "name":
        value = f"{first} {last}"
        return f"My name is {value}.", value
    if kind == "address":
        value = f"{rng.randint(1, 200)} {rng.choice(STREETS)}, Nairobi"
        return f"Please send the reply to {value}.", value
    value = f"ORD-{rng.randint(100000, 999999)}"
    return f"My order reference is {value}.", value


def content_key(record: dict) -> tuple:
    """Identify a conversation by its customer and its exact messages."""
    return (record["customer_id"], tuple((m["role"], m["text"]) for m in record["messages"]))


def _draw_conversation(rng: random.Random, n: int) -> tuple[dict, str, list[dict]]:
    """Draw conversation number n.

    The random calls happen in a fixed order: greeting, PII, customer. Keep that
    order, or the demo file for SEED changes.
    """
    conv_id = f"conv-{n:04d}"
    category, customer_text, agent_text = TEMPLATES[(n - 1) % len(TEMPLATES)]
    messages = [
        {"role": "customer", "text": rng.choice(GREETINGS) + customer_text},
        {"role": "agent", "text": agent_text},
    ]
    found: list[dict] = []

    if n % 4 == 0:
        kind = PII_KINDS[(n // 4) % len(PII_KINDS)]
        sentence, value = make_pii(rng, kind)
        messages[0]["text"] += " " + sentence
        found.append({"conversation_id": conv_id, "issue": f"pii_{kind}", "value": value})

    if n in INCOMPLETE_IDS:
        messages = messages[:1]
        found.append({"conversation_id": conv_id, "issue": "incomplete_no_agent_reply"})

    if n == EMPTY_MESSAGE_ID:
        messages[0]["text"] = ""
        found.append({"conversation_id": conv_id, "issue": "empty_customer_message"})

    record = {
        "conversation_id": conv_id,
        "customer_id": rng.choice(CUSTOMER_IDS),
        "source": "synthetic-demo",
        "messages": messages,
    }
    return record, category, found


def build_records(rng: random.Random) -> tuple[list[dict], list[dict], dict]:
    records: list[dict] = []
    issues: list[dict] = []
    labels: dict[str, str] = {}
    seen_keys: set[tuple] = set()

    for n in range(1, NUM_CONVERSATIONS + 1):
        record, category, found = _draw_conversation(rng, n)

        if n not in _UNCHECKED_IDS:
            redraws = 0
            while content_key(record) in seen_keys:
                redraws += 1
                if redraws > MAX_REDRAWS:
                    raise RuntimeError(f"could not draw a unique conversation for conv-{n:04d}")
                record, category, found = _draw_conversation(rng, n)
            seen_keys.add(content_key(record))

        records.append(record)
        labels[record["conversation_id"]] = category
        issues.extend(found)

    # Planted duplicate: conv-0058 becomes an exact copy of conv-0003.
    duplicate = copy.deepcopy(records[DUPLICATE_OF - 1])
    duplicate["conversation_id"] = f"conv-{DUPLICATE_ID:04d}"
    records[DUPLICATE_ID - 1] = duplicate
    labels[f"conv-{DUPLICATE_ID:04d}"] = labels[f"conv-{DUPLICATE_OF:04d}"]
    issues.append({
        "conversation_id": f"conv-{DUPLICATE_ID:04d}",
        "issue": "duplicate_of",
        "duplicate_of": f"conv-{DUPLICATE_OF:04d}",
    })

    # Schema violation: a role that is neither "customer" nor "agent".
    records[INVALID_ROLE_ID - 1]["messages"][0]["role"] = "custmer"
    issues.append({
        "conversation_id": f"conv-{INVALID_ROLE_ID:04d}",
        "issue": "invalid_role",
        "value": "custmer",
    })

    return records, issues, labels


def main() -> None:
    rng = random.Random(SEED)
    records, issues, labels = build_records(rng)
    lines = [json.dumps(r, ensure_ascii=False) for r in records]

    # Malformed lines appended after the valid records.
    lines.append(json.dumps(
        {"conversation_id": "conv-0061", "customer_id": "cust-0009", "source": "synthetic-demo"},
        ensure_ascii=False,
    ))
    issues.append({"line": 61, "conversation_id": "conv-0061", "issue": "missing_messages_field"})

    lines.append('{"conversation_id": "conv-0062", "messages": [')
    issues.append({"line": 62, "conversation_id": "conv-0062", "issue": "invalid_json"})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "conversations.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "brand": BRAND,
        "seed": SEED,
        "description": (
            "Synthetic demo transcripts with planted problems. All names, contact details, "
            "addresses, and order references are fictional."
        ),
        "labels": labels,
        "planted_issues": issues,
    }
    (OUT_DIR / "planted_issues.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"Wrote {len(lines)} lines to {OUT_DIR / 'conversations.jsonl'}")
    print(f"Recorded {len(issues)} planted issues in {OUT_DIR / 'planted_issues.json'}")


if __name__ == "__main__":
    main()
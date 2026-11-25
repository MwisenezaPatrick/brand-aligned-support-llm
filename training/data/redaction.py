"""Detect and replace personal data in one piece of customer-support text.

This module works on a single string. It reads no files, keeps no state, and
gives the same output for the same input every time. Applying it to whole
conversations, writing the redaction report, and checking the demo data are
separate steps.

Categories and placeholders
---------------------------

    EMAIL      email addresses                          -> [EMAIL]
    PHONE      Kenyan and international phone numbers   -> [PHONE]
               (M-Pesa numbers are phone numbers)
    NAME       names from a known-name list             -> [NAME]
    ADDRESS    a house number, a street type and,       -> [ADDRESS]
               if present, a known city
    ORDER_REF  ORD- followed by 5 to 12 digits          -> [ORDER_REF]

Phone formats
-------------

Mobile-style numbers starting 07 or 01 (07XXXXXXXX, 01XXXXXXXX), and
international numbers starting +2547 or +2541 (+2547XXXXXXXX,
+2541XXXXXXXX). Spaces or hyphens may appear between digits. +254 followed by
a trunk 0 (for example +254 0712 345 678) is also matched.

Limits
------

- Names: only names in the list passed to redact_text are redacted, and they
  must be capitalised as in the list. The demo list is not comprehensive.
  Production use needs a dedicated named-entity recognition (NER) system.
- A single name is redacted only after an introduction: "my name is", "I am",
  "I'm" or "this is". A first name used in other text, such as "Grace
  period", is left alone.
- Addresses need a house number and one of the listed street types. A place
  name without a number is left alone.
- Only ORD-prefixed order references are matched. A bare number such as
  "order 123456" is not.
- Landline numbers (for example 020 for Nairobi) are not matched.
- M-Pesa till and paybill numbers are business numbers, not personal data, and
  are not matched.
- National ID numbers are not a category.

Overlaps
--------

When two matches overlap, the longer one wins. "ORD-0712345678" is therefore
an order reference, not a phone number.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache

PLACEHOLDERS: dict[str, str] = {
    "EMAIL": "[EMAIL]",
    "PHONE": "[PHONE]",
    "NAME": "[NAME]",
    "ADDRESS": "[ADDRESS]",
    "ORDER_REF": "[ORDER_REF]",
}

# Names for the synthetic demo. A test checks that they cover the generator's names.
DEMO_FIRST_NAMES: tuple[str, ...] = (
    "Amina", "Brian", "Cynthia", "David", "Esther",
    "Felix", "Grace", "Hassan", "Irene", "John",
)
DEMO_LAST_NAMES: tuple[str, ...] = (
    "Otieno", "Mwangi", "Kamau", "Wekesa",
    "Njeri", "Odhiambo", "Achieng", "Kiptoo",
)

KNOWN_CITIES: tuple[str, ...] = ("Nairobi", "Kisumu", "Mombasa", "Nakuru", "Eldoret")
STREET_TYPES: tuple[str, ...] = (
    "Road", "Street", "Avenue", "Drive", "Lane", "Close", "Way", "Crescent", "Boulevard",
)


def _alt(words: Sequence[str]) -> str:
    """Regex alternation of literal words, longest first so the longest word wins."""
    return "|".join(re.escape(word) for word in sorted(set(words), key=len, reverse=True))


_EMAIL = r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}"
_PHONE_LOCAL = r"(?<!\d)0[17](?:[ \-]?\d){8}(?!\d)"
_PHONE_INTERNATIONAL = r"\+254[ \-]?0?[17](?:[ \-]?\d){8}(?!\d)"
_ORDER_REF = r"\b(?i:ord)-\d{5,12}\b"
_ADDRESS = (
    r"\b\d{1,5}[ \t]+"
    r"(?:[A-Z][a-z]+[ \t]+){1,3}"
    rf"(?:{_alt(STREET_TYPES)})\b"
    rf"(?:,[ \t]*(?:{_alt(KNOWN_CITIES)})\b)?"
)
_INTRODUCTION = r"(?i:\bmy name is|\bi am|\bi['\u2019]m|\bthis is)"


@dataclass(frozen=True)
class _Rule:
    kind: str
    regex: re.Pattern[str]
    group: int = 0


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    kind: str
    priority: int


@dataclass(frozen=True)
class RedactionResult:
    """Redacted text and counts per category. Holds no removed values."""

    text: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


@lru_cache(maxsize=32)
def _rules(first_names: tuple[str, ...], last_names: tuple[str, ...]) -> tuple[_Rule, ...]:
    rules = [
        _Rule("ORDER_REF", re.compile(_ORDER_REF)),
        _Rule("EMAIL", re.compile(_EMAIL)),
        _Rule("PHONE", re.compile(_PHONE_INTERNATIONAL)),
        _Rule("PHONE", re.compile(_PHONE_LOCAL)),
        _Rule("ADDRESS", re.compile(_ADDRESS)),
    ]
    if first_names and last_names:
        full_name = rf"\b(?:{_alt(first_names)})[ \t]+(?:{_alt(last_names)})\b"
        rules.append(_Rule("NAME", re.compile(full_name)))
    all_names = tuple(dict.fromkeys(first_names + last_names))
    if all_names:
        single_name = rf"{_INTRODUCTION}[ \t]+((?:{_alt(all_names)}))\b"
        rules.append(_Rule("NAME", re.compile(single_name), group=1))
    return tuple(rules)


def _find_spans(text: str, rules: tuple[_Rule, ...]) -> list[_Span]:
    candidates: list[_Span] = []
    for priority, rule in enumerate(rules):
        for match in rule.regex.finditer(text):
            start, end = match.span(rule.group)
            if end > start:
                candidates.append(_Span(start, end, rule.kind, priority))

    # Longest first. Ties go to the earlier rule, then to the earlier position.
    candidates.sort(key=lambda span: (-(span.end - span.start), span.priority, span.start))
    chosen: list[_Span] = []
    for span in candidates:
        if all(span.end <= other.start or span.start >= other.end for other in chosen):
            chosen.append(span)
    chosen.sort(key=lambda span: span.start)
    return chosen


def redact_text(
    text: str,
    first_names: Sequence[str] = DEMO_FIRST_NAMES,
    last_names: Sequence[str] = DEMO_LAST_NAMES,
) -> RedactionResult:
    """Replace personal data in text with typed placeholders.

    Pass first_names and last_names to redact names for another brand. Pass
    empty lists to switch name redaction off.
    """
    spans = _find_spans(text, _rules(tuple(first_names), tuple(last_names)))
    counts = {kind: 0 for kind in PLACEHOLDERS}
    pieces: list[str] = []
    position = 0
    for span in spans:
        pieces.append(text[position:span.start])
        pieces.append(PLACEHOLDERS[span.kind])
        counts[span.kind] += 1
        position = span.end
    pieces.append(text[position:])
    return RedactionResult(text="".join(pieces), counts=counts)
"""Brand profile: strict loading, validation, and escalation sentences built from the profile.

The profile is the only place where the brand, catalogue, teams, team order, routing, and
unknown facts are defined. The prompt tables are rendered from it, so routing is never
written twice.

The loader raises ProfileError with a specific message for:
- malformed YAML (with the line number), non-UTF-8 files, and duplicate keys at any level;
- missing or unexpected keys at the top level and inside brand;
- a synthetic value other than the boolean true, and a brand.language other than English;
- team phrases that do not name a team (for example "the " or "a ");
- undefined teams, teams that are defined but never routed to, and a team_order that does
  not list every team exactly once;
- routing and not_provided that name different topics;
- braces in any text, and catalogue names that are empty or repeated.

Escalation sentences:
- A question that needs one team gets the single-team sentence, using that team's phrase.
- A question that needs two or more distinct teams gets one sentence naming each team once,
  in team_order. Two topics routed to the same team name that team once, so they produce the
  single-team sentence.

This module reads one local file and makes no network calls.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PROFILE_PATH = Path(__file__).resolve().parents[2] / "configs" / "brand_profile.yaml"
ESCALATION_PREFIX = "I don't have that information, so I will pass your question to "
TOP_LEVEL_KEYS = frozenset(
    {"profile_version", "synthetic", "brand", "catalogue", "teams", "team_order", "routing", "not_provided"}
)
BRAND_KEYS = frozenset({"name", "notice", "language"})
REQUIRED_LANGUAGE = "English"
VERSION_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
KEY_PATTERN = re.compile(r"[a-z]+(?:_[a-z]+)*")
TEAM_PHRASE_PATTERN = re.compile(r"(?:the|a) [A-Za-z][A-Za-z ]*[A-Za-z]")
ARTICLE_PATTERN = re.compile(r"^(?:the|a) ")
THREE_TEAM_EXAMPLE_TOPICS = ("prices_and_discounts", "order_status", "warranty")


class ProfileError(ValueError):
    """The brand profile is missing, malformed, or inconsistent."""


class _StrictLoader(yaml.SafeLoader):
    """A YAML loader that refuses duplicate mapping keys instead of overwriting them."""

    def construct_mapping(self, node, deep=False):
        if isinstance(node, yaml.MappingNode):
            self.flatten_mapping(node)
            seen = set()
            for key_node, _value_node in node.value:
                key = self.construct_object(key_node, deep=True)
                try:
                    duplicate = key in seen
                except TypeError as exc:
                    raise yaml.constructor.ConstructorError(
                        "while constructing a mapping",
                        node.start_mark,
                        "found an unhashable key",
                        key_node.start_mark,
                    ) from exc
                if duplicate:
                    raise yaml.constructor.ConstructorError(
                        "while constructing a mapping",
                        node.start_mark,
                        f"found duplicate key {key!r}",
                        key_node.start_mark,
                    )
                seen.add(key)
        return super().construct_mapping(node, deep=deep)


@dataclass(frozen=True)
class Profile:
    profile_version: str
    brand_name: str
    catalogue: tuple[str, ...]
    teams: dict[str, str]
    team_order: tuple[str, ...]
    routing: dict[str, str]
    not_provided: dict[str, str]
    file_sha256: str


def _yaml_problem(exc: yaml.YAMLError) -> str:
    problem = getattr(exc, "problem", None) or "syntax error"
    mark = getattr(exc, "problem_mark", None)
    where = f" at line {mark.line + 1}" if mark is not None else ""
    return f"the profile is not valid YAML: {problem}{where}"


def _exact_keys(mapping: dict, required: frozenset[str], where: str) -> None:
    unexpected = sorted(str(key) for key in set(mapping) - required)
    if unexpected:
        raise ProfileError(f"unexpected key(s) in {where}: {unexpected}")
    missing = sorted(required - set(mapping))
    if missing:
        raise ProfileError(f"missing key(s) in {where}: {missing}")


def _check_text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProfileError(f"{where} must be non-empty text")
    if "{" in value or "}" in value:
        raise ProfileError(f"{where} must not contain braces")
    return value


def _check_key(name: Any, where: str) -> str:
    if not isinstance(name, str) or not KEY_PATTERN.fullmatch(name):
        raise ProfileError(f"{where} key {name!r} must be lowercase words joined by single underscores")
    return name


def _check_brand(brand: Any) -> str:
    if not isinstance(brand, dict):
        raise ProfileError("brand must be a mapping")
    _exact_keys(brand, BRAND_KEYS, "brand")
    name = _check_text(brand["name"], "brand.name")
    notice = _check_text(brand["notice"], "brand.notice").lower()
    if "fictional" not in notice or "not a real company" not in notice:
        raise ProfileError("brand.notice must say the brand is fictional and not a real company")
    if brand["language"] != REQUIRED_LANGUAGE:
        raise ProfileError(f"brand.language must be {REQUIRED_LANGUAGE!r}")
    return name


def _check_catalogue(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ProfileError("catalogue must be a non-empty list")
    names = tuple(_check_text(item, "catalogue entry") for item in value)
    if len(set(names)) != len(names):
        raise ProfileError("catalogue names must be unique")
    return names


def _check_teams(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ProfileError("teams must be a non-empty mapping")
    teams: dict[str, str] = {}
    for key, phrase in value.items():
        _check_key(key, "teams")
        if not isinstance(phrase, str) or not TEAM_PHRASE_PATTERN.fullmatch(phrase):
            raise ProfileError(
                f"teams.{key} must be an article and a team name, such as 'the sales team' (got {phrase!r})"
            )
        teams[key] = phrase
    if len(set(teams.values())) != len(teams):
        raise ProfileError("two teams share the same phrase")
    return teams


def _check_team_order(value: Any, teams: dict[str, str]) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ProfileError("team_order must be a list of team keys")
    if len(value) != len(set(value)):
        raise ProfileError("team_order must not repeat a team")
    unknown = sorted(set(value) - set(teams))
    if unknown:
        raise ProfileError(f"team_order names undefined teams: {unknown}")
    missing = sorted(set(teams) - set(value))
    if missing:
        raise ProfileError(f"team_order does not list: {missing}")
    return tuple(value)


def _check_routing(value: Any, teams: dict[str, str]) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ProfileError("routing must be a non-empty mapping")
    routing: dict[str, str] = {}
    for topic, team in value.items():
        _check_key(topic, "routing")
        if not isinstance(team, str) or team not in teams:
            raise ProfileError(f"routing.{topic} points to an undefined team: {team!r}")
        routing[topic] = team
    return routing


def _check_every_team_routed(teams: dict[str, str], routing: dict[str, str]) -> None:
    unused = sorted(set(teams) - set(routing.values()))
    if unused:
        raise ProfileError(f"teams are defined but never routed to: {unused}")


def _check_not_provided(value: Any, routing: dict[str, str]) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ProfileError("not_provided must be a non-empty mapping")
    if set(value) != set(routing):
        raise ProfileError("not_provided and routing must name the same topics")
    return {topic: _check_text(text, f"not_provided.{topic}") for topic, text in value.items()}


def load_profile(path: Path = DEFAULT_PROFILE_PATH) -> Profile:
    """Read and validate the profile file. Raises ProfileError with a specific message."""
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise ProfileError(f"cannot read the profile: {exc.strerror or exc}") from exc
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ProfileError("the profile is not valid UTF-8") from exc
    try:
        data = yaml.load(text, Loader=_StrictLoader)
    except yaml.YAMLError as exc:
        raise ProfileError(_yaml_problem(exc)) from exc
    if not isinstance(data, dict):
        raise ProfileError("the profile must be a mapping at the top level")
    _exact_keys(data, TOP_LEVEL_KEYS, "top level")

    version = data["profile_version"]
    if not isinstance(version, str) or not VERSION_PATTERN.fullmatch(version):
        raise ProfileError("profile_version must be lowercase letters, digits, and single hyphens")
    if data["synthetic"] is not True:
        raise ProfileError("synthetic must be the boolean true")
    brand_name = _check_brand(data["brand"])
    catalogue = _check_catalogue(data["catalogue"])
    teams = _check_teams(data["teams"])
    team_order = _check_team_order(data["team_order"], teams)
    routing = _check_routing(data["routing"], teams)
    _check_every_team_routed(teams, routing)
    not_provided = _check_not_provided(data["not_provided"], routing)
    return Profile(
        profile_version=version,
        brand_name=brand_name,
        catalogue=catalogue,
        teams=teams,
        team_order=team_order,
        routing=routing,
        not_provided=not_provided,
        file_sha256=hashlib.sha256(raw).hexdigest(),
    )


def required_teams(profile: Profile, topics: Sequence[str]) -> tuple[str, ...]:
    """The team keys a question needs, in team_order, each named once."""
    if not isinstance(topics, (list, tuple)):
        raise ValueError("topics must be a list or tuple of topic names")
    if not topics:
        raise ValueError("at least one topic is required")
    unknown = [topic for topic in topics if not isinstance(topic, str) or topic not in profile.routing]
    if unknown:
        raise ValueError(f"unknown topic(s): {unknown!r}")
    needed = {profile.routing[topic] for topic in topics}
    return tuple(key for key in profile.team_order if key in needed)


def _join_phrases(phrases: list[str]) -> str:
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return ", ".join(phrases[:-1]) + ", and " + phrases[-1]


def escalation_sentence(profile: Profile, topics: Sequence[str]) -> str:
    """One escalation sentence for the given topics. See the module docstring for the rules."""
    phrases = [profile.teams[key] for key in required_teams(profile, topics)]
    return ESCALATION_PREFIX + _join_phrases(phrases) + "."


def team_label(phrase: str) -> str:
    """The team name without its leading article, for use in tables."""
    return ARTICLE_PATTERN.sub("", phrase, count=1)


def render_escalation_table(profile: Profile) -> str:
    lines = []
    for key in profile.team_order:
        phrase = profile.teams[key]
        lines.append(f"- {team_label(phrase)}: {ESCALATION_PREFIX}{phrase}.")
    return "\n".join(lines)


def render_routing_table(profile: Profile) -> str:
    return "\n".join(
        f"{topic.replace('_', ' ')} -> {profile.teams[team]}" for topic, team in profile.routing.items()
    )


def render_unknown_facts(profile: Profile) -> str:
    return "\n".join(f"- {text}" for text in profile.not_provided.values())


def render_catalogue(profile: Profile) -> str:
    return ", ".join(profile.catalogue)


def render_three_team_example(profile: Profile) -> str:
    return escalation_sentence(profile, THREE_TEAM_EXAMPLE_TOPICS)

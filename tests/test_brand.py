"""Tests for the brand profile and the escalation sentences built from it.

These read the committed profile or write temporary copies. No model is used. They verify
the loader's refusals, the one-team and multi-team sentence rules, team order, duplicate
teams, and the rendered tables. They do not verify that a model follows these rules.
"""

import copy
import hashlib
import itertools
import re

import pytest
import yaml

from training.inference.brand import (
    DEFAULT_PROFILE_PATH,
    ESCALATION_PREFIX,
    ProfileError,
    escalation_sentence,
    load_profile,
    render_catalogue,
    render_escalation_table,
    render_routing_table,
    render_unknown_facts,
    team_label,
)

PROFILE_TEXT = DEFAULT_PROFILE_PATH.read_text(encoding="utf-8-sig")
TOP_LEVEL_KEYS = [
    "profile_version", "synthetic", "brand", "catalogue",
    "teams", "team_order", "routing", "not_provided",
]
EXPECTED_TEAM_ORDER = (
    "sales_team", "logistics_team", "support_team", "account_team",
    "payments_team", "product_team", "senior_team",
)
EXPECTED_ROUTING = {
    "prices_and_discounts": "sales_team",
    "stock_and_availability": "sales_team",
    "delivery_dates_and_areas": "logistics_team",
    "order_status": "logistics_team",
    "order_cancellation": "support_team",
    "warranty": "support_team",
    "refunds_and_returns": "support_team",
    "opening_hours_and_contact": "support_team",
    "privacy_and_data_requests": "support_team",
    "email_and_account_changes": "account_team",
    "password_and_login": "account_team",
    "payment_methods": "payments_team",
    "product_features_and_care": "product_team",
    "troubleshooting_and_faults": "product_team",
    "safety_certifications": "product_team",
    "product_not_in_catalogue": "product_team",
    "safety_and_injury": "senior_team",
    "complaints_about_staff_or_repeated_problems": "senior_team",
}


@pytest.fixture(scope="module")
def profile():
    return load_profile()


def profile_data():
    return copy.deepcopy(yaml.safe_load(PROFILE_TEXT))


def dump(data):
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def refuse(tmp_path, match, data=None, text=None):
    path = tmp_path / "profile.yaml"
    path.write_text(text if text is not None else dump(data), encoding="utf-8")
    with pytest.raises(ProfileError, match=match):
        load_profile(path)


# Committed profile


def test_committed_profile_loads_with_expected_routes_and_order():
    loaded = load_profile()
    assert loaded.profile_version == "nairobitech-brand-profile-v1"
    assert loaded.brand_name == "NairobiTech (demo)"
    assert loaded.team_order == EXPECTED_TEAM_ORDER
    assert loaded.routing == EXPECTED_ROUTING


def test_profile_hash_is_sha256_of_the_raw_bytes():
    assert load_profile().file_sha256 == hashlib.sha256(DEFAULT_PROFILE_PATH.read_bytes()).hexdigest()


def test_utf8_bom_and_crlf_line_endings_are_accepted(tmp_path):
    path = tmp_path / "profile.yaml"
    path.write_bytes(b"\xef\xbb\xbf" + dump(profile_data()).replace("\n", "\r\n").encode("utf-8"))
    assert load_profile(path).profile_version == "nairobitech-brand-profile-v1"


# Malformed input


def test_malformed_yaml_reports_a_line_number(tmp_path):
    refuse(tmp_path, r"not valid YAML.*at line \d+", text="profile_version: x\nbrand: [unclosed\n")


def test_empty_file_is_refused(tmp_path):
    refuse(tmp_path, "must be a mapping", text="")


def test_top_level_list_is_refused(tmp_path):
    refuse(tmp_path, "must be a mapping", text="- a\n- b\n")


def test_duplicate_top_level_key_is_refused(tmp_path):
    refuse(tmp_path, "duplicate key", text="profile_version: a\nprofile_version: b\n")


def test_duplicate_routing_key_is_refused(tmp_path):
    text = (
        "profile_version: nairobitech-brand-profile-v1\n"
        "synthetic: true\n"
        "routing:\n"
        "  warranty: support_team\n"
        "  warranty: sales_team\n"
    )
    refuse(tmp_path, "duplicate key", text=text)


def test_duplicate_brand_key_is_refused(tmp_path):
    refuse(tmp_path, "duplicate key", text="brand:\n  name: a\n  name: b\n")


def test_non_utf8_file_is_refused(tmp_path):
    path = tmp_path / "profile.yaml"
    path.write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(ProfileError, match="not valid UTF-8"):
        load_profile(path)


# Missing, unexpected, and invalid fields


@pytest.mark.parametrize("key", TOP_LEVEL_KEYS)
def test_missing_top_level_key_is_refused(tmp_path, key):
    data = profile_data()
    del data[key]
    refuse(tmp_path, r"missing key\(s\) in top level", data=data)


def test_unexpected_top_level_key_is_refused(tmp_path):
    data = profile_data()
    data["notes"] = "x"
    refuse(tmp_path, r"unexpected key\(s\) in top level", data=data)


@pytest.mark.parametrize("key", ["name", "notice", "language"])
def test_missing_brand_field_is_refused(tmp_path, key):
    data = profile_data()
    del data["brand"][key]
    refuse(tmp_path, r"missing key\(s\) in brand", data=data)


def test_unexpected_brand_field_is_refused(tmp_path):
    data = profile_data()
    data["brand"]["tone"] = ["professional"]
    refuse(tmp_path, r"unexpected key\(s\) in brand", data=data)


def test_brand_must_be_a_mapping(tmp_path):
    data = profile_data()
    data["brand"] = "NairobiTech"
    refuse(tmp_path, "brand must be a mapping", data=data)


def test_brand_language_must_be_english(tmp_path):
    data = profile_data()
    data["brand"]["language"] = "French"
    refuse(tmp_path, "brand.language must be", data=data)


def test_brand_notice_must_say_fictional_and_not_a_real_company(tmp_path):
    data = profile_data()
    data["brand"]["notice"] = "Fictional brand for testing only."
    refuse(tmp_path, "must say the brand is fictional and not a real company", data=data)


@pytest.mark.parametrize("value", [False, "true", None])
def test_synthetic_must_be_boolean_true(tmp_path, value):
    data = profile_data()
    data["synthetic"] = value
    refuse(tmp_path, "synthetic must be the boolean true", data=data)


@pytest.mark.parametrize("value", ["NairobiTech v1", "nairobitech_v1", ""])
def test_profile_version_format_is_checked(tmp_path, value):
    data = profile_data()
    data["profile_version"] = value
    refuse(tmp_path, "profile_version must be lowercase", data=data)


# Teams


@pytest.mark.parametrize("phrase", ["the ", "the", "a ", " ", "sales team"])
def test_team_phrase_needs_an_article_and_a_name(tmp_path, phrase):
    data = profile_data()
    data["teams"]["sales_team"] = phrase
    refuse(tmp_path, "must be an article and a team name", data=data)


def test_duplicate_team_phrases_are_refused(tmp_path):
    data = profile_data()
    data["teams"]["logistics_team"] = "the sales team"
    refuse(tmp_path, "two teams share the same phrase", data=data)


@pytest.mark.parametrize("new_key", ["Sales", "sales-team"])
def test_team_keys_must_be_lowercase_words(tmp_path, new_key):
    data = profile_data()
    data["teams"][new_key] = data["teams"].pop("sales_team")
    refuse(tmp_path, "must be lowercase words", data=data)


def test_undefined_team_in_routing_is_refused(tmp_path):
    data = profile_data()
    data["routing"]["prices_and_discounts"] = "legal_team"
    refuse(tmp_path, "points to an undefined team", data=data)


def test_unused_team_is_refused(tmp_path):
    data = profile_data()
    data["teams"]["legal_team"] = "the legal team"
    data["team_order"].append("legal_team")
    refuse(tmp_path, "never routed to", data=data)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda d: d["team_order"].remove("senior_team"), "does not list"),
        (lambda d: d["team_order"].append("sales_team"), "must not repeat"),
        (lambda d: d["team_order"].__setitem__(-1, "legal_team"), "undefined teams"),
    ],
)
def test_team_order_must_list_each_team_once(tmp_path, mutate, match):
    data = profile_data()
    mutate(data)
    refuse(tmp_path, match, data=data)


# Routing, unknown facts, and catalogue


def test_routing_and_not_provided_must_name_the_same_topics(tmp_path):
    data = profile_data()
    del data["not_provided"]["warranty"]
    refuse(tmp_path, "must name the same topics", data=data)


def test_braces_are_refused_in_profile_text(tmp_path):
    data = profile_data()
    data["catalogue"][0] = "{blender}"
    refuse(tmp_path, "must not contain braces", data=data)


@pytest.mark.parametrize("value, match", [([], "non-empty list"), (["kettle", "kettle"], "must be unique")])
def test_catalogue_must_be_nonempty_and_unique(tmp_path, value, match):
    data = profile_data()
    data["catalogue"] = value
    refuse(tmp_path, match, data=data)


# Escalation sentences


def test_one_topic_gives_the_single_team_sentence(profile):
    assert escalation_sentence(profile, ["warranty"]) == ESCALATION_PREFIX + "the support team."


def test_two_topics_routed_to_one_team_name_that_team_once(profile):
    sentence = escalation_sentence(profile, ["prices_and_discounts", "stock_and_availability"])
    assert sentence == escalation_sentence(profile, ["prices_and_discounts"])
    assert sentence.count("the sales team") == 1


def test_two_teams_are_joined_with_and_in_team_order(profile):
    sentence = escalation_sentence(profile, ["warranty", "prices_and_discounts"])
    assert sentence == ESCALATION_PREFIX + "the sales team and the support team."


def test_three_teams_use_commas_and_and_before_the_last(profile):
    sentence = escalation_sentence(profile, ["warranty", "order_status", "prices_and_discounts"])
    assert sentence == ESCALATION_PREFIX + "the sales team, the logistics team, and the support team."


def test_result_does_not_depend_on_topic_order(profile):
    topics = ["warranty", "order_status", "prices_and_discounts"]
    results = {escalation_sentence(profile, list(order)) for order in itertools.permutations(topics)}
    assert len(results) == 1


def test_every_team_is_named_exactly_once_when_all_topics_are_asked(profile):
    sentence = escalation_sentence(profile, list(profile.routing))
    for phrase in profile.teams.values():
        assert sentence.count(phrase) == 1, phrase
    assert sentence.endswith(".")


@pytest.mark.parametrize(
    "topics",
    [
        ["warranty"],
        ["prices_and_discounts", "stock_and_availability"],
        ["warranty", "prices_and_discounts"],
        ["warranty", "order_status", "prices_and_discounts"],
    ],
)
def test_each_sentence_has_exactly_one_sentence_ending(profile, topics):
    sentence = escalation_sentence(profile, topics)
    assert re.findall(r"[.!?]", sentence) == ["."]
    assert sentence.endswith(".")


def test_unknown_topic_is_refused(profile):
    with pytest.raises(ValueError, match="unknown topic"):
        escalation_sentence(profile, ["warranty", "not_a_topic"])


def test_empty_topic_list_is_refused(profile):
    with pytest.raises(ValueError, match="at least one topic"):
        escalation_sentence(profile, [])


def test_string_topics_are_refused(profile):
    with pytest.raises(ValueError, match="list or tuple"):
        escalation_sentence(profile, "warranty")


# Rendered tables


def test_escalation_table_has_one_line_per_team_in_team_order(profile):
    lines = render_escalation_table(profile).splitlines()
    assert len(lines) == len(profile.team_order)
    labels = [line[2:].split(": ", 1)[0] for line in lines]
    assert labels == [team_label(profile.teams[key]) for key in profile.team_order]


def test_routing_table_has_one_line_per_topic(profile):
    lines = render_routing_table(profile).splitlines()
    assert len(lines) == len(profile.routing)
    assert lines[0] == "prices and discounts -> the sales team"


def test_unknown_facts_and_catalogue_render_in_profile_order(profile):
    lines = render_unknown_facts(profile).splitlines()
    assert len(lines) == len(profile.not_provided)
    assert lines[0] == "- prices, discounts, and bundle offers"
    assert render_catalogue(profile).startswith("1-litre steel bottle, insulated lunch box")


def test_team_label_removes_only_the_leading_article():
    assert team_label("the sales team") == "sales team"
    assert team_label("a senior team member") == "senior team member"

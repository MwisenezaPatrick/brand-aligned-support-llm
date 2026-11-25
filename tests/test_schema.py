import pytest

from training.data.schema import validate_conversation, validate_message


def make_conversation(**overrides):
    record = {
        "conversation_id": "conv-0001",
        "customer_id": "cust-0001",
        "messages": [
            {"role": "customer", "text": "Does the kettle come with a lid?"},
            {"role": "agent", "text": "Yes, it does."},
        ],
    }
    record.update(overrides)
    return record


def test_valid_conversation_has_no_errors():
    assert validate_conversation(make_conversation()) == []


def test_customer_only_conversation_is_schema_valid():
    # Incomplete conversations are flagged by the cleaning step, not rejected here.
    record = make_conversation(messages=[{"role": "customer", "text": "Hello?"}])
    assert validate_conversation(record) == []


@pytest.mark.parametrize("field_name", ["conversation_id", "customer_id"])
def test_missing_id_is_reported(field_name):
    record = make_conversation()
    del record[field_name]
    errors = validate_conversation(record)
    assert [(e.field, e.reason) for e in errors] == [(field_name, "is missing")]


@pytest.mark.parametrize(
    "bad_id", ["", "conv 0001", "conv/0001", "../etc", "x" * 101, None, 42]
)
def test_invalid_conversation_id_is_reported(bad_id):
    errors = validate_conversation(make_conversation(conversation_id=bad_id))
    assert [e.field for e in errors] == ["conversation_id"]


@pytest.mark.parametrize("messages", [[], "not a list", None])
def test_messages_must_be_a_non_empty_list(messages):
    errors = validate_conversation(make_conversation(messages=messages))
    assert len(errors) == 1
    assert errors[0].field == "messages"


def test_missing_messages_is_reported():
    record = make_conversation()
    del record["messages"]
    errors = validate_conversation(record)
    assert [(e.field, e.reason) for e in errors] == [("messages", "is missing")]


@pytest.mark.parametrize("role", ["custmer", "Customer", "bot", "", None])
def test_invalid_role_is_reported_with_message_index(role):
    messages = [
        {"role": "customer", "text": "Hi"},
        {"role": role, "text": "Hello"},
    ]
    errors = validate_conversation(make_conversation(messages=messages))
    assert len(errors) == 1
    assert errors[0].field == "messages[1].role"
    assert errors[0].message_index == 1


def test_missing_role_is_reported():
    errors = validate_message({"text": "Hi"}, 0)
    assert [(e.field, e.reason) for e in errors] == [("messages[0].role", "is missing")]


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_empty_text_is_reported(text):
    messages = [{"role": "customer", "text": text}]
    errors = validate_conversation(make_conversation(messages=messages))
    assert len(errors) == 1
    assert errors[0].field == "messages[0].text"
    assert errors[0].reason == "must not be empty"


def test_non_string_text_is_reported():
    errors = validate_message({"role": "agent", "text": 123}, 0)
    assert errors[0].field == "messages[0].text"
    assert errors[0].reason == "must be a string"


def test_message_must_be_an_object():
    errors = validate_message("hello", 2)
    assert errors[0].field == "messages[2]"
    assert errors[0].message_index == 2


def test_record_must_be_an_object():
    errors = validate_conversation(["not", "a", "dict"])
    assert errors[0].field == "record"


def test_error_string_is_readable():
    errors = validate_message({"role": "custmer", "text": "Hi"}, 0)
    assert str(errors[0]) == (
        "messages[0].role must be 'customer' or 'agent', got 'custmer'"
    )

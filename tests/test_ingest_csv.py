import io
from pathlib import Path

from training.data.ingest import load_csv, parse_csv

HEADER = "conversation_id,customer_id,role,text\n"


def parse(text: str):
    return parse_csv(io.StringIO(text), source="test.csv")


def messages_of(result, conversation_id: str) -> list[tuple[str, str]]:
    for conversation in result.conversations:
        if conversation["conversation_id"] == conversation_id:
            return [(m["role"], m["text"]) for m in conversation["messages"]]
    raise AssertionError(f"conversation {conversation_id!r} not found")


def texts_of(result, conversation_id: str) -> list[str]:
    return [text for _, text in messages_of(result, conversation_id)]


def write_file(tmp_path, content: str) -> Path:
    path = tmp_path / "messages.csv"
    path.write_text(content, encoding="utf-8", newline="\n")
    return path


# Valid files and grouping


def test_valid_file_builds_conversations_in_file_order():
    result = parse(HEADER + (
        "conv-0001,cust-0001,customer,Does the kettle have a lid?\n"
        "conv-0001,cust-0001,agent,Yes it does.\n"
        "conv-0002,cust-0002,customer,Where is my order?\n"
    ))
    assert result.errors == []
    assert result.rows_read == 3
    assert result.rows_accepted == 3
    assert result.order_source == "file order"
    assert messages_of(result, "conv-0001") == [
        ("customer", "Does the kettle have a lid?"),
        ("agent", "Yes it does."),
    ]
    assert [c["conversation_id"] for c in result.conversations] == ["conv-0001", "conv-0002"]
    assert result.conversations[1]["customer_id"] == "cust-0002"


def test_interleaved_rows_are_grouped_by_conversation():
    result = parse(HEADER + (
        "conv-0001,cust-0001,customer,Hi from one\n"
        "conv-0002,cust-0002,customer,Hi from two\n"
        "conv-0001,cust-0001,agent,Reply to one\n"
        "conv-0002,cust-0002,agent,Reply to two\n"
    ))
    assert result.errors == []
    assert messages_of(result, "conv-0001") == [
        ("customer", "Hi from one"),
        ("agent", "Reply to one"),
    ]
    assert messages_of(result, "conv-0002") == [
        ("customer", "Hi from two"),
        ("agent", "Reply to two"),
    ]


def test_rows_separated_by_other_conversations_are_merged_without_error():
    result = parse(HEADER + (
        "conv-0001,cust-0001,customer,First question\n"
        "conv-0002,cust-0002,customer,Unrelated\n"
        "conv-0003,cust-0003,customer,Another\n"
        "conv-0001,cust-0001,agent,Late reply\n"
    ))
    assert result.errors == []
    assert texts_of(result, "conv-0001") == ["First question", "Late reply"]


def test_identical_message_rows_are_kept_for_the_cleaning_step():
    result = parse(HEADER + (
        "conv-0001,cust-0001,customer,Same text\n"
        "conv-0001,cust-0001,customer,Same text\n"
    ))
    assert result.errors == []
    assert texts_of(result, "conv-0001") == ["Same text", "Same text"]


# Message ordering


def test_message_order_column_sorts_messages():
    result = parse(
        "conversation_id,customer_id,role,text,message_order\n"
        "conv-0001,cust-0001,agent,Second,2\n"
        "conv-0001,cust-0001,customer,First,1\n"
        "conv-0001,cust-0001,customer,Third,3\n"
    )
    assert result.errors == []
    assert result.order_source == "message_order"
    assert texts_of(result, "conv-0001") == ["First", "Second", "Third"]


def test_order_applies_within_each_conversation_when_rows_are_interleaved():
    result = parse(
        "conversation_id,customer_id,role,text,message_order\n"
        "conv-0001,cust-0001,agent,A2,2\n"
        "conv-0002,cust-0002,agent,B2,2\n"
        "conv-0001,cust-0001,customer,A1,1\n"
        "conv-0002,cust-0002,customer,B1,1\n"
    )
    assert result.errors == []
    assert texts_of(result, "conv-0001") == ["A1", "A2"]
    assert texts_of(result, "conv-0002") == ["B1", "B2"]


def test_equal_message_order_values_keep_file_order():
    result = parse(
        "conversation_id,customer_id,role,text,message_order\n"
        "conv-0001,cust-0001,customer,Written first,5\n"
        "conv-0001,cust-0001,agent,Written second,5\n"
    )
    assert texts_of(result, "conv-0001") == ["Written first", "Written second"]


def test_timestamp_used_when_message_order_is_absent():
    result = parse(
        "conversation_id,customer_id,role,text,timestamp\n"
        "conv-0001,cust-0001,agent,Reply,2026-11-25T09:05:00Z\n"
        "conv-0001,cust-0001,customer,Question,2026-11-25T09:00:00Z\n"
    )
    assert result.errors == []
    assert result.order_source == "timestamp"
    assert texts_of(result, "conv-0001") == ["Question", "Reply"]


def test_timestamp_without_timezone_is_read_as_utc():
    result = parse(
        "conversation_id,customer_id,role,text,timestamp\n"
        "conv-0001,cust-0001,agent,Reply,2026-11-25T09:05:00\n"
        "conv-0001,cust-0001,customer,Question,2026-11-25T09:00:00Z\n"
    )
    assert result.errors == []
    assert texts_of(result, "conv-0001") == ["Question", "Reply"]


def test_message_order_takes_precedence_over_timestamp():
    result = parse(
        "conversation_id,customer_id,role,text,message_order,timestamp\n"
        "conv-0001,cust-0001,customer,Second by order,2,2026-11-25T08:00:00Z\n"
        "conv-0001,cust-0001,agent,First by order,1,2026-11-25T10:00:00Z\n"
    )
    assert result.order_source == "message_order"
    assert texts_of(result, "conv-0001") == ["First by order", "Second by order"]


# Malformed rows


def test_invalid_message_order_is_reported_and_other_rows_kept():
    result = parse(
        "conversation_id,customer_id,role,text,message_order\n"
        "conv-0001,cust-0001,customer,Good row,1\n"
        "conv-0001,cust-0001,agent,Bad order,second\n"
        "conv-0001,cust-0001,agent,Empty order,\n"
    )
    assert [e.line_number for e in result.errors] == [3, 4]
    assert "message_order must be a whole number" in result.errors[0].message
    assert messages_of(result, "conv-0001") == [("customer", "Good row")]
    assert result.rows_accepted == 1


def test_invalid_timestamp_is_reported():
    result = parse(
        "conversation_id,customer_id,role,text,timestamp\n"
        "conv-0001,cust-0001,customer,Good,2026-11-25T09:00:00Z\n"
        "conv-0001,cust-0001,agent,Bad,yesterday\n"
    )
    assert [e.line_number for e in result.errors] == [3]
    assert "timestamp must be an ISO 8601" in str(result.errors[0])
    assert texts_of(result, "conv-0001") == ["Good"]


def test_invalid_role_is_reported_and_other_messages_kept():
    result = parse(HEADER + (
        "conv-0001,cust-0001,customer,Hello\n"
        "conv-0001,cust-0001,custmer,Typo in role\n"
        "conv-0001,cust-0001,agent,Hi back\n"
    ))
    assert [e.line_number for e in result.errors] == [3]
    assert "role must be 'customer' or 'agent', got 'custmer'" in result.errors[0].message
    assert messages_of(result, "conv-0001") == [("customer", "Hello"), ("agent", "Hi back")]


def test_ids_and_role_are_trimmed_but_text_is_kept_as_written():
    result = parse(HEADER + " conv-0001 , cust-0001 , agent ,   Spaced text  \n")
    assert result.errors == []
    assert result.conversations[0]["conversation_id"] == "conv-0001"
    assert messages_of(result, "conv-0001") == [("agent", "   Spaced text  ")]


def test_empty_text_is_reported():
    result = parse(HEADER + (
        "conv-0001,cust-0001,customer,\n"
        "conv-0001,cust-0001,agent,Reply\n"
    ))
    assert [e.line_number for e in result.errors] == [2]
    assert "text must not be empty" in result.errors[0].message
    assert messages_of(result, "conv-0001") == [("agent", "Reply")]


def test_row_with_missing_fields_is_reported():
    result = parse(HEADER + (
        "conv-0001,cust-0001,customer\n"
        "conv-0001,cust-0001,agent,Reply\n"
    ))
    assert [e.line_number for e in result.errors] == [2]
    assert "expected 4 fields but found 3" in result.errors[0].message
    assert messages_of(result, "conv-0001") == [("agent", "Reply")]


def test_unquoted_comma_is_reported_as_extra_field():
    result = parse(HEADER + "conv-0001,cust-0001,customer,Yes, it does\n")
    assert "expected 4 fields but found 5" in result.errors[0].message
    assert result.conversations == []


def test_empty_conversation_id_is_reported():
    result = parse(HEADER + (
        ",cust-0001,customer,Orphan message\n"
        "conv-0001,cust-0001,customer,Good message\n"
    ))
    assert [e.line_number for e in result.errors] == [2]
    assert "conversation_id must be a non-empty string" in result.errors[0].message
    assert texts_of(result, "conv-0001") == ["Good message"]


def test_customer_mismatch_is_reported_and_first_customer_kept():
    result = parse(HEADER + (
        "conv-0001,cust-0001,customer,Question\n"
        "conv-0001,cust-0002,agent,Reply from wrong customer\n"
        "conv-0001,cust-0001,agent,Correct reply\n"
    ))
    assert [e.line_number for e in result.errors] == [3]
    assert "does not match" in result.errors[0].message
    assert result.conversations[0]["customer_id"] == "cust-0001"
    assert messages_of(result, "conv-0001") == [
        ("customer", "Question"),
        ("agent", "Correct reply"),
    ]


def test_error_line_numbers_match_source_after_multiline_field():
    result = parse(HEADER + (
        'conv-0001,cust-0001,customer,"Line one\nLine two"\n'
        "conv-0001,cust-0001,custmer,Bad role after two-line message\n"
    ))
    assert [e.line_number for e in result.errors] == [4]
    assert texts_of(result, "conv-0001") == ["Line one\nLine two"]


def test_blank_rows_are_skipped_without_error():
    result = parse(HEADER + (
        "conv-0001,cust-0001,customer,Hello\n"
        "\n"
        ",,,\n"
        "conv-0001,cust-0001,agent,Hi\n"
    ))
    assert result.errors == []
    assert result.rows_read == 2
    assert messages_of(result, "conv-0001") == [("customer", "Hello"), ("agent", "Hi")]


# File-level problems


def test_missing_required_column_is_a_file_error():
    result = parse("conversation_id,customer_id,role\nconv-0001,cust-0001,customer\n")
    assert result.conversations == []
    assert result.errors[0].line_number == 1
    assert "missing required column(s): text" in result.errors[0].message


def test_duplicate_column_name_is_a_file_error():
    result = parse("conversation_id,customer_id,role,text,text\nconv-0001,cust-0001,customer,Hi,Hi\n")
    assert result.conversations == []
    assert "duplicate column name(s): text" in result.errors[0].message


def test_empty_file_is_a_file_error():
    result = parse("")
    assert result.conversations == []
    assert result.errors[0].line_number is None
    assert "file is empty" in result.errors[0].message


def test_utf8_bom_and_header_case_are_accepted(tmp_path):
    path = write_file(
        tmp_path,
        "\ufeffConversation_ID,Customer_ID,Role,Text\n"
        "conv-0001,cust-0001,customer,Hi there\n",
    )
    result = load_csv(path)
    assert result.errors == []
    assert result.rows_accepted == 1


def test_file_that_is_not_utf8_is_a_file_error(tmp_path):
    path = tmp_path / "messages.csv"
    path.write_bytes(HEADER.encode("utf-8") + b"conv-0001,cust-0001,customer,caf\xe9\n")
    result = load_csv(path)
    assert result.conversations == []
    assert result.errors[0].line_number is None
    assert "not valid UTF-8" in result.errors[0].message


def test_missing_file_is_a_file_error(tmp_path):
    result = load_csv(tmp_path / "does_not_exist.csv")
    assert result.conversations == []
    assert "file not found" in result.errors[0].message


def test_loading_does_not_modify_the_source_file(tmp_path):
    path = write_file(tmp_path, HEADER + "conv-0001,cust-0001,customer,Hi\n")
    before = path.read_bytes()
    load_csv(path)
    assert path.read_bytes() == before
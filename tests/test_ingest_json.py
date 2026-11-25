import io
import json

from training.data.ingest import (
    load_json,
    load_jsonl,
    parse_csv,
    parse_json,
    parse_jsonl,
)

KETTLE_QUESTION = "Does the kettle come with a lid?"
KETTLE_ANSWER = "Yes, it does."


def valid_record(conversation_id="conv-0001", customer_id="cust-0001", **extra):
    record = {
        "conversation_id": conversation_id,
        "customer_id": customer_id,
        "messages": [
            {"role": "customer", "text": KETTLE_QUESTION},
            {"role": "agent", "text": KETTLE_ANSWER},
        ],
    }
    record.update(extra)
    return record


def as_json(*records) -> str:
    return json.dumps(list(records), ensure_ascii=False)


def as_jsonl(*records) -> str:
    return "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)


def parse_json_text(text: str):
    return parse_json(text, source="test.json")


def parse_jsonl_text(text: str):
    return parse_jsonl(io.StringIO(text), source="test.jsonl")


def conversation_ids(result) -> list[str]:
    return [conversation["conversation_id"] for conversation in result.conversations]


# JSON arrays


def test_json_array_loads_every_valid_conversation():
    result = parse_json_text(as_json(
        valid_record("conv-0001", "cust-0001"),
        valid_record("conv-0002", "cust-0002"),
    ))
    assert result.errors == []
    assert result.rows_read == 2
    assert result.rows_accepted == 2
    assert conversation_ids(result) == ["conv-0001", "conv-0002"]


def test_output_matches_canonical_shape_and_drops_extra_fields():
    record = valid_record(source="synthetic-demo")
    record["messages"][0]["message_id"] = 17
    result = parse_json_text(as_json(record))
    assert result.conversations == [{
        "conversation_id": "conv-0001",
        "customer_id": "cust-0001",
        "messages": [
            {"role": "customer", "text": KETTLE_QUESTION},
            {"role": "agent", "text": KETTLE_ANSWER},
        ],
    }]


def test_message_order_is_kept_as_written():
    record = valid_record()
    record["messages"] = [
        {"role": "customer", "text": "First"},
        {"role": "agent", "text": "Second"},
        {"role": "customer", "text": "Third"},
    ]
    result = parse_json_text(as_json(record))
    assert [m["text"] for m in result.conversations[0]["messages"]] == [
        "First", "Second", "Third",
    ]


def test_invalid_record_is_reported_and_valid_records_are_kept():
    bad = valid_record("conv-0002", "cust-0002")
    bad["messages"][0]["role"] = "custmer"
    result = parse_json_text(as_json(
        valid_record("conv-0001"),
        bad,
        valid_record("conv-0003", "cust-0003"),
    ))
    assert conversation_ids(result) == ["conv-0001", "conv-0003"]
    assert result.rows_read == 3
    assert result.rows_accepted == 2
    assert result.rows_rejected == 1
    assert [str(e) for e in result.errors] == [
        "record 2 (conversation_id 'conv-0002'): "
        "messages[0].role must be 'customer' or 'agent', got 'custmer'"
    ]
    assert result.errors[0].line_number is None


def test_missing_field_is_reported_with_record_number():
    record = valid_record()
    del record["customer_id"]
    result = parse_json_text(as_json(record))
    assert result.conversations == []
    assert [str(e) for e in result.errors] == [
        "record 1 (conversation_id 'conv-0001'): customer_id is missing"
    ]


def test_empty_text_and_empty_messages_are_reported():
    blank_text = valid_record("conv-0001")
    blank_text["messages"][1]["text"] = "   "
    no_messages = valid_record("conv-0002", "cust-0002")
    no_messages["messages"] = []
    result = parse_json_text(as_json(blank_text, no_messages))
    assert result.conversations == []
    assert [str(e) for e in result.errors] == [
        "record 1 (conversation_id 'conv-0001'): messages[1].text must not be empty",
        "record 2 (conversation_id 'conv-0002'): messages must contain at least one message",
    ]


def test_record_that_is_not_an_object_is_reported():
    result = parse_json_text(as_json("not a conversation", valid_record("conv-0002")))
    assert conversation_ids(result) == ["conv-0002"]
    assert [str(e) for e in result.errors] == ["record 1: record must be an object"]


def test_top_level_object_is_a_file_error():
    result = parse_json_text(json.dumps(valid_record()))
    assert result.conversations == []
    assert result.errors[0].line_number is None
    assert result.errors[0].message == "top-level value must be a JSON array, got an object"


def test_invalid_json_reports_line_number_and_returns_nothing():
    result = parse_json_text('[{"conversation_id": "conv-0001",\n "customer_id": }]\n')
    assert result.conversations == []
    assert result.errors[0].line_number == 2
    assert result.errors[0].message.startswith("invalid JSON: Expecting value")


def test_empty_json_input_is_a_file_error():
    for text in ("", "   \n\t"):
        result = parse_json_text(text)
        assert result.conversations == []
        assert result.errors[0].message == "file is empty"


def test_empty_array_is_valid_and_has_no_conversations():
    result = parse_json_text("[]")
    assert result.errors == []
    assert result.conversations == []
    assert result.rows_read == 0


def test_utf8_bom_is_accepted_by_parse_json():
    result = parse_json_text("\ufeff" + as_json(valid_record()))
    assert result.errors == []
    assert conversation_ids(result) == ["conv-0001"]


def test_non_ascii_text_is_kept_unchanged():
    text = "Karibu sana — café ✓"
    record = valid_record()
    record["messages"][1]["text"] = text
    result = parse_json_text(as_json(record))
    assert result.conversations[0]["messages"][1]["text"] == text


def test_load_json_reads_utf8_file(tmp_path):
    record = valid_record()
    record["messages"][1]["text"] = "Karibu, café ✓"
    path = tmp_path / "conversations.json"
    path.write_text(as_json(record), encoding="utf-8")
    result = load_json(path)
    assert result.errors == []
    assert result.conversations[0]["messages"][1]["text"] == "Karibu, café ✓"


def test_load_json_accepts_utf8_bom(tmp_path):
    path = tmp_path / "conversations.json"
    path.write_bytes(b"\xef\xbb\xbf" + as_json(valid_record()).encode("utf-8"))
    result = load_json(path)
    assert result.errors == []
    assert conversation_ids(result) == ["conv-0001"]


def test_load_json_reports_invalid_utf8(tmp_path):
    path = tmp_path / "conversations.json"
    path.write_bytes(
        b'[{"conversation_id": "conv-0001", "customer_id": "cust-0001", '
        b'"messages": [{"role": "customer", "text": "caf\xe9"}]}]'
    )
    result = load_json(path)
    assert result.conversations == []
    assert "not valid UTF-8" in result.errors[0].message


def test_load_json_missing_file(tmp_path):
    result = load_json(tmp_path / "missing.json")
    assert result.conversations == []
    assert "file not found" in result.errors[0].message


def test_load_json_does_not_modify_source(tmp_path):
    path = tmp_path / "conversations.json"
    path.write_text(as_json(valid_record()), encoding="utf-8")
    before = path.read_bytes()
    load_json(path)
    assert path.read_bytes() == before


# JSONL


def test_jsonl_loads_one_conversation_per_line():
    result = parse_jsonl_text(as_jsonl(
        valid_record("conv-0001"),
        valid_record("conv-0002", "cust-0002"),
    ))
    assert result.errors == []
    assert result.rows_read == 2
    assert conversation_ids(result) == ["conv-0001", "conv-0002"]


def test_jsonl_skips_blank_lines_silently():
    text = as_jsonl(valid_record("conv-0001")) + "\n   \n" + as_jsonl(valid_record("conv-0002"))
    result = parse_jsonl_text(text)
    assert result.errors == []
    assert result.rows_read == 2
    assert conversation_ids(result) == ["conv-0001", "conv-0002"]


def test_jsonl_invalid_json_line_is_reported_and_others_kept():
    text = (
        as_jsonl(valid_record("conv-0001"))
        + "{this is not json}\n"
        + as_jsonl(valid_record("conv-0003", "cust-0003"))
    )
    result = parse_jsonl_text(text)
    assert conversation_ids(result) == ["conv-0001", "conv-0003"]
    assert result.rows_read == 3
    assert result.rows_rejected == 1
    assert result.errors[0].line_number == 2
    assert result.errors[0].message.startswith("invalid JSON:")


def test_jsonl_invalid_record_reports_its_line_number():
    bad = valid_record("conv-0002", "cust-0002")
    bad["messages"][1]["role"] = "bot"
    result = parse_jsonl_text(as_jsonl(valid_record("conv-0001"), bad))
    assert conversation_ids(result) == ["conv-0001"]
    assert [str(e) for e in result.errors] == [
        "line 2: messages[1].role must be 'customer' or 'agent', got 'bot'"
    ]


def test_jsonl_line_that_is_not_an_object_is_reported():
    text = "[1, 2]\n" + as_jsonl(valid_record("conv-0002"))
    result = parse_jsonl_text(text)
    assert [str(e) for e in result.errors] == ["line 1: record must be an object"]
    assert conversation_ids(result) == ["conv-0002"]


def test_jsonl_empty_input_is_a_file_error():
    for text in ("", "\n\n   \n"):
        result = parse_jsonl_text(text)
        assert result.conversations == []
        assert result.errors[0].message == "file is empty"


def test_jsonl_utf8_bom_on_first_line_is_accepted():
    result = parse_jsonl_text("\ufeff" + as_jsonl(valid_record("conv-0001")))
    assert result.errors == []
    assert conversation_ids(result) == ["conv-0001"]


def test_jsonl_line_numbers_count_blank_lines():
    result = parse_jsonl_text("\n\n{bad json}\n")
    assert result.errors[0].line_number == 3


def test_load_jsonl_reads_utf8_and_bom_files(tmp_path):
    record = valid_record()
    record["messages"][1]["text"] = "Karibu, café ✓"
    plain = tmp_path / "plain.jsonl"
    plain.write_text(as_jsonl(record), encoding="utf-8")
    with_bom = tmp_path / "bom.jsonl"
    with_bom.write_bytes(b"\xef\xbb\xbf" + as_jsonl(record).encode("utf-8"))
    for path in (plain, with_bom):
        result = load_jsonl(path)
        assert result.errors == []
        assert result.conversations[0]["messages"][1]["text"] == "Karibu, café ✓"


def test_load_jsonl_reports_invalid_utf8(tmp_path):
    path = tmp_path / "conversations.jsonl"
    path.write_bytes(
        b'{"conversation_id": "conv-0001", "customer_id": "cust-0001", '
        b'"messages": [{"role": "customer", "text": "caf\xe9"}]}\n'
    )
    result = load_jsonl(path)
    assert result.conversations == []
    assert "not valid UTF-8" in result.errors[0].message


def test_load_jsonl_missing_file(tmp_path):
    result = load_jsonl(tmp_path / "missing.jsonl")
    assert "file not found" in result.errors[0].message


def test_load_jsonl_does_not_modify_source(tmp_path):
    path = tmp_path / "conversations.jsonl"
    path.write_text(as_jsonl(valid_record()), encoding="utf-8")
    before = path.read_bytes()
    load_jsonl(path)
    assert path.read_bytes() == before


# Same canonical output across CSV, JSON, and JSONL


def test_json_jsonl_and_csv_produce_identical_conversations():
    csv_text = (
        "conversation_id,customer_id,role,text\n"
        f"conv-0001,cust-0001,customer,{KETTLE_QUESTION}\n"
        'conv-0001,cust-0001,agent,"Yes, it does."\n'
    )
    from_csv = parse_csv(io.StringIO(csv_text), source="test.csv")
    from_json = parse_json_text(as_json(valid_record()))
    from_jsonl = parse_jsonl_text(as_jsonl(valid_record()))
    assert from_csv.errors == from_json.errors == from_jsonl.errors == []
    assert from_csv.conversations == from_json.conversations == from_jsonl.conversations
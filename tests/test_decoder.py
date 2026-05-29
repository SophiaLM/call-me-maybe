import json
import tempfile

import numpy as np
import pytest

from src.decoder import (
    ConstrainedDecoder,
    _get_expected_first_chars,
    _get_partial_prefix_state,
    _is_valid_json_prefix,
    _matches_function_schema,
    _validate_complete_schema,
    _validate_incomplete_schema,
)
from src.models import FunctionDefinition, ParameterDefinition, ReturnDefinition
from src.vocabulary import Vocabulary


SAMPLE_VOCAB = {
    "0": "<unk>",
    "1": "<s>",
    "2": "</s>",
    "3": " ",
}
for code in range(32, 127):
    SAMPLE_VOCAB[str(len(SAMPLE_VOCAB))] = chr(code)


@pytest.fixture
def vocab():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(SAMPLE_VOCAB, f)
        path = f.name
    v = Vocabulary(path)
    import os
    os.unlink(path)
    return v


def make_fn_map():
    return {
        "fn_add_numbers": FunctionDefinition(
            name="fn_add_numbers",
            description="Add two numbers",
            parameters={
                "a": ParameterDefinition(type="number"),
                "b": ParameterDefinition(type="number"),
            },
            returns=ReturnDefinition(type="number"),
        ),
        "fn_greet": FunctionDefinition(
            name="fn_greet",
            description="Greet someone",
            parameters={
                "name": ParameterDefinition(type="string"),
            },
            returns=ReturnDefinition(type="string"),
        ),
    }


class TestIsValidJsonPrefix:
    def test_empty_string(self):
        assert _is_valid_json_prefix("")

    def test_whitespace_only(self):
        assert _is_valid_json_prefix("   ")

    def test_valid_complete_json(self):
        assert _is_valid_json_prefix('{"a": 1}')

    def test_incomplete_object(self):
        assert _is_valid_json_prefix('{"a"')

    def test_invalid_start(self):
        assert not _is_valid_json_prefix("x")

    def test_unclosed_string(self):
        assert _is_valid_json_prefix('{"key": "unclosed')

    def test_incomplete_keyword(self):
        assert _is_valid_json_prefix("t")
        assert _is_valid_json_prefix("tr")
        assert _is_valid_json_prefix("n")
        assert _is_valid_json_prefix("f")

    def test_numeric_prefix(self):
        assert _is_valid_json_prefix("1")
        assert _is_valid_json_prefix("-")
        assert _is_valid_json_prefix("1.5")


class TestGetPartialPrefixState:
    def test_empty(self):
        state = _get_partial_prefix_state("")
        assert state["brace_depth"] == 0
        assert not state["in_string"]

    def test_open_brace(self):
        state = _get_partial_prefix_state("{")
        assert state["brace_depth"] == 1
        assert state["last_significant"] == "{"

    def test_inside_key_string(self):
        state = _get_partial_prefix_state('{"f')
        assert state["in_string"]
        assert state["brace_depth"] == 1
        assert state["reading_value"] is False
        assert state["current_key"] == "f"

    def test_inside_value_string(self):
        state = _get_partial_prefix_state('{"function": "fn')
        assert state["in_string"]
        assert state["reading_value"] is True
        assert state["current_key"] == "fn"
        assert state["last_key_at_depth"].get(1) == "function"

    def test_completed_key_value(self):
        state = _get_partial_prefix_state('{"function":"fn_add_numbers","arguments":{')
        assert not state["in_string"]
        assert state["brace_depth"] == 2
        assert "function" in state.get("keys_at_level", {}).get(1, [])
        assert "arguments" in state.get("keys_at_level", {}).get(1, [])

    def test_after_colon(self):
        state = _get_partial_prefix_state('{"function":')
        assert state["after_colon"] is True

    def test_closing_brace(self):
        state = _get_partial_prefix_state('{"function":"fn","arguments":{}}')
        assert state["brace_depth"] == 0


class TestGetExpectedFirstChars:
    def test_empty_state(self):
        state = {"in_string": False, "last_significant": "", "brace_depth": 0}
        chars = _get_expected_first_chars(state)
        assert chars == "{"

    def test_after_open_brace(self):
        state = {"in_string": False, "last_significant": "{", "brace_depth": 1}
        chars = _get_expected_first_chars(state)
        assert chars == '"'

    def test_closing_brace_depth0(self):
        state = {"in_string": False, "last_significant": "}", "brace_depth": 0}
        chars = _get_expected_first_chars(state)
        assert chars == ""

    def test_function_key_first(self):
        state = {
            "in_string": False, "last_significant": '"', "brace_depth": 1,
            "keys_at_level": {1: []}, "reading_value": False,
        }
        chars = _get_expected_first_chars(state)
        assert chars == ":"

    def test_in_string_key_depth1_first_char(self):
        state = {
            "in_string": True, "brace_depth": 1,
            "reading_value": False, "current_key": "",
            "keys_at_level": {1: []},
        }
        chars = _get_expected_first_chars(state)
        assert chars == "f"

    def test_in_string_key_depth1_second_key(self):
        state = {
            "in_string": True, "brace_depth": 1,
            "reading_value": False, "current_key": "",
            "keys_at_level": {1: ["function"]},
        }
        chars = _get_expected_first_chars(state)
        assert chars == "a"

    def test_in_string_key_depth2(self):
        fn_map = make_fn_map()
        state = {
            "in_string": True, "brace_depth": 2,
            "reading_value": False, "current_key": "",
            "keys_at_level": {1: ["function", "arguments"], 2: []},
            "completed_values": {1: {"function": "fn_add_numbers"}},
        }
        chars = _get_expected_first_chars(state, fn_map)
        assert "a" in chars or "b" in chars

    def test_in_string_value_function_name(self):
        state = {
            "in_string": True, "brace_depth": 1,
            "reading_value": True, "current_key": "",
            "last_key_at_depth": {1: "function"},
        }
        chars = _get_expected_first_chars(state)
        assert '"' in chars
        assert "f" in chars

    def test_after_comma(self):
        state = {
            "in_string": False, "last_significant": ",", "brace_depth": 1,
        }
        chars = _get_expected_first_chars(state)
        assert '"' in chars

    def test_after_close_brace_depth1_complete(self):
        state = {
            "in_string": False, "last_significant": "}", "brace_depth": 1,
            "keys_at_level": {1: ["function", "arguments"]},
        }
        chars = _get_expected_first_chars(state)
        assert chars == "}"

    def test_after_colon_number_param(self):
        fn_map = make_fn_map()
        state = {
            "in_string": False, "last_significant": ":", "brace_depth": 2,
            "completed_values": {1: {"function": "fn_add_numbers"}},
            "last_key_at_depth": {2: "a"},
        }
        chars = _get_expected_first_chars(state, fn_map)
        assert "0" in chars
        assert '"' not in chars


class TestValidateCompleteSchema:
    def test_valid_function_call(self):
        fn_map = make_fn_map()
        obj = {"function": "fn_add_numbers", "arguments": {"a": 1, "b": 2}}
        assert _validate_complete_schema(obj, fn_map)

    def test_not_a_dict(self):
        fn_map = make_fn_map()
        assert not _validate_complete_schema([], fn_map)

    def test_extra_keys(self):
        fn_map = make_fn_map()
        obj = {"function": "fn_add_numbers", "arguments": {}, "extra": 1}
        assert not _validate_complete_schema(obj, fn_map)

    def test_unknown_function(self):
        fn_map = make_fn_map()
        obj = {"function": "unknown", "arguments": {}}
        assert not _validate_complete_schema(obj, fn_map)

    def test_wrong_arg_type(self):
        fn_map = make_fn_map()
        obj = {"function": "fn_add_numbers", "arguments": {"a": "string", "b": 2}}
        assert not _validate_complete_schema(obj, fn_map)

    def test_extra_args(self):
        fn_map = make_fn_map()
        obj = {"function": "fn_add_numbers", "arguments": {"x": 1}}
        assert not _validate_complete_schema(obj, fn_map)


class TestValidateIncompleteSchema:
    def test_valid_partial(self):
        fn_map = make_fn_map()
        assert _validate_incomplete_schema('{"function": "fn_add_n', fn_map)

    def test_invalid_key_prefix(self):
        fn_map = make_fn_map()
        assert not _validate_incomplete_schema('{"funcx', fn_map)

    def test_invalid_function_name(self):
        fn_map = make_fn_map()
        assert not _validate_incomplete_schema('{"function": "invalid_fn', fn_map)

    def test_string_for_arguments(self):
        fn_map = make_fn_map()
        assert not _validate_incomplete_schema(
            '{"function": "fn_add_numbers", "arguments": "',
            fn_map,
        )

    def test_invalid_param_name(self):
        fn_map = make_fn_map()
        assert not _validate_incomplete_schema(
            '{"function": "fn_add_numbers", "arguments": {"x',
            fn_map,
        )

    def test_keys_out_of_order(self):
        fn_map = make_fn_map()
        assert not _validate_incomplete_schema('{"arguments', fn_map)

    def test_empty_function_name(self):
        fn_map = make_fn_map()
        assert not _validate_incomplete_schema(
            '{"function": ""}', fn_map,
        )

    def test_duplicate_param(self):
        fn_map = make_fn_map()
        assert not _validate_incomplete_schema(
            '{"function": "fn_add_numbers", "arguments": {"a": 1, "a": 2}}',
            fn_map,
        )

    def test_empty_key(self):
        fn_map = make_fn_map()
        assert not _validate_incomplete_schema('{"":', fn_map)


class TestMatchesFunctionSchema:
    def test_valid_complete(self):
        fn_map = make_fn_map()
        assert _matches_function_schema(
            '{"function": "fn_add_numbers", "arguments": {"a": 1, "b": 2}}',
            fn_map,
        )

    def test_invalid_partial(self):
        fn_map = make_fn_map()
        assert not _matches_function_schema('{"function": "x', fn_map)

    def test_valid_partial(self):
        fn_map = make_fn_map()
        assert _matches_function_schema('{"function": "fn_add_n', fn_map)


class TestConstrainedDecoder:
    def test_init(self):
        fn_map = make_fn_map()
        fn_defs = list(fn_map.values())
        decoder = ConstrainedDecoder(fn_defs)
        assert len(decoder.fn_map) == 2

    def test_reset(self):
        fn_map = make_fn_map()
        fn_defs = list(fn_map.values())
        decoder = ConstrainedDecoder(fn_defs)
        decoder.partial = "some text"
        decoder.reset()
        assert decoder.partial == ""

    def test_get_valid_token_ids_first_token(self, vocab):
        fn_map = make_fn_map()
        fn_defs = list(fn_map.values())
        decoder = ConstrainedDecoder(fn_defs)
        logits = np.full(vocab.size(), -1e9, dtype=np.float64)
        logits[vocab.string_to_token_ids("{")[0]] = 0.0
        masked = decoder.get_valid_token_ids(logits, vocab)
        valid = np.where(masked > -1e8)[0]
        assert len(valid) > 0

    def test_step_until_complete(self, vocab):
        fn_map = make_fn_map()
        fn_defs = list(fn_map.values())
        decoder = ConstrainedDecoder(fn_defs)
        decoder.reset()

        logits = np.full(vocab.size(), -1e9, dtype=np.float64)
        json_chars = '{"function":"fn_add_numbers","arguments":{"a":0,"b":0}}'
        for c in json_chars:
            for tid in vocab.get_tokens_by_first_char(c):
                logits[tid] = 0.0
            masked = decoder.get_valid_token_ids(logits, vocab)
            next_token = int(np.argmax(masked))
            if next_token == decoder.eos_token_id:
                break
            token_str = vocab.token_to_string(next_token)
            if not decoder.step(next_token, token_str):
                break
        assert decoder.partial == json_chars

    def test_is_complete_call(self, vocab):
        fn_map = make_fn_map()
        fn_defs = list(fn_map.values())
        decoder = ConstrainedDecoder(fn_defs)
        decoder.partial = '{"function":"fn_add_numbers","arguments":{"a":0,"b":0}}'
        assert decoder._is_complete_call(decoder.partial)

    def test_is_not_complete_call(self, vocab):
        fn_map = make_fn_map()
        fn_defs = list(fn_map.values())
        decoder = ConstrainedDecoder(fn_defs)
        decoder.partial = '{"function":"fn_add_numb'
        assert not decoder._is_complete_call(decoder.partial)

    def test_eos_stops_generation(self, vocab):
        fn_map = make_fn_map()
        fn_defs = list(fn_map.values())
        decoder = ConstrainedDecoder(fn_defs)
        result = decoder.step(decoder.eos_token_id, "")
        assert result is False

    def test_long_partial_stops(self, vocab):
        fn_map = make_fn_map()
        fn_defs = list(fn_map.values())
        decoder = ConstrainedDecoder(fn_defs)
        decoder.partial = "x" * 501
        result = decoder.step(0, "y")
        assert result is False

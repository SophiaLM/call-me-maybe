import json
import tempfile

import pytest

from src.errors import ValidationError
from src.models import FunctionDefinition, ParameterDefinition, ReturnDefinition
from src.validator import validate_function_call_json, validate_output_file


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


class TestValidateFunctionCallJson:
    def test_valid_function_call(self):
        fn_map = make_fn_map()
        result = validate_function_call_json(
            '{"function": "fn_add_numbers", "arguments": {"a": 1, "b": 2}}',
            fn_map,
        )
        assert result.function == "fn_add_numbers"
        assert result.arguments == {"a": 1, "b": 2}

    def test_empty_json(self):
        fn_map = make_fn_map()
        with pytest.raises(ValidationError, match="Empty"):
            validate_function_call_json("", fn_map)

    def test_invalid_json_syntax(self):
        fn_map = make_fn_map()
        with pytest.raises(ValidationError, match="Invalid JSON"):
            validate_function_call_json("{invalid}", fn_map)

    def test_non_object_json(self):
        fn_map = make_fn_map()
        with pytest.raises(ValidationError, match="object"):
            validate_function_call_json("[]", fn_map)

    def test_missing_required_keys(self):
        fn_map = make_fn_map()
        with pytest.raises(ValidationError, match="Missing"):
            validate_function_call_json('{"function": "fn_add_numbers"}', fn_map)

    def test_extra_keys(self):
        fn_map = make_fn_map()
        with pytest.raises(ValidationError, match="Unexpected"):
            validate_function_call_json(
                '{"function": "fn_add_numbers", "arguments": {}, "extra": 1}',
                fn_map,
            )

    def test_function_not_string(self):
        fn_map = make_fn_map()
        with pytest.raises(ValidationError, match="string"):
            validate_function_call_json(
                '{"function": 42, "arguments": {}}', fn_map,
            )

    def test_unknown_function(self):
        fn_map = make_fn_map()
        with pytest.raises(ValidationError, match="Unknown"):
            validate_function_call_json(
                '{"function": "nonexistent", "arguments": {}}', fn_map,
            )

    def test_arguments_not_object(self):
        fn_map = make_fn_map()
        with pytest.raises(ValidationError, match="object"):
            validate_function_call_json(
                '{"function": "fn_add_numbers", "arguments": "invalid"}',
                fn_map,
            )

    def test_extra_arguments(self):
        fn_map = make_fn_map()
        with pytest.raises(ValidationError, match="Unknown arguments"):
            validate_function_call_json(
                '{"function": "fn_add_numbers", "arguments": {"x": 1}}',
                fn_map,
            )

    def test_wrong_argument_type(self):
        fn_map = make_fn_map()
        with pytest.raises(ValidationError, match="type"):
            validate_function_call_json(
                '{"function": "fn_add_numbers", "arguments": {"a": "not_a_number", "b": 2}}',
                fn_map,
            )

    def test_string_argument(self):
        fn_map = make_fn_map()
        result = validate_function_call_json(
            '{"function": "fn_greet", "arguments": {"name": "world"}}',
            fn_map,
        )
        assert result.arguments["name"] == "world"


class TestValidateOutputFile:
    def test_file_not_found(self):
        with pytest.raises(ValidationError, match="not found"):
            validate_output_file("/nonexistent/output.json")

    def test_invalid_json_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("not json")
            path = f.name
        try:
            with pytest.raises(ValidationError, match="invalid JSON"):
                validate_output_file(path)
        finally:
            import os
            os.unlink(path)

    def test_valid_output_file(self):
        data = [
            {"prompt": "test", "fn_name": "fn_add", "args": {"a": 1}},
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            path = f.name
        try:
            result = validate_output_file(path)
            assert len(result) == 1
            assert result[0]["prompt"] == "test"
        finally:
            import os
            os.unlink(path)

    def test_not_a_list(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({"not": "list"}, f)
            path = f.name
        try:
            with pytest.raises(ValidationError, match="array"):
                validate_output_file(path)
        finally:
            import os
            os.unlink(path)

    def test_empty_array(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump([], f)
            path = f.name
        try:
            with pytest.raises(ValidationError, match="empty"):
                validate_output_file(path)
        finally:
            import os
            os.unlink(path)

    def test_entry_missing_keys(self):
        data = [{"prompt": "test"}]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            path = f.name
        try:
            with pytest.raises(ValidationError, match="missing keys"):
                validate_output_file(path)
        finally:
            import os
            os.unlink(path)

import json
import os
import tempfile

import pytest

from src.errors import LoaderError
from src.loader import load_function_definitions, load_prompts


SAMPLE_FUNCTIONS = [
    {
        "name": "fn_add_numbers",
        "description": "Add two numbers",
        "parameters": {
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "returns": {"type": "number"},
    },
]

SAMPLE_TESTS = [
    {"prompt": "What is 2+2?"},
    {"prompt": "Greet Alice"},
]


@pytest.fixture
def temp_input_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        (yield tmpdir)


class TestLoadFunctionDefinitions:
    def test_load_from_functions_definition_json(self, temp_input_dir):
        path = os.path.join(temp_input_dir, "functions_definition.json")
        with open(path, "w") as f:
            json.dump(SAMPLE_FUNCTIONS, f)
        result = load_function_definitions(temp_input_dir)
        assert "fn_add_numbers" in result
        assert result["fn_add_numbers"].name == "fn_add_numbers"

    def test_load_from_function_definitions_json(self, temp_input_dir):
        path = os.path.join(temp_input_dir, "function_definitions.json")
        with open(path, "w") as f:
            json.dump(SAMPLE_FUNCTIONS, f)
        result = load_function_definitions(temp_input_dir)
        assert "fn_add_numbers" in result

    def test_load_from_functions_json(self, temp_input_dir):
        path = os.path.join(temp_input_dir, "functions.json")
        with open(path, "w") as f:
            json.dump(SAMPLE_FUNCTIONS, f)
        result = load_function_definitions(temp_input_dir)
        assert "fn_add_numbers" in result

    def test_no_definitions_file(self, temp_input_dir):
        with pytest.raises(LoaderError, match="No definitions file"):
            load_function_definitions(temp_input_dir)

    def test_invalid_json(self, temp_input_dir):
        path = os.path.join(temp_input_dir, "functions_definition.json")
        with open(path, "w") as f:
            f.write("not json")
        with pytest.raises(LoaderError, match="Invalid JSON"):
            load_function_definitions(temp_input_dir)

    def test_not_a_list(self, temp_input_dir):
        path = os.path.join(temp_input_dir, "functions_definition.json")
        with open(path, "w") as f:
            json.dump({"not": "list"}, f)
        with pytest.raises(LoaderError, match="array"):
            load_function_definitions(temp_input_dir)

    def test_entry_not_dict(self, temp_input_dir):
        path = os.path.join(temp_input_dir, "functions_definition.json")
        with open(path, "w") as f:
            json.dump(["not a dict"], f)
        with pytest.raises(LoaderError, match="not a JSON object"):
            load_function_definitions(temp_input_dir)

    def test_entry_no_name(self, temp_input_dir):
        path = os.path.join(temp_input_dir, "functions_definition.json")
        with open(path, "w") as f:
            json.dump([{"description": "no name"}], f)
        with pytest.raises(LoaderError, match="no 'name' field"):
            load_function_definitions(temp_input_dir)

    def test_duplicate_function(self, temp_input_dir):
        path = os.path.join(temp_input_dir, "functions_definition.json")
        with open(path, "w") as f:
            json.dump([SAMPLE_FUNCTIONS[0], SAMPLE_FUNCTIONS[0]], f)
        with pytest.raises(LoaderError, match="Duplicate"):
            load_function_definitions(temp_input_dir)

    def test_empty_functions(self, temp_input_dir):
        path = os.path.join(temp_input_dir, "functions_definition.json")
        with open(path, "w") as f:
            json.dump([], f)
        with pytest.raises(LoaderError, match="No function definitions"):
            load_function_definitions(temp_input_dir)

    def test_input_dir_not_found(self):
        with pytest.raises(LoaderError, match="not found"):
            load_function_definitions("/nonexistent/dir")


class TestLoadPrompts:
    def test_load_prompts(self, temp_input_dir):
        path = os.path.join(temp_input_dir, "function_calling_tests.json")
        with open(path, "w") as f:
            json.dump(SAMPLE_TESTS, f)
        result = load_prompts(temp_input_dir)
        assert len(result) == 2
        assert result[0].prompt == "What is 2+2?"

    def test_file_not_found(self, temp_input_dir):
        with pytest.raises(LoaderError, match="not found"):
            load_prompts(temp_input_dir)

    def test_invalid_json(self, temp_input_dir):
        path = os.path.join(temp_input_dir, "function_calling_tests.json")
        with open(path, "w") as f:
            f.write("not json")
        with pytest.raises(LoaderError, match="Invalid JSON"):
            load_prompts(temp_input_dir)

    def test_not_a_list(self, temp_input_dir):
        path = os.path.join(temp_input_dir, "function_calling_tests.json")
        with open(path, "w") as f:
            json.dump({"not": "list"}, f)
        with pytest.raises(LoaderError, match="array"):
            load_prompts(temp_input_dir)

    def test_string_entries(self, temp_input_dir):
        path = os.path.join(temp_input_dir, "function_calling_tests.json")
        with open(path, "w") as f:
            json.dump(["Just a string prompt"], f)
        result = load_prompts(temp_input_dir)
        assert len(result) == 1
        assert result[0].prompt == "Just a string prompt"

    def test_dict_entry_no_prompt(self, temp_input_dir):
        path = os.path.join(temp_input_dir, "function_calling_tests.json")
        with open(path, "w") as f:
            json.dump([{"not_prompt": "value"}], f)
        with pytest.raises(LoaderError, match="no 'prompt' field"):
            load_prompts(temp_input_dir)

    def test_unexpected_type(self, temp_input_dir):
        path = os.path.join(temp_input_dir, "function_calling_tests.json")
        with open(path, "w") as f:
            json.dump([42], f)
        with pytest.raises(LoaderError, match="unexpected type"):
            load_prompts(temp_input_dir)

    def test_empty_prompts(self, temp_input_dir):
        path = os.path.join(temp_input_dir, "function_calling_tests.json")
        with open(path, "w") as f:
            json.dump([], f)
        with pytest.raises(LoaderError, match="No prompts found"):
            load_prompts(temp_input_dir)

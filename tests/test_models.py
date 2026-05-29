from src.models import (
    FunctionDefinition,
    FunctionCallOutput,
    ParameterDefinition,
    PromptEntry,
    ReturnDefinition,
    type_matches,
)


class TestTypeMatches:
    def test_string_type(self):
        assert type_matches("hello", "string")
        assert not type_matches(123, "string")

    def test_number_type(self):
        assert type_matches(42, "number")
        assert type_matches(3.14, "number")
        assert not type_matches("hello", "number")

    def test_boolean_type(self):
        assert type_matches(True, "boolean")
        assert type_matches(False, "boolean")
        assert not type_matches(1, "boolean")

    def test_integer_type(self):
        assert type_matches(42, "integer")
        assert not type_matches(3.14, "integer")
        assert not type_matches("42", "integer")

    def test_object_type(self):
        assert type_matches({"key": "val"}, "object")
        assert not type_matches("not object", "object")

    def test_array_type(self):
        assert type_matches([1, 2, 3], "array")
        assert not type_matches("not array", "array")

    def test_null_type(self):
        assert type_matches(None, "null")
        assert not type_matches(0, "null")

    def test_unknown_type(self):
        assert type_matches("anything", "unknown_type")


class TestParameterDefinition:
    def test_create(self):
        p = ParameterDefinition(type="string")
        assert p.type == "string"


class TestReturnDefinition:
    def test_create(self):
        r = ReturnDefinition(type="number")
        assert r.type == "number"


class TestFunctionDefinition:
    def test_create_minimal(self):
        fn = FunctionDefinition(name="test_fn", description="A test function")
        assert fn.name == "test_fn"
        assert fn.description == "A test function"
        assert fn.parameters == {}
        assert fn.returns is None

    def test_create_with_params(self):
        fn = FunctionDefinition(
            name="fn_add",
            description="Add two numbers",
            parameters={
                "a": ParameterDefinition(type="number"),
                "b": ParameterDefinition(type="number"),
            },
            returns=ReturnDefinition(type="number"),
        )
        assert fn.name == "fn_add"
        assert len(fn.parameters) == 2
        assert fn.parameters["a"].type == "number"


class TestPromptEntry:
    def test_create(self):
        entry = PromptEntry(prompt="What is 2+2?")
        assert entry.prompt == "What is 2+2?"


class TestFunctionCallOutput:
    def test_create_valid(self):
        result = FunctionCallOutput(
            function="fn_add", arguments={"a": 1, "b": 2}
        )
        assert result.function == "fn_add"
        assert result.arguments == {"a": 1, "b": 2}

    def test_empty_name_raises(self):
        try:
            FunctionCallOutput(function="", arguments={})
            assert False, "Should have raised"
        except ValueError:
            pass

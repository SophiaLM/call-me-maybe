from src.__main__ import build_results
from src.models import FunctionCallOutput, FunctionDefinition, ParameterDefinition, PromptEntry, ReturnDefinition


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
    }


class TestBuildResults:
    def test_build_results(self):
        fn_map = make_fn_map()
        prompts = [PromptEntry(prompt="What is 2+2?")]
        generated = [
            (prompts[0], FunctionCallOutput(function="fn_add_numbers", arguments={"a": 1, "b": 2})),
        ]
        results = build_results(fn_map, prompts, generated)
        assert len(results) == 1
        assert results[0]["prompt"] == "What is 2+2?"
        assert results[0]["fn_name"] == "fn_add_numbers"
        assert results[0]["args"] == {"a": 1, "b": 2}

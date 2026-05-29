import json
import tempfile

import numpy as np
import pytest

from src.decoder import ConstrainedDecoder
from src.errors import GeneratorError
from src.generator import build_prompt, generate_function_call, process_prompts
from src.models import FunctionDefinition, ParameterDefinition, PromptEntry, ReturnDefinition
from src.vocabulary import Vocabulary


SAMPLE_VOCAB = {
    "0": "<unk>",
    "1": "<s>",
    "2": "</s>",
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
    }


class MockModel:
    def __init__(self, vocab):
        self._vocab = vocab

    def encode(self, text):
        return [0]

    def get_logits_from_input_ids(self, input_ids):
        logits = np.full(self._vocab.size(), -1e9, dtype=np.float64)
        for tid in range(self._vocab.size()):
            token_str = self._vocab.token_to_string(tid)
            if not token_str:
                continue
            c = token_str[0]
            score = 0.0
            if c in '{"}:,':
                score = 3.0
            elif c == '"':
                score = 3.5
            elif c.isalpha():
                score = 1.0
            elif c.isdigit():
                score = 0.5
            elif c == '_':
                score = 2.0
            else:
                score = -1.0
            logits[tid] = score
        return logits


class TestBuildPrompt:
    def test_build_prompt(self):
        fn_map = make_fn_map()
        entry = PromptEntry(prompt="What is 2+2?")
        prompt = build_prompt(entry, fn_map)
        assert "fn_add_numbers" in prompt
        assert "What is 2+2?" in prompt
        assert "number" in prompt

    def test_prompt_contains_function_json(self):
        fn_map = make_fn_map()
        entry = PromptEntry(prompt="test")
        prompt = build_prompt(entry, fn_map)
        assert '"name": "fn_add_numbers"' in prompt
        assert '"type": "number"' in prompt


class TestGenerateFunctionCall:
    def test_generates_valid_call(self, vocab):
        fn_defs = list(make_fn_map().values())
        decoder = ConstrainedDecoder(fn_defs)
        model = MockModel(vocab)
        result = generate_function_call(
            model=model,
            prompt="test prompt",
            decoder=decoder,
            vocab=vocab,
        )
        parsed = json.loads(result)
        assert "function" in parsed
        assert "arguments" in parsed
        assert isinstance(parsed["arguments"], dict)

    def test_generation_error_on_encode_failure(self, vocab):
        class FailingModel:
            def encode(self, text):
                raise RuntimeError("encode failed")
        fn_defs = list(make_fn_map().values())
        decoder = ConstrainedDecoder(fn_defs)
        with pytest.raises(GeneratorError, match="encode"):
            generate_function_call(
                model=FailingModel(),
                prompt="test",
                decoder=decoder,
                vocab=vocab,
            )


class TestProcessPrompts:
    def test_process_single_prompt(self, vocab):
        fn_defs = list(make_fn_map().values())
        prompts = [PromptEntry(prompt="What is 2+2?")]
        model = MockModel(vocab)
        results = process_prompts(model, prompts, fn_defs, vocab)
        assert len(results) == 1
        prompt_entry, func_call = results[0]
        assert prompt_entry.prompt == "What is 2+2?"
        assert func_call.function in make_fn_map()
        assert isinstance(func_call.arguments, dict)

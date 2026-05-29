"""LLM interaction and function call generation."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.decoder import ConstrainedDecoder
from src.errors import GeneratorError
from src.models import FunctionCallOutput, FunctionDefinition, PromptEntry
from src.validator import validate_function_call_json
from src.vocabulary import Vocabulary


SYSTEM_PROMPT_TEMPLATE = (
    "You are a function calling system. Given a user request, you must"
    " respond with a JSON object containing the function name and"
    " arguments.\n\n"
    "Available functions:\n"
    "{FUNCTIONS_JSON}\n\n"
    "Rules:\n"
    "- Choose the most appropriate function based on the user request.\n"
    "- The function name must be one of the available functions.\n"
    "- Provide all required arguments with correct types.\n"
    "- Respond ONLY with the JSON object, no other text.\n\n"
    "User request: {USER_PROMPT}\n\n"
    "Function call:"
)


def build_prompt(prompt_entry: PromptEntry,
                 fn_map: Dict[str, FunctionDefinition]) -> str:
    """Build the full prompt for the LLM.

    Constructs a prompt that includes the system instructions,
    available function definitions (as JSON), and the user's
    natural language request.

    Args:
        prompt_entry: The user's prompt/question.
        fn_map: Map of available function definitions.

    Returns:
        Formatted prompt string ready for LLM input.
    """
    fn_list = []
    for fn_name, fn_def in fn_map.items():
        fn_entry = {
            "name": fn_def.name,
            "description": fn_def.description,
            "parameters": {
                p_name: {"type": p_def.type}
                for p_name, p_def in fn_def.parameters.items()
            },
            "returns": {
                "type": fn_def.returns.type
            } if fn_def.returns else None,
        }
        fn_list.append(fn_entry)

    functions_json = json.dumps(fn_list, indent=2)

    return SYSTEM_PROMPT_TEMPLATE.format(
        FUNCTIONS_JSON=functions_json,
        USER_PROMPT=prompt_entry.prompt,
    )


def generate_function_call(
    model: Any,
    prompt: str,
    decoder: ConstrainedDecoder,
    vocab: Vocabulary,
    max_new_tokens: int = 200,
) -> str:
    """Generate a function call JSON using constrained decoding.

    Runs the LLM generation loop with token-level constraints.
    At each step:
    1. Encode the current text (prompt + partial output)
    2. Get logits from the model
    3. Use the decoder to mask invalid tokens
    4. Select the next token (argmax)
    5. Check for completion

    Args:
        model: The LLM model instance with encode/get_logits_from_input_ids.
        prompt: The full prompt text.
        decoder: Initialized constrained decoder.
        vocab: Vocabulary for token-to-string mapping.
        max_new_tokens: Maximum number of new tokens to generate.

    Returns:
        Generated function call JSON string.

    Raises:
        GeneratorError: If generation fails or produces invalid output.
    """
    decoder.reset()

    try:
        input_ids = model.encode(prompt)
    except Exception as e:
        raise GeneratorError(f"Failed to encode prompt: {e}") from e

    full_ids: List[int] = list(input_ids)
    generated_tokens: List[int] = []

    for _ in range(max_new_tokens):
        try:
            logits = model.get_logits_from_input_ids(
                np.array([full_ids], dtype=np.int64),
            )
        except Exception as e:
            raise GeneratorError(
                f"Failed to get logits from model: {e}",
            ) from e

        while logits.ndim > 1:
            logits = logits[0]

        masked_logits = decoder.get_valid_token_ids(logits, vocab)

        next_token = int(np.argmax(masked_logits))

        if next_token == decoder.eos_token_id:
            break

        token_str = vocab.token_to_string(next_token)

        generated_tokens.append(next_token)
        full_ids.append(next_token)

        should_continue = decoder.step(next_token, token_str)
        if not should_continue:
            break

    result = decoder.partial

    if not result.strip():
        raise GeneratorError(
            "Generation produced empty output",
        )

    return result


def process_prompts(
    model: object,
    prompts: List[PromptEntry],
    fn_defs: List[FunctionDefinition],
    vocab: Vocabulary,
    decoder: Optional[ConstrainedDecoder] = None,
) -> List[Tuple[PromptEntry, FunctionCallOutput]]:
    """Process all prompts and generate function calls.

    For each prompt, builds the LLM prompt, generates a constrained
    function call, and parses the result.

    Args:
        model: The LLM model instance.
        prompts: List of prompt entries to process.
        fn_defs: List of function definitions.
        vocab: Vocabulary for the model.
        decoder: Optional pre-configured decoder. Created if not provided.

    Returns:
        List of (prompt_entry, function_call_output) tuples.

    Raises:
        GeneratorError: If all prompts fail to generate.
    """
    fn_map = {f.name: f for f in fn_defs}

    if decoder is None:
        decoder = ConstrainedDecoder(fn_defs)
        decoder.set_vocabulary(vocab)

    results: List[Tuple[PromptEntry, FunctionCallOutput]] = []

    for i, prompt_entry in enumerate(prompts):
        full_prompt = build_prompt(prompt_entry, fn_map)

        try:
            json_str = generate_function_call(
                model=model,
                prompt=full_prompt,
                decoder=decoder,
                vocab=vocab,
            )
        except GeneratorError as e:
            raise GeneratorError(
                f"Failed to generate for prompt {i} "
                f"({prompt_entry.prompt!r}): {e}",
            ) from e

        try:
            func_call = validate_function_call_json(json_str, fn_map)
        except Exception as e:
            raise GeneratorError(
                f"Validation failed for prompt {i} "
                f"({prompt_entry.prompt!r}): {e}",
            ) from e

        results.append((prompt_entry, func_call))

    return results

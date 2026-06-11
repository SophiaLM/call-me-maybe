"""Output validation utilities for function call results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.errors import ValidationError
from src.models import FunctionCallOutput, FunctionDefinition, type_matches


def validate_function_call_json(
    json_str: str,
    fn_map: Dict[str, FunctionDefinition],
) -> FunctionCallOutput:
    """Parse and validate a raw JSON string as a function call.

    Checks that the JSON:
    - Is valid JSON syntax
    - Has 'function' and 'arguments' keys
    - 'function' is a valid function name
    - 'arguments' is an object with valid keys/types

    Args:
        json_str: Raw JSON string to validate.
        fn_map: Map of valid function names to definitions.

    Returns:
        Parsed FunctionCallOutput.

    Raises:
        ValidationError: If the JSON is invalid or doesn't match schema.
    """
    if not json_str or not json_str.strip():
        raise ValidationError("Empty or blank function call JSON")

    try:
        obj = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON syntax: {e}") from e

    if not isinstance(obj, dict):
        raise ValidationError(
            f"Expected JSON object, got {type(obj).__name__}",
        )

    required_keys = {"function", "arguments"}
    missing = required_keys - set(obj.keys())
    if missing:
        raise ValidationError(
            f"Missing required keys: {', '.join(sorted(missing))}",
        )

    extra_keys = set(obj.keys()) - required_keys
    if extra_keys:
        raise ValidationError(
            f"Unexpected keys: {', '.join(sorted(extra_keys))}",
        )

    if not isinstance(obj["function"], str):
        raise ValidationError(
            "'function' must be a string, "
            f"got {type(obj['function']).__name__}",
        )

    fn_name: str = obj["function"]
    if fn_name not in fn_map:
        valid_names = sorted(fn_map.keys())
        raise ValidationError(
            f"Unknown function '{fn_name}'. "
            f"Valid functions: {', '.join(valid_names)}",
        )

    if not isinstance(obj["arguments"], dict):
        raise ValidationError(
            f"'arguments' must be an object, "
            f"got {type(obj['arguments']).__name__}",
        )

    fn_def = fn_map[fn_name]
    args_raw: Dict[str, Any] = obj["arguments"]

    valid_arg_keys = set(fn_def.parameters.keys())
    actual_arg_keys = set(args_raw.keys())

    extra_args = actual_arg_keys - valid_arg_keys
    if extra_args:
        raise ValidationError(
            f"Unknown arguments for '{fn_name}': "
            f"{', '.join(sorted(extra_args))}. "
            f"Valid: {', '.join(sorted(valid_arg_keys))}",
        )

    for arg_name, arg_value in args_raw.items():
        if arg_name in fn_def.parameters:
            expected_type = fn_def.parameters[arg_name].type
            if not type_matches(arg_value, expected_type):
                raise ValidationError(
                    f"Argument '{arg_name}' expects type "
                    f"'{expected_type}', got value {arg_value!r}",
                )

    return FunctionCallOutput(function=fn_name, arguments=args_raw)


def validate_output_file(path: str) -> List[Dict[str, Any]]:
    """Validate a generated output JSON file.

    Reads and parses the output file, checking structure and content.

    Args:
        path: Path to the output file.

    Returns:
        List of parsed result dictionaries.

    Raises:
        ValidationError: If the file is invalid.
    """
    filepath = Path(path)
    if not filepath.is_file():
        raise ValidationError(f"Output file not found: {path}")

    try:
        data = json.loads(filepath.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValidationError(
            f"Output file contains invalid JSON: {e}",
        ) from e
    except OSError as e:
        raise ValidationError(
            f"Error reading output file: {e}",
        ) from e

    if not isinstance(data, list):
        raise ValidationError(
            f"Output must be a JSON array, got {type(data).__name__}",
        )

    if not data:
        raise ValidationError("Output file contains an empty array")

    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValidationError(
                f"Entry {i} is not a JSON object",
            )

        required = {"prompt", "fn_name", "args"}
        missing = required - set(entry.keys())
        if missing:
            raise ValidationError(
                f"Entry {i} missing keys: {', '.join(sorted(missing))}",
            )

        if not isinstance(entry.get("prompt"), str):
            raise ValidationError(
                f"Entry {i} 'prompt' must be a string",
            )

        if not isinstance(entry.get("fn_name"), str):
            raise ValidationError(
                f"Entry {i} 'fn_name' must be a string",
            )

        if not isinstance(entry.get("args"), dict):
            raise ValidationError(
                f"Entry {i} 'args' must be a JSON object",
            )

    return data

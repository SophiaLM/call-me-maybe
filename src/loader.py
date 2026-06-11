"""Input file loading and parsing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from src.errors import LoaderError
from src.models import FunctionDefinition, PromptEntry


DEFINITIONS_FILENAMES = [
    "functions_definition.json",
    "function_definitions.json",
    "functions.json",
]


def load_function_definitions(
    input_dir: str,
) -> Dict[str, FunctionDefinition]:
    """Load function definitions from the input directory.

    Tries multiple possible filenames for the definitions file.
    Validates and indexes function definitions by name.

    Args:
        input_dir: Path to the input directory.

    Returns:
        Dictionary mapping function names to their definitions.

    Raises:
        LoaderError: If no valid definitions file is found or
            content is invalid.
    """
    dir_path = Path(input_dir)
    if not dir_path.is_dir():
        raise LoaderError(f"Input directory not found: {input_dir}")

    definitions_path: Optional[Path] = None
    for filename in DEFINITIONS_FILENAMES:
        candidate = dir_path / filename
        if candidate.is_file():
            definitions_path = candidate
            break

    if definitions_path is None:
        raise LoaderError(
            f"No definitions file found in {input_dir}. "
            f"Tried: {', '.join(DEFINITIONS_FILENAMES)}",
        )

    try:
        raw = json.loads(definitions_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise LoaderError(
            f"Invalid JSON in definitions file '{definitions_path.name}': {e}",
        ) from e
    except OSError as e:
        raise LoaderError(
            f"Error reading definitions file '{definitions_path.name}': {e}",
        ) from e

    if not isinstance(raw, list):
        raise LoaderError(
            "Definitions file must contain a JSON array, "
            f"got {type(raw).__name__}",
        )

    functions: Dict[str, FunctionDefinition] = {}
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise LoaderError(
                f"Entry {i} in definitions is not a JSON object",
            )
        if "name" not in entry:
            raise LoaderError(f"Entry {i} in definitions has no 'name' field")

        try:
            func_def = FunctionDefinition(**entry)
        except Exception as e:
            raise LoaderError(
                f"Entry {i} ('{entry.get('name', '?')}') "
                f"failed validation: {e}",
            ) from e

        if func_def.name in functions:
            raise LoaderError(
                f"Duplicate function name: '{func_def.name}' at entry {i}",
            )
        functions[func_def.name] = func_def

    if not functions:
        raise LoaderError("No function definitions found")

    return functions


def load_prompts(input_dir: str) -> List[PromptEntry]:
    """Load prompts from the function calling tests file.

    Args:
        input_dir: Path to the input directory.

    Returns:
        List of PromptEntry objects.

    Raises:
        LoaderError: If the file cannot be found, parsed, or is empty.
    """
    dir_path = Path(input_dir)
    tests_file = dir_path / "function_calling_tests.json"

    if not tests_file.is_file():
        raise LoaderError(
            f"Tests file not found: {tests_file}",
        )

    try:
        raw = json.loads(tests_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise LoaderError(
            f"Invalid JSON in tests file '{tests_file.name}': {e}",
        ) from e
    except OSError as e:
        raise LoaderError(
            f"Error reading tests file '{tests_file.name}': {e}",
        ) from e

    if not isinstance(raw, list):
        raise LoaderError(
            f"Tests file must contain a JSON array, got {type(raw).__name__}",
        )

    prompts: List[PromptEntry] = []
    for i, entry in enumerate(raw):
        if isinstance(entry, str):
            prompts.append(PromptEntry(prompt=entry))
        elif isinstance(entry, dict):
            if "prompt" not in entry:
                raise LoaderError(
                    f"Entry {i} in tests is a dict but has no 'prompt' field",
                )
            try:
                prompts.append(PromptEntry(**entry))
            except Exception as e:
                raise LoaderError(
                    f"Entry {i} failed validation: {e}",
                ) from e
        else:
            raise LoaderError(
                f"Entry {i} in tests has unexpected type: "
                f"{type(entry).__name__}",
            )

    if not prompts:
        raise LoaderError("No prompts found in tests file")

    return prompts

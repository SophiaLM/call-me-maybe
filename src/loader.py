from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

from .models import FunctionDefinition


def load_tests(path: Path) -> List[str]:
    """Load and validate the test prompts JSON file.

    Supports both an array of strings and an array of objects
    with a 'prompt' key.

    Args:
        path: Path to the JSON file.

    Returns:
        List of prompt strings.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the JSON is malformed or has wrong structure.
    """
    if not path.exists():
        raise FileNotFoundError(f"Tests file not found: {path}")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in tests file '{path}': {e}")
    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON array in '{path}', "
            f"got {type(data).__name__}"
        )
    tests: List[str] = []
    for i, item in enumerate(data):
        if isinstance(item, str):
            tests.append(item)
        elif isinstance(item, dict) and "prompt" in item:
            val = item["prompt"]
            if isinstance(val, str):
                tests.append(val)
            else:
                raise ValueError(
                    f"Item at index {i} in '{path}' has "
                    f"non-string 'prompt': {val!r}"
                )
        else:
            raise ValueError(
                f"Item at index {i} in '{path}' has "
                f"unexpected format: {item!r}"
            )
    return tests


def load_functions(path: Path) -> List[FunctionDefinition]:
    """Load and validate the function definitions JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        List of validated FunctionDefinition objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the JSON is malformed or has invalid
            function definitions.
    """
    if not path.exists():
        raise FileNotFoundError(f"Functions file not found: {path}")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in functions file '{path}': {e}"
        )
    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON array in '{path}', "
            f"got {type(data).__name__}"
        )
    functions: List[FunctionDefinition] = []
    for i, item in enumerate(data):
        try:
            fn = FunctionDefinition(**item)
        except Exception as e:
            raise ValueError(
                f"Invalid function definition at index {i} "
                f"in '{path}': {e}"
            )
        functions.append(fn)
    return functions


def resolve_input_paths(
    input_arg: str,
) -> Tuple[Path, Path]:
    """Resolve input argument to paths for tests and definitions.

    If input_arg points to a directory, looks for
    function_calling_tests.json and function_definitions.json (or
    functions_definition.json) inside.

    If input_arg points to a file, treats it as the tests file
    and looks for definitions in the same directory.

    Args:
        input_arg: Path or directory from --input argument.

    Returns:
        Tuple of (tests_path, functions_path).

    Raises:
        FileNotFoundError: If no function definitions file is found.
    """
    input_path = Path(input_arg)
    fn_candidates: List[str] = [
        "function_definitions.json",
        "functions_definition.json",
    ]
    if input_path.is_dir():
        tests_path = input_path / "function_calling_tests.json"
        fn_path: Path | None = None
        for candidate in fn_candidates:
            candidate_path = input_path / candidate
            if candidate_path.exists():
                fn_path = candidate_path
                break
        if fn_path is None:
            raise FileNotFoundError(
                f"No function definitions file found in "
                f"'{input_path}'. Tried: {fn_candidates}"
            )
    else:
        tests_path = input_path
        parent = input_path.parent
        fn_path = None
        for candidate in fn_candidates:
            candidate_path = parent / candidate
            if candidate_path.exists():
                fn_path = candidate_path
                break
        if fn_path is None:
            raise FileNotFoundError(
                f"No function definitions file found next to "
                f"'{input_path}'. Tried: {fn_candidates}"
            )
    return tests_path, fn_path


def load_all(
    input_path: str,
) -> Tuple[List[str], List[FunctionDefinition]]:
    """Convenience function to load tests and definitions.

    Args:
        input_path: Path or directory of input files.

    Returns:
        Tuple of (tests, functions).
    """
    tests_path, fn_path = resolve_input_paths(input_path)
    tests = load_tests(tests_path)
    functions = load_functions(fn_path)
    return tests, functions

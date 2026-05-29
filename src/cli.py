"""Command-line interface argument parsing."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple


DEFAULT_INPUT_DIR = os.path.join("data", "input")
DEFAULT_OUTPUT_PATH = os.path.join("data", "output", "function_calling_results.json")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the call_me_maybe tool.

    Supports:
    - --input / -i: Custom input file path (overrides default dir)
    - --output / -o: Custom output file path (overrides default)

    Args:
        argv: Argument list (defaults to sys.argv).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="Call Me Maybe - LLM Function Calling with Constrained Decoding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python -m src\n"
            "  uv run python -m src --input data/input/my_tests.json\n"
            "  uv run python -m src --output results.json\n"
        ),
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help=(
            "Path to input tests file. "
            f"Default: reads from {DEFAULT_INPUT_DIR}/"
        ),
        dest="input_path",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help=(
            "Path to output results file. "
            f"Default: {DEFAULT_OUTPUT_PATH}"
        ),
        dest="output_path",
    )

    return parser.parse_args(argv)


def resolve_paths(args: argparse.Namespace) -> Tuple[str, str]:
    """Resolve input and output paths from parsed arguments.

    If --input is provided, treats it as the input tests file path
    and derives the input directory from it.
    If --output is provided, uses it as the output file path.
    Otherwise uses defaults.

    Args:
        args: Parsed arguments from parse_args().

    Returns:
        Tuple of (input_directory, output_file_path).
    """
    if args.input_path:
        input_path = Path(args.input_path)
        input_dir = str(input_path.parent)
    else:
        input_dir = DEFAULT_INPUT_DIR
        input_path = Path(input_dir) / "function_calling_tests.json"

    output_path = args.output_path or DEFAULT_OUTPUT_PATH

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    return input_dir, output_path

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .models import FunctionCall


def write_output(
    results: List[FunctionCall], output_path: Path
) -> None:
    """Write the list of FunctionCall results to a JSON file.

    Creates the output directory if it does not exist.
    Writes to output_path / function_calling_results.json.

    Args:
        results: List of FunctionCall objects to serialize.
        output_path: Directory to write the output file in.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / "function_calling_results.json"
    data = [r.model_dump() for r in results]
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Output written to {file_path}")

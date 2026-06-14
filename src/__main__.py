from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from llm_sdk import Small_LLM_Model

from .args_generator import generate_args
from .constrained import filter_vocab
from .function_selector import select_function
from .loader import load_all
from .models import FunctionCall
from .writer import write_output


def _load_vocab(
    model: Small_LLM_Model,
) -> Dict[int, str]:
    raw_path = model.get_path_to_vocabulary_json()
    with open(raw_path, encoding="utf-8") as f:
        raw_vocab: Dict[str, int] = json.load(f)
    id_to_token: Dict[int, str] = {}
    for token_str, token_id in raw_vocab.items():
        decoded = (
            token_str.replace("\u0120", " ")
            .replace("\u010a", "\n")
        )
        id_to_token[int(token_id)] = decoded
    return id_to_token


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Function calling with constrained decoding"
    )
    parser.add_argument(
        "--input",
        default="data/input",
        help="Input directory or test file (default: data/input)",
    )
    parser.add_argument(
        "--output",
        default="data/output",
        help="Output directory (default: data/output)",
    )
    args = parser.parse_args()

    try:
        print("Loading model...", file=sys.stderr)
        model = Small_LLM_Model()
        print("Loading vocabulary...", file=sys.stderr)
        id_to_token = _load_vocab(model)
        id_to_token = filter_vocab(id_to_token)
        print("Loading test data...", file=sys.stderr)
        tests, functions = load_all(args.input)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)
    results: List[FunctionCall] = []

    total = len(tests)
    print(
        f"Processing {total} prompts...\n", file=sys.stderr
    )

    for i, prompt in enumerate(tests, 1):
        if not prompt or not prompt.strip():
            results.append(
                FunctionCall(
                    prompt=prompt, fn_name="", args={}
                )
            )
            continue

        try:
            fn_name = select_function(
                model, prompt, functions,
                id_to_token,
            )
            if fn_name is None:
                print(
                    f"WARNING: Could not select function "
                    f"for: {prompt[:50]}...",
                    file=sys.stderr,
                )
                results.append(
                    FunctionCall(
                        prompt=prompt, fn_name="", args={}
                    )
                )
                continue

            fn_def = next(
                f for f in functions if f.name == fn_name
            )
            args_json = generate_args(
                model, prompt, fn_def,
                id_to_token,
            )
            if args_json is None:
                print(
                    f"WARNING: Could not generate args for "
                    f"{fn_name} for: {prompt[:50]}...",
                    file=sys.stderr,
                )
                results.append(
                    FunctionCall(
                        prompt=prompt, fn_name=fn_name,
                        args={},
                    )
                )
                continue

            parsed_args: Dict[str, Any] = json.loads(args_json)
            results.append(
                FunctionCall(
                    prompt=prompt,
                    fn_name=fn_name,
                    args=parsed_args,
                )
            )
            print(
                f"  [{i}/{total}] {prompt[:40]}... "
                f"-> {fn_name}",
                file=sys.stderr,
            )
        except Exception as exc:
            print(
                f"ERROR processing '{prompt[:50]}...': {exc}",
                file=sys.stderr,
            )
            results.append(
                FunctionCall(
                    prompt=prompt, fn_name="", args={}
                )
            )

    try:
        write_output(results, output_path)
    except Exception as e:
        print(f"ERROR writing output: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Main entry point for the call_me_maybe project.

Orchestrates the full pipeline:
1. Parse CLI arguments
2. Load input files (function definitions + prompts)
3. Initialize the LLM model and vocabulary
4. Generate function calls using constrained decoding
5. Validate and write output

"""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any, Dict, List

from llm_sdk import Small_LLM_Model

from src.cli import parse_args, resolve_paths
from src.errors import (
    CallMeMaybeError,
    GeneratorError,
    LoaderError,
    VocabularyError,
)
from src.generator import process_prompts
from src.loader import load_function_definitions, load_prompts
from src.models import (
    FunctionDefinition,
    PromptEntry,
)
from src.vocabulary import Vocabulary


def build_results(
    fn_map: Dict[str, FunctionDefinition],
    prompts: List[PromptEntry],
    generated: List[Any],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for prompt_entry, func_call in generated:
        args_typed: Dict[str, Any] = {}
        fn_def = fn_map.get(func_call.function)
        if fn_def:
            for arg_name, arg_value in func_call.arguments.items():
                if arg_name in fn_def.parameters:
                    expected_type = fn_def.parameters[arg_name].type
                    if expected_type == "boolean":
                        args_typed[arg_name] = arg_value if isinstance(arg_value, bool) else bool(arg_value)
                    else:
                        args_typed[arg_name] = arg_value
                else:
                    args_typed[arg_name] = arg_value
        else:
            args_typed = dict(func_call.arguments)

        results.append({
            "prompt": prompt_entry.prompt,
            "fn_name": func_call.function,
            "args": args_typed,
        })
    return results


def main() -> int:
    """Main execution function.

    Returns:
        0 on success, 1 on error.
    """
    try:
        args = parse_args()
        input_dir, output_path = resolve_paths(args)

        print(f"Loading function definitions from: {input_dir}")
        fn_map = load_function_definitions(input_dir)
        fn_defs = list(fn_map.values())
        print(f"  Found {len(fn_defs)} functions: {', '.join(fn_map.keys())}")

        print(f"Loading prompts from: {input_dir}")
        prompts = load_prompts(input_dir)
        print(f"  Found {len(prompts)} prompts")

        print("Initializing LLM model...")
        model = Small_LLM_Model()

        print("Loading vocabulary...")
        vocab_path = model.get_path_to_vocabulary_json()
        vocab = Vocabulary(vocab_path)
        print(f"  Vocabulary size: {vocab.size()}")

        print("Generating function calls with constrained decoding...")
        generated = process_prompts(model, prompts, fn_defs, vocab)
        print(f"  Generated {len(generated)} function calls")

        results = build_results(fn_map, prompts, generated)

        print(f"Writing output to: {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            f.write("\n")

        print("Done! All prompts processed successfully.")
        return 0

    except LoaderError as e:
        print(f"Error loading input files: {e}", file=sys.stderr)
        return 1
    except VocabularyError as e:
        print(f"Error loading vocabulary: {e}", file=sys.stderr)
        return 1
    except GeneratorError as e:
        print(f"Error during generation: {e}", file=sys.stderr)
        return 1
    except CallMeMaybeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except NotImplementedError as e:
        print(
            f"SDK not available: {e}\n\n"
            "This project requires the llm_sdk package provided by 42. "
            "Replace the mock in llm_sdk/ with the real package.",
            file=sys.stderr,
        )
        return 1
    except Exception as e:
        print(
            f"Unexpected error: {e}\n{traceback.format_exc()}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())

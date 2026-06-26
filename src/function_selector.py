from __future__ import annotations

from typing import Any, Dict, List, Optional

from .constrained import FunctionNameConstraint, generate
from .models import FunctionDefinition


def _build_prompt(
    functions: List[FunctionDefinition], user_prompt: str
) -> str:
    lines: List[str] = [
        "You are a function-calling assistant. "
        "Pick the best function for the request.\n",
        "Available functions:",
    ]
    for fn in functions:
        params_desc = ", ".join(
            f"{name}: {p.type}"
            for name, p in fn.parameters.items()
        )
        lines.append(
            f"  {fn.name}({params_desc}) — {fn.description}"
        )

    lines.append("")
    for fn in functions:
        if "add" in fn.name.lower() or "sum" in fn.description.lower():
            lines.append('Example: "Add 5 and 10" → ' + fn.name)
            break
    for fn in functions:
        if "greet" in fn.name.lower():
            lines.append('Example: "Say hello to Alice" → ' + fn.name)
            break
    for fn in functions:
        if "reverse" in fn.name.lower():
            lines.append('Example: "Reverse hello" → ' + fn.name)
            break
    for fn in functions:
        if "square" in fn.name.lower() or "sqrt" in fn.name.lower():
            lines.append('Example: "Square root of 9" → ' + fn.name)
            break

    lines.extend([
        "",
        f'Request: "{user_prompt}"',
        "Function:",
    ])
    return "\n".join(lines)


def select_function(
    model: Any,
    user_prompt: str,
    functions: List[FunctionDefinition],
    id_to_token: Dict[int, str],
) -> Optional[str]:
    prompt = _build_prompt(functions, user_prompt)
    input_ids = model.encode(prompt).tolist()[0]
    valid_names = [fn.name for fn in functions]
    constraint = FunctionNameConstraint(valid_names)
    result = generate(
        model, input_ids, constraint, id_to_token,
        max_tokens=50,
    )
    return result

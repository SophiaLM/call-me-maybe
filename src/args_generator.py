from __future__ import annotations

from typing import Any, Dict, Optional

from .constrained import ArgsConstraint, generate
from .models import FunctionDefinition


def _build_prompt(
    fn_def: FunctionDefinition, user_prompt: str
) -> str:
    params_desc = ", ".join(
        f"{name}: {p.type}"
        for name, p in fn_def.parameters.items()
    )
    keys = ", ".join(fn_def.parameters.keys())
    return (
        f"Extract exact values for {fn_def.name}({params_desc}).\n"
        f"Request: {user_prompt}\n"
        f"Answer with ONLY a JSON object with keys: {keys}\n"
        f"JSON:"
    )


def generate_args(
    model: Any,
    user_prompt: str,
    fn_def: FunctionDefinition,
    id_to_token: Dict[int, str],
) -> Optional[str]:
    prompt = _build_prompt(fn_def, user_prompt)
    input_ids = model.encode(prompt).tolist()[0]
    constraint = ArgsConstraint(fn_def.parameters)
    result = generate(
        model, input_ids, constraint, id_to_token,
        max_tokens=120,
    )
    return result

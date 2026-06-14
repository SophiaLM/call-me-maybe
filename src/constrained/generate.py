from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Protocol

import numpy as np

from .vocab import mask_logits


_DEBUG = False


def _debug_print(*args: Any, **kwargs: Any) -> None:
    if _DEBUG:
        print(*args, file=sys.stderr, flush=True, **kwargs)


class _ConstraintProtocol(Protocol):
    generated: str

    def valid_token_ids(
        self, id_to_token: Dict[int, str]
    ) -> List[int]: ...

    def advance_str(self, token_str: str) -> None: ...

    def is_complete(self) -> bool: ...


def generate(
    model: Any,
    input_ids: List[int],
    constraint: _ConstraintProtocol,
    id_to_token: Dict[int, str],
    max_tokens: int = 150,
) -> Optional[str]:
    past_key_values = None
    first_step = True
    token_id = 0

    for step in range(max_tokens):
        valid_ids = constraint.valid_token_ids(id_to_token)
        if not valid_ids:
            if constraint.is_complete():
                _debug_print("]")
                return _extract_best_match(constraint)
            return None

        if first_step:
            inp = np.array(input_ids, dtype=np.int64)
        else:
            inp = np.array([token_id], dtype=np.int64)

        logits, past_key_values = model.get_logits_from_input_ids(
            inp, past_key_values=past_key_values
        )
        logits = np.asarray(logits, dtype=np.float32)
        if logits.ndim == 2:
            logits = logits[0]

        masked = mask_logits(logits, valid_ids)
        token_id = int(np.argmax(masked))
        token_str = id_to_token.get(token_id, "")
        if not token_str:
            _debug_print("]")
            return _extract_best_match(constraint)
        if step == 0:
            _debug_print("[", end="")
        _debug_print(".", end="")
        constraint.advance_str(token_str)
        input_ids.append(token_id)
        first_step = False

        if constraint.is_complete():
            _debug_print("]")
            return _extract_best_match(constraint)
    _debug_print("]")
    return _extract_best_match(constraint)


def _extract_best_match(
    constraint: _ConstraintProtocol,
) -> Optional[str]:
    generated = constraint.generated
    if hasattr(constraint, "valid_names"):
        valid_names: List[str] = constraint.valid_names
        if generated in valid_names:
            return generated
        matches = [
            n for n in valid_names if generated.startswith(n)
        ]
        if matches:
            return max(matches, key=len)
        matches = [
            n for n in valid_names if n in generated
        ]
        if matches:
            return max(matches, key=len)
        return None
    return generated if generated else None

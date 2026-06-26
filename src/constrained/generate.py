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
    token_id = 0

    _debug_print("[", end="")
    for step in range(max_tokens):
        valid_ids = constraint.valid_token_ids(id_to_token)
        if not valid_ids:
            if constraint.is_complete():
                _debug_print("]")
                return _extract_best_match(constraint)
            return None

        logits_raw = model.get_logits_from_input_ids(input_ids)
        logits = np.array(logits_raw, dtype=np.float32)

        masked = mask_logits(logits, valid_ids)
        token_id = int(np.argmax(masked))
        token_str = id_to_token.get(token_id, "")
        if not token_str:
            _debug_print("]")
            return _extract_best_match(constraint)
        _debug_print(".", end="")
        constraint.advance_str(token_str)
        input_ids.append(token_id)

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

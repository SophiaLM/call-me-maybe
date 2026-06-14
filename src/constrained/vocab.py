from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List

import numpy as np

_CONTROL_CHARS = frozenset(
    chr(i) for i in range(32) if chr(i) not in "\n\r\t"
)


def filter_vocab(
    vocab: Dict[int, str],
) -> Dict[int, str]:
    return {
        tid: s for tid, s in vocab.items()
        if s and not any(c in _CONTROL_CHARS for c in s)
    }


@dataclass
class _VocabIndex:
    first_char_index: Dict[str, List[int]] = field(default_factory=dict)
    number_token_ids: FrozenSet[int] = frozenset()
    string_token_ids: FrozenSet[int] = frozenset()
    quote_token_ids: FrozenSet[int] = frozenset()


_vocab_index_cache: Dict[int, _VocabIndex] = {}


def _build_first_char_index(
    id_to_token: Dict[int, str],
) -> Dict[str, List[int]]:
    index: Dict[str, List[int]] = {}
    for tid, s in id_to_token.items():
        if not s:
            continue
        index.setdefault(s[0], []).append(tid)
    return index


def _build_number_token_ids(
    id_to_token: Dict[int, str],
) -> FrozenSet[int]:
    _num_chars = frozenset("-0123456789.eE+")
    return frozenset(
        tid for tid, s in id_to_token.items()
        if s and all(c in _num_chars for c in s)
    )


def _build_string_token_ids(
    id_to_token: Dict[int, str],
) -> FrozenSet[int]:
    return frozenset(
        tid for tid, s in id_to_token.items()
        if s and '"' not in s
    )


def _build_quote_token_ids(
    id_to_token: Dict[int, str],
) -> FrozenSet[int]:
    return frozenset(
        tid for tid, s in id_to_token.items()
        if s and '"' in s
    )


def _get_vocab_index(
    id_to_token: Dict[int, str],
) -> _VocabIndex:
    key = id(id_to_token)
    cached = _vocab_index_cache.get(key)
    if cached is not None:
        return cached
    idx = _VocabIndex(
        first_char_index=_build_first_char_index(id_to_token),
        number_token_ids=_build_number_token_ids(id_to_token),
        string_token_ids=_build_string_token_ids(id_to_token),
        quote_token_ids=_build_quote_token_ids(id_to_token),
    )
    _vocab_index_cache[key] = idx
    return idx


def mask_logits(
    logits: np.ndarray, valid_ids: List[int]
) -> np.ndarray:
    masked = np.full_like(logits, -np.inf)
    if not valid_ids:
        return masked
    arr = np.array(valid_ids, dtype=np.int32)
    arr = arr[(arr >= 0) & (arr < logits.shape[0])]
    masked[arr] = logits[arr]
    return masked

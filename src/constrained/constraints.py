from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from ..models import FunctionParameter
from .vocab import _get_vocab_index, _VocabIndex


_prefix_cache_store: Dict[
    Tuple[int, Tuple[str, ...]],
    Dict[str, FrozenSet[int]],
] = {}


def _build_prefix_cache(
    valid_names: List[str],
    id_to_token: Dict[int, str],
) -> Dict[str, FrozenSet[int]]:
    prefixes: Set[str] = {""}
    for name in valid_names:
        for i in range(len(name) + 1):
            prefixes.add(name[:i])
    cache: Dict[str, FrozenSet[int]] = {}
    for prefix in prefixes:
        valid: List[int] = []
        for tid, token_str in id_to_token.items():
            if not token_str:
                continue
            test = prefix + token_str
            if any(n.startswith(test) for n in valid_names):
                valid.append(tid)
        cache[prefix] = frozenset(valid)
    return cache


def _get_prefix_cache(
    valid_names: List[str],
    id_to_token: Dict[int, str],
) -> Dict[str, FrozenSet[int]]:
    key = (id(id_to_token), tuple(sorted(valid_names)))
    cached = _prefix_cache_store.get(key)
    if cached is not None:
        return cached
    cache = _build_prefix_cache(valid_names, id_to_token)
    _prefix_cache_store[key] = cache
    return cache


class FunctionNameConstraint:
    def __init__(self, valid_names: List[str]) -> None:
        self.valid_names = valid_names
        self.generated = ""

    def valid_token_ids(
        self, id_to_token: Dict[int, str]
    ) -> List[int]:
        cache = _get_prefix_cache(self.valid_names, id_to_token)
        return list(cache.get(self.generated, frozenset()))

    def advance_str(self, token_str: str) -> None:
        self.generated += token_str

    def is_complete(self) -> bool:
        if self.generated in self.valid_names:
            return True
        if not any(
            n.startswith(self.generated)
            for n in self.valid_names
        ):
            return True
        return False


_TYPE_ALIASES = {
    "integer": "number",
    "int": "number",
    "float": "number",
    "double": "number",
    "str": "string",
    "bool": "boolean",
}


class ArgsConstraint:
    def __init__(
        self,
        parameters: Dict[str, FunctionParameter],
    ) -> None:
        self.params = parameters
        self.state = "START"
        self.generated = ""
        self.completed_keys: Set[str] = set()
        self.current_key = ""
        self.param_type: Optional[str] = None
        self.bool_built = ""
        self._index: Optional[_VocabIndex] = None

    def valid_token_ids(
        self, id_to_token: Dict[int, str]
    ) -> List[int]:
        if self._index is None:
            self._index = _get_vocab_index(id_to_token)
        idx = self._index
        remaining = (
            set(self.params.keys()) - self.completed_keys
        )

        if self.state == "START":
            return self._check_first_char(idx, "{", id_to_token)

        if self.state == "KEY_OR_CLOSE":
            if not remaining:
                return self._check_first_char(
                    idx, "}", id_to_token
                )
            return self._check_first_char(
                idx, '"', id_to_token
            )

        if self.state == "AFTER_KEY":
            return self._check_first_char(
                idx, ":", id_to_token
            )

        if self.state == "AFTER_VALUE":
            result = self._check_first_char(
                idx, ",", id_to_token
            ) if remaining else []
            result += self._check_first_char(
                idx, "}", id_to_token
            )
            return result

        if self.state == "IN_TRUE":
            expected = "true"[len(self.bool_built)]
            return self._check_first_char(
                idx, expected, id_to_token
            )
        if self.state == "IN_FALSE":
            expected = "false"[len(self.bool_built)]
            return self._check_first_char(
                idx, expected, id_to_token
            )

        if self.state == "IN_STRING":
            result = list(idx.string_token_ids)
            for tid in idx.quote_token_ids:
                s = id_to_token.get(tid, "")
                if s and self._check_string(s):
                    result.append(tid)
            return result

        if (
            self.state == "BEFORE_VALUE"
            and self.param_type == "number"
        ):
            result = []
            for tid in idx.number_token_ids:
                s = id_to_token.get(tid, "")
                if s and self._check_string(s):
                    result.append(tid)
            if not result:
                result = list(id_to_token.keys())
            return result

        if self.state == "IN_KEY":
            return self._valid_key_tokens(idx, id_to_token)

        if self.state == "IN_NUMBER":
            chars = set("0123456789.")
            if remaining:
                chars.add(",")
            chars.add("}")
            return self._check_multi_char(
                idx, chars, id_to_token
            )

        if self.state == "BEFORE_VALUE":
            return self._valid_value_start_tokens(
                idx, id_to_token
            )

        return []

    def advance_str(self, token_str: str) -> None:
        for c in token_str:
            self._advance_char(c)

    def is_complete(self) -> bool:
        return self.state == "DONE"

    def _check_first_char(
        self,
        idx: _VocabIndex,
        c: str,
        id_to_token: Dict[int, str],
    ) -> List[int]:
        valid: List[int] = []
        for tid in idx.first_char_index.get(c, []):
            s = id_to_token.get(tid, "")
            if s and self._check_string(s):
                valid.append(tid)
        return valid

    def _check_multi_char(
        self,
        idx: _VocabIndex,
        chars: Set[str],
        id_to_token: Dict[int, str],
    ) -> List[int]:
        valid: List[int] = []
        for c in chars:
            for tid in idx.first_char_index.get(c, []):
                s = id_to_token.get(tid, "")
                if s and self._check_string(s):
                    valid.append(tid)
        return valid

    def _valid_key_tokens(
        self,
        idx: _VocabIndex,
        id_to_token: Dict[int, str],
    ) -> List[int]:
        remaining = (
            set(self.params.keys()) - self.completed_keys
        )
        if not remaining:
            return []
        chars: Set[str] = set()
        pos = len(self.current_key)
        for key in remaining:
            if pos < len(key):
                chars.add(key[pos])
        if self.current_key in remaining:
            chars.add('"')
        result = self._check_multi_char(idx, chars, id_to_token)
        if result:
            return result
        valid: List[int] = []
        for tid, token_str in id_to_token.items():
            if not token_str:
                continue
            candidate = self.current_key + token_str
            if any(k.startswith(candidate) for k in remaining):
                valid.append(tid)
            elif candidate in remaining:
                for tid2, s2 in id_to_token.items():
                    if s2 == '"':
                        valid.append(tid2)
                break
        return valid

    def _valid_value_start_tokens(
        self,
        idx: _VocabIndex,
        id_to_token: Dict[int, str],
    ) -> List[int]:
        if self.param_type == "string":
            result = self._check_first_char(
                idx, '"', id_to_token
            )
            result += self._check_first_char(
                idx, " ", id_to_token
            )
            return result
        if self.param_type == "number":
            return self._check_multi_char(
                idx, set(" 0123456789-."), id_to_token
            )
        if self.param_type == "boolean":
            result = self._check_first_char(
                idx, "t", id_to_token
            )
            result += self._check_first_char(
                idx, "f", id_to_token
            )
            result += self._check_first_char(
                idx, " ", id_to_token
            )
            return result
        return list(id_to_token.keys())

    def _check_string(self, token_str: str) -> bool:
        if not token_str:
            return False
        if not self._is_valid_char(token_str[0]):
            return False
        saved = self._save_state()
        try:
            self._advance_char(token_str[0])
            for c in token_str[1:]:
                if self.state == "DONE":
                    return False
                if not self._is_valid_char(c):
                    return False
                self._advance_char(c)
            return True
        finally:
            self._restore_state(saved)

    def _save_state(self) -> dict:
        return {
            "state": self.state,
            "generated": self.generated,
            "completed_keys": set(self.completed_keys),
            "current_key": self.current_key,
            "param_type": self.param_type,
            "bool_built": self.bool_built,
        }

    def _restore_state(self, saved: dict) -> None:
        self.state = saved["state"]
        self.generated = saved["generated"]
        self.completed_keys = saved["completed_keys"]
        self.current_key = saved["current_key"]
        self.param_type = saved["param_type"]
        self.bool_built = saved["bool_built"]

    def _advance_char(self, c: str) -> None:
        self.generated += c

        if self.state == "START":
            self.state = "KEY_OR_CLOSE"

        elif self.state == "KEY_OR_CLOSE":
            if c == '"':
                self.current_key = ""
                self.state = "IN_KEY"
            elif c == "}":
                self.state = "DONE"

        elif self.state == "IN_KEY":
            if c == '"':
                self.completed_keys.add(self.current_key)
                self.state = "AFTER_KEY"
            else:
                self.current_key += c

        elif self.state == "AFTER_KEY":
            if c == ":":
                self.state = "BEFORE_VALUE"
                raw_type = self.params[
                    self.current_key
                ].type
                self.param_type = _TYPE_ALIASES.get(
                    raw_type, raw_type
                )

        elif self.state == "BEFORE_VALUE":
            if c == '"':
                self.state = "IN_STRING"
            elif c in "0123456789-":
                self.state = "IN_NUMBER"
            elif c == "t":
                self.state = "IN_TRUE"
                self.bool_built = "t"
            elif c == "f":
                self.state = "IN_FALSE"
                self.bool_built = "f"

        elif self.state == "IN_STRING":
            if c == '"':
                self.state = "AFTER_VALUE"

        elif self.state == "IN_NUMBER":
            if c == ",":
                self.state = "KEY_OR_CLOSE"
            elif c == "}":
                self.state = "DONE"

        elif self.state == "IN_TRUE":
            self.bool_built += c
            if self.bool_built == "true":
                self.state = "AFTER_VALUE"

        elif self.state == "IN_FALSE":
            self.bool_built += c
            if self.bool_built == "false":
                self.state = "AFTER_VALUE"

        elif self.state == "AFTER_VALUE":
            if c == ",":
                self.state = "KEY_OR_CLOSE"
            elif c == "}":
                self.state = "DONE"

    def _is_valid_char(self, c: str) -> bool:
        remaining = set(self.params.keys()) - self.completed_keys

        if self.state == "START":
            return c == "{"

        if self.state == "KEY_OR_CLOSE":
            if remaining:
                return c == '"'
            return c == "}"

        if self.state == "IN_KEY":
            idx = len(self.current_key)
            for key in remaining:
                if idx < len(key) and key[idx] == c:
                    return True
            if self.current_key in remaining and c == '"':
                return True
            return False

        if self.state == "AFTER_KEY":
            return c == ":"

        if self.state == "BEFORE_VALUE":
            return self._is_valid_value_start(c)

        if self.state == "IN_STRING":
            return True

        if self.state == "IN_NUMBER":
            if c in "0123456789.":
                return True
            if c == "," and remaining:
                return True
            if c == "}":
                return True
            return False

        if self.state == "IN_TRUE":
            expected = "true"[len(self.bool_built)]
            return c == expected

        if self.state == "IN_FALSE":
            expected = "false"[len(self.bool_built)]
            return c == expected

        if self.state == "AFTER_VALUE":
            if c == "," and remaining:
                return True
            if c == "}":
                return True
            return False

        return False

    def _is_valid_value_start(self, c: str) -> bool:
        if self.param_type == "string":
            return c in ' "'
        if self.param_type == "number":
            return c in " 0123456789-."
        if self.param_type == "boolean":
            return c in " tf"
        return True

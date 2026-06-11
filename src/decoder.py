"""Constrained decoding engine for structured JSON function calls.

This module implements token-level constrained decoding that enforces
both JSON syntactic validity and schema compliance during LLM generation.
At each step, only tokens that maintain valid JSON structure and match
the expected function call schema are allowed.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set

import numpy as np

from src.models import FunctionDefinition, type_matches
from src.vocabulary import Vocabulary


def _is_valid_json_prefix(text: str) -> bool:
    """Check if text is a valid prefix of some complete JSON value.

    Uses json.loads with error-position analysis. A string is a valid
    JSON prefix if either:
    - It parses as complete valid JSON
    - Parsing fails at or very near the end (incomplete = prefix)
    - We are inside an unclosed string, after a structural character,
      or partway through a keyword

    Args:
        text: The string to check.

    Returns:
        True if text could be the start of valid JSON.
    """
    stripped = text.lstrip()
    if not stripped:
        return True

    # Must start with a valid JSON-starting character
    if stripped[0] not in '{"[0123456789tfn-':
        return False

    try:
        json.loads(stripped)
        return True
    except json.JSONDecodeError as e:
        cleaned = stripped.rstrip()
        error_pos = e.pos

        if error_pos >= len(cleaned):
            return True

        if error_pos >= len(cleaned) - 1 and cleaned[-1] in '",:{[':
            return True

        in_str = False
        escaped = False
        for c in cleaned:
            if escaped:
                escaped = False
                continue
            if c == '\\' and in_str:
                escaped = True
                continue
            if c == '"':
                in_str = not in_str

        if in_str:
            return True

        if error_pos == 0:
            if any(kw.startswith(stripped) for kw in ["true", "false", "null"]):
                return True
            if stripped == "-":
                return True

        return False


def _get_partial_prefix_state(text: str) -> Dict[str, Any]:
    """Analyze the partial JSON string to determine generation context.

    Examines the string character by character, tracking:
    - Whether we are inside a string literal
    - Whether we are escaping
    - Brace depth
    - What the last significant (non-whitespace) character was
    - Which key we are currently reading a value for (if any)
    - Completed value strings (key -> value at each depth)

    Args:
        text: The partial generated string.

    Returns:
        Dict with keys: in_string, escaped, brace_depth,
        last_significant, current_key, keys_at_level,
        reading_value, last_key_at_depth, completed_values.
    """
    in_string = False
    escaped = False
    brace_depth = 0
    last_significant = ""
    current_key_chars: List[str] = []
    keys_at_level: Dict[int, List[str]] = {0: []}
    after_colon = False
    reading_value = False
    last_key_at_depth: Dict[int, str] = {}
    completed_values: Dict[int, Dict[str, str]] = {}

    for c in text:
        if escaped:
            escaped = False
            if in_string:
                current_key_chars.append('\\' + c)
            continue

        if c == '\\' and in_string:
            escaped = True
            continue

        if c == '"':
            if not in_string:
                in_string = True
                current_key_chars = []
                if after_colon:
                    reading_value = True
                else:
                    reading_value = False
            else:
                in_string = False
                key = "".join(current_key_chars)
                if brace_depth >= 0:
                    if not reading_value:
                        keys_at_level.setdefault(brace_depth, []).append(key)
                    else:
                        pending_key = last_key_at_depth.get(brace_depth, "")
                        if pending_key:
                            completed_values.setdefault(brace_depth, {})[pending_key] = key
                    after_colon = False
            last_significant = c
            continue

        if in_string:
            current_key_chars.append(c)
            continue

        if c in ' \t\n\r':
            continue

        if c == '{':
            brace_depth += 1
            after_colon = False
        elif c == '}':
            brace_depth = max(0, brace_depth - 1)
            after_colon = False
        elif c == ':':
            after_colon = True
            keys = keys_at_level.get(brace_depth, [])
            if keys:
                last_key_at_depth[brace_depth] = keys[-1]
        elif c == ',':
            after_colon = False
        else:
            after_colon = False

        last_significant = c

    current_key_val = "".join(current_key_chars) if in_string else ""

    return {
        "in_string": in_string,
        "escaped": escaped,
        "brace_depth": brace_depth,
        "last_significant": last_significant,
        "current_key": current_key_val,
        "keys_at_level": keys_at_level,
        "after_colon": after_colon,
        "reading_value": reading_value,
        "last_key_at_depth": last_key_at_depth,
        "completed_values": completed_values,
    }


def _get_expected_first_chars(
    state: Dict[str, Any],
    fn_map: Optional[Dict[str, FunctionDefinition]] = None,
) -> str:
    if state["in_string"]:
        bd = state["brace_depth"]
        rv = state.get("reading_value", False)
        first_char_of_key = not state.get("current_key", "")

        if bd == 1 and not rv and first_char_of_key:
            completed = state.get("keys_at_level", {}).get(1, [])
            if "function" not in completed:
                return 'f'
            if "arguments" not in completed:
                return 'a'
            return ''

        if bd == 2 and not rv and first_char_of_key:
            fn_name = state.get("completed_values", {}).get(1, {}).get("function", "")
            if fn_name and fn_map is not None and fn_name in fn_map:
                all_params = fn_map[fn_name].parameters
                used_keys = set(state.get("keys_at_level", {}).get(2, []))
                available = [k for k in all_params if k not in used_keys]
                if not available:
                    return ''
                valid_fc = sorted({k[0] for k in available if k})
                return ''.join(valid_fc)
            return ''

        if bd == 1 and rv:
            pending_key = state.get("last_key_at_depth", {}).get(1, "")
            if pending_key == "function":
                return 'abcdefghijklmnopqrstuvwxyz_0123456789"'

        return ('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
                ' _-0123456789{}[]:,"\'\\!@#$%^&*()+=<>?/|~`')

    last = state["last_significant"]
    depth = state["brace_depth"]

    if not last:
        return '{'

    if last == '{':
        return '"'

    if last == '}':
        if depth > 0:
            if depth == 1:
                closed = set(state.get("keys_at_level", {}).get(1, []))
                if {"function", "arguments"}.issubset(closed):
                    return '}'
            return ',}'
        return ''

    if last in '[':
        return '"{[0123456789.-'

    if last in ']':
        return ',]'

    if last == ':':
        expected = '"{[0123456789.-'
        if fn_map is not None and depth == 2:
            completed_depth1 = state.get("completed_values", {}).get(1, {})
            fn_name = completed_depth1.get("function", "")
            current_key = state.get("last_key_at_depth", {}).get(2, "")
            if fn_name and fn_map is not None and fn_name in fn_map and current_key:
                param_def = fn_map[fn_name].parameters.get(current_key)
                if param_def is not None and param_def.type in ("number", "integer"):
                    expected = expected.replace('"', '').replace('{', '').replace('[', '')
        return expected

    if last == ',':
        return '"{['
    if last == '"':
        if state.get("reading_value"):
            if depth == 1:
                closed = set(state.get("keys_at_level", {}).get(1, []))
                remaining_keys = {"function", "arguments"} - closed
                if not remaining_keys:
                    return '}'
            if depth == 2:
                fn_name = state.get("completed_values", {}).get(1, {}).get("function", "")
                if fn_name and fn_map is not None and fn_name in fn_map:
                    used = set(state.get("keys_at_level", {}).get(2, []))
                    remaining_params = [k for k in fn_map[fn_name].parameters if k not in used]
                    if not remaining_params:
                        return '}'
            return ',}'
        return ':'

    if last in '0123456789':
        if depth == 1:
            closed = set(state.get("keys_at_level", {}).get(1, []))
            if {"function", "arguments"}.issubset(closed):
                return '}0123456789.'
        if depth == 2:
            fn_name = state.get("completed_values", {}).get(1, {}).get("function", "")
            if fn_name and fn_map is not None and fn_name in fn_map:
                used = set(state.get("keys_at_level", {}).get(2, []))
                remaining_params = [k for k in fn_map[fn_name].parameters if k not in used]
                if not remaining_params:
                    return '}0123456789.'
        return ',}0123456789.'

    if last in 'eE':
        return '0123456789+-'

    if last in '+-':
        return '0123456789.'

    if last == '.':
        return '0123456789'

    return ',}'


def _has_duplicate_keys(text: str) -> bool:
    """Check for duplicate keys in JSON text using object_pairs_hook."""
    has_dupes: list[bool] = [False]

    def hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        keys = [p[0] for p in pairs]
        if len(keys) != len(set(keys)):
            has_dupes[0] = True
        return dict(pairs)

    try:
        json.JSONDecoder(object_pairs_hook=hook).decode(text)
    except json.JSONDecodeError:
        return False

    return has_dupes[0]


def _matches_function_schema(text: str, fn_map: Dict[str, FunctionDefinition]) -> bool:
    """Validate that text is consistent with the function call schema.

    Checks:
    - Top-level keys are only 'function' and 'arguments'
    - No duplicate keys at same level
    - 'function' value (if present) is a valid function name
    - 'arguments' value (if present) is an object
    - Argument keys inside 'arguments' are valid for the selected function
    - Argument values have correct types

    Args:
        text: The partial or complete generated string.
        fn_map: Map of function names to their definitions.

    Returns:
        True if text is schema-compliant.
    """
    if _has_duplicate_keys(text):
        return False

    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            return False
    except (json.JSONDecodeError, ValueError):
        return _validate_incomplete_schema(text, fn_map)

    return _validate_complete_schema(obj, fn_map)


def _validate_complete_schema(
    obj: Any,
    fn_map: Dict[str, FunctionDefinition],
) -> bool:
    """Validate a complete parsed JSON object against function schema.

    Args:
        obj: The parsed JSON object.
        fn_map: Map of function names to definitions.

    Returns:
        True if the object matches the schema.
    """
    if not isinstance(obj, dict):
        return False

    valid_keys = {"function", "arguments"}
    obj_keys = set(obj.keys())

    if not obj_keys.issubset(valid_keys):
        return False

    if "function" in obj:
        fn_name = obj["function"]
        if not isinstance(fn_name, str):
            return False
        if fn_name not in fn_map:
            return False

    if "arguments" in obj:
        args = obj["arguments"]
        if not isinstance(args, dict):
            return False

        fn_name = obj.get("function", "")
        if fn_name in fn_map:
            fn_def = fn_map[fn_name]
            valid_arg_keys = set(fn_def.parameters.keys())
            actual_keys = set(args.keys())

            if not actual_keys.issubset(valid_arg_keys):
                return False

            for arg_name, arg_value in args.items():
                if arg_name in fn_def.parameters:
                    expected_type = fn_def.parameters[arg_name].type
                    if not type_matches(arg_value, expected_type):
                        return False

    return True


def _validate_incomplete_schema(
    text: str,
    fn_map: Dict[str, FunctionDefinition],
) -> bool:
    """Validate an incomplete string against function call schema.

    Validates partially generated JSON against:
    - Top-level keys: only 'function' and 'arguments' (depth 1)
    - 'function' value: must be a valid function name
    - 'arguments' value: must be an object (not a string)
    - Inside 'arguments': keys must be valid parameter names (depth 2)
    - Empty string keys are never valid at any depth

    Args:
        text: The incomplete JSON string.
        fn_map: Map of function names to definitions.

    Returns:
        True if what has been parsed so far is schema-compliant.
    """
    try:
        decoder = json.JSONDecoder()
        idx = 0
        while idx < len(text):
            while idx < len(text) and text[idx] in ' \t\n\r':
                idx += 1
            if idx >= len(text):
                break
            try:
                obj, end = decoder.raw_decode(text, idx)
                if not _validate_complete_schema(obj, fn_map):
                    return False
                idx = end
            except (json.JSONDecodeError, ValueError):
                break
    except (json.JSONDecodeError, ValueError):
        pass

    state = _get_partial_prefix_state(text)
    valid_top_keys = {"function", "arguments"}
    depth = state["brace_depth"]
    completed_depth1 = state.get("completed_values", {}).get(1, {})
    fn_name = completed_depth1.get("function", "")

    # (1) Empty completed keys are never valid at any depth
    for keys in state.get("keys_at_level", {}).values():
        if any(k == "" for k in keys):
            return False

    # (2) In-progress key string at depth 1: must be prefix of valid top key
    if state["in_string"] and not state["reading_value"] and depth == 1:
        cur = state.get("current_key", "")
        if cur and not any(k.startswith(cur) for k in valid_top_keys):
            return False

    # (2b) Ordering constraint: "function" key must precede "arguments"
    if state["in_string"] and not state["reading_value"] and depth == 1:
        cur = state.get("current_key", "")
        completed = state.get("keys_at_level", {}).get(1, [])
        if "function" not in completed and cur:
            if not "function".startswith(cur):
                return False

    # (3) In-progress key string at depth 2 (inside arguments): must be arg name prefix
    if state["in_string"] and not state["reading_value"] and depth == 2:
        cur = state.get("current_key", "")
        if fn_name and fn_name in fn_map:
            param_keys = set(fn_map[fn_name].parameters.keys())
            if cur and not any(k.startswith(cur) for k in param_keys):
                return False

    # (3.5) In-progress value string for "function": must be prefix of a valid fn name
    if state["in_string"] and state["reading_value"] and depth == 1:
        pending_key = state.get("last_key_at_depth", {}).get(1, "")
        if pending_key == "function":
            cur = state.get("current_key", "")
            if cur and not any(fn.startswith(cur) for fn in fn_map):
                return False

    # (4) In-progress value string for "arguments" at depth 1: not allowed
    if state["in_string"] and state["reading_value"] and depth == 1:
        pending_key = state.get("last_key_at_depth", {}).get(1, "")
        if pending_key == "arguments":
            return False

    # (5) Just closed a string at depth 1: if it was a key, it must be valid
    if not state["in_string"] and depth == 1:
        if state.get("last_significant") == '"':
            closed_keys = state.get("keys_at_level", {}).get(1, [])
            if closed_keys and closed_keys[-1] not in valid_top_keys:
                return False
            if closed_keys and closed_keys[-1] == "arguments":
                if "function" not in closed_keys:
                    return False

    # (6) Just closed a string at depth 2: key must be valid parameter name
    if not state["in_string"] and depth == 2:
        if state.get("last_significant") == '"':
            closed_keys = state.get("keys_at_level", {}).get(2, [])
            if closed_keys and closed_keys[-1]:
                if fn_name and fn_name in fn_map:
                    param_keys = set(fn_map[fn_name].parameters.keys())
                    if closed_keys[-1] not in param_keys:
                        return False
                    if closed_keys.count(closed_keys[-1]) > 1:
                        return False
                elif not fn_name:
                    return False

    # (6.5) Check for duplicate keys at any depth
    for keys in state.get("keys_at_level", {}).values():
        if len(keys) != len(set(keys)):
            return False

    # (7) Validate completed values at depth 1
    if "function" in completed_depth1:
        if completed_depth1["function"] not in fn_map:
            return False

    # (8) Validate completed argument names against parameter names
    completed_depth2 = state.get("completed_values", {}).get(2, {})
    if fn_name and fn_name in fn_map:
        fn_def = fn_map[fn_name]
        for arg_name, arg_value in completed_depth2.items():
            if arg_name not in fn_def.parameters:
                return False

    return True


class ConstrainedDecoder:
    """Constrained decoding engine for JSON function calls.

    Guides LLM generation token by token to produce valid JSON that
    conforms to the expected function call schema. Maintains generation
    state and filters out invalid tokens at each step.

    Attributes:
        fn_map: Map of valid function names to their definitions.
        valid_fn_names: List of all valid function names.
        eos_token_id: End-of-sequence token ID.
        partial: Current partial generated string.
    """

    def __init__(
        self,
        function_defs: List[FunctionDefinition],
        eos_token_id: int = 2,
    ) -> None:
        self.fn_map: Dict[str, FunctionDefinition] = {
            f.name: f for f in function_defs
        }
        self.valid_fn_names: List[str] = list(self.fn_map.keys())
        self.eos_token_id = eos_token_id
        self.partial: str = ""

    def set_vocabulary(self, vocab: Vocabulary) -> None:
        pass

    def reset(self) -> None:
        self.partial = ""

    def get_valid_token_ids(
        self,
        logits: np.ndarray,
        vocab: Vocabulary,
    ) -> np.ndarray:
        state = _get_partial_prefix_state(self.partial)
        expected_chars = _get_expected_first_chars(state, self.fn_map)

        if not expected_chars:
            masked = np.full_like(logits, -np.inf)
            if self.partial.strip():
                masked[self.eos_token_id] = 0.0
            return masked

        candidates: Set[int] = set()
        for c in expected_chars:
            candidates.update(vocab.get_tokens_by_first_char(c))

        if self.eos_token_id not in candidates:
            candidates.add(self.eos_token_id)

        valid: Set[int] = set()
        for tid in candidates:
            token_str = vocab.token_to_string(tid)
            if not token_str:
                continue
            new_partial = self.partial + token_str
            if _is_valid_json_prefix(new_partial):
                if _matches_function_schema(new_partial, self.fn_map):
                    valid.add(tid)

        if not valid:
            valid.add(self.eos_token_id)

        masked = np.full_like(logits, -np.inf)
        for tid in valid:
            if tid < len(masked):
                masked[tid] = logits[tid]

        return masked

    def _is_complete_call(self, text: str) -> bool:
        stripped = text.strip()
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict) and "function" in obj and "arguments" in obj:
                brace_count = 0
                in_str = False
                for c in stripped:
                    if c == '"':
                        in_str = not in_str
                    elif not in_str:
                        if c == '{':
                            brace_count += 1
                        elif c == '}':
                            brace_count -= 1
                return brace_count == 0
        except (json.JSONDecodeError, ValueError):
            pass
        return False

    def step(self, token_id: int, token_str: str) -> bool:
        if self._is_complete_call(self.partial):
            return False

        if token_id == self.eos_token_id:
            return False

        self.partial += token_str

        if self._is_complete_call(self.partial):
            return False

        if len(self.partial) > 500:
            return False

        return True

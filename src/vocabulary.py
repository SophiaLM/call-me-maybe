"""Vocabulary mapping between token IDs and their string representations."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

from src.errors import VocabularyError


class Vocabulary:
    """Maps token IDs to their string representations.

    Loads a JSON vocabulary file where keys are string token IDs
    and values are the corresponding token strings. Provides methods
    for efficient lookup and prefix matching.

    Attributes:
        id_to_token: Mapping from token ID to string.
        token_to_ids: Mapping from token string to list of IDs.
        by_first_char: Tokens indexed by their first character.
    """

    def __init__(self, path: str) -> None:
        """Initialize vocabulary from a JSON file.

        Args:
            path: Path to the vocabulary JSON file.

        Raises:
            VocabularyError: If the file cannot be loaded or is malformed.
        """
        self.id_to_token: Dict[int, str] = {}
        self.token_to_ids: Dict[str, List[int]] = defaultdict(list)
        self.by_first_char: Dict[str, List[int]] = defaultdict(list)
        self._load(path)

    def _load(self, path: str) -> None:
        """Load and parse the vocabulary JSON file.

        Args:
            path: Path to the vocabulary JSON file.

        Raises:
            VocabularyError: On loading or parsing failures.
        """
        filepath = Path(path)
        if not filepath.exists():
            raise VocabularyError(f"Vocabulary file not found: {path}")
        if not filepath.is_file():
            raise VocabularyError(f"Path is not a file: {path}")

        try:
            raw: Dict[str, str] = json.loads(filepath.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise VocabularyError(
                f"Invalid JSON in vocabulary file: {e}",
            ) from e
        except OSError as e:
            raise VocabularyError(
                f"Error reading vocabulary file: {e}",
            ) from e

        if not raw:
            raise VocabularyError("Vocabulary file is empty")

        for id_str, token_text in raw.items():
            try:
                token_id = int(id_str)
            except ValueError:
                raise VocabularyError(
                    f"Invalid token ID (not an integer): {id_str!r}",
                )

            self.id_to_token[token_id] = token_text
            self.token_to_ids[token_text].append(token_id)

            if token_text:
                first_char = token_text[0]
                self.by_first_char[first_char].append(token_id)

    def token_to_string(self, token_id: int) -> str:
        """Get the string representation of a token ID.

        Args:
            token_id: The token ID.

        Returns:
            The string representation, or empty string if not found.
        """
        return self.id_to_token.get(token_id, "")

    def string_to_token_ids(self, text: str) -> List[int]:
        """Get all token IDs that match the given text exactly.

        Args:
            text: The token text to look up.

        Returns:
            List of matching token IDs.
        """
        return self.token_to_ids.get(text, [])

    def get_tokens_by_first_char(self, char: str) -> List[int]:
        """Get all token IDs whose string starts with a given character.

        Args:
            char: The first character to match.

        Returns:
            List of matching token IDs.
        """
        return self.by_first_char.get(char, [])

    def get_tokens_by_prefix(self, prefix: str) -> Set[int]:
        """Get all token IDs whose string starts with a given prefix.

        This is slower than first-char lookup but useful for
        detailed prefix matching.

        Args:
            prefix: The prefix to match.

        Returns:
            Set of matching token IDs.
        """
        if not prefix:
            return set()

        # Start with tokens matching first character
        candidates = self.get_tokens_by_first_char(prefix[0])
        if not candidates:
            return set()

        result: Set[int] = set()
        for tid in candidates:
            token_str = self.id_to_token.get(tid, "")
            if token_str.startswith(prefix):
                result.add(tid)

        return result

    def all_token_ids(self) -> List[int]:
        """Get all token IDs in the vocabulary.

        Returns:
            List of all token IDs.
        """
        return list(self.id_to_token.keys())

    def size(self) -> int:
        """Get the vocabulary size.

        Returns:
            Number of tokens in the vocabulary.
        """
        return len(self.id_to_token)

    def __repr__(self) -> str:
        return f"Vocabulary(size={self.size()})"

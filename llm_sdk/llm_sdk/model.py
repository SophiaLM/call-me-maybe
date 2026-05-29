from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


LETTER_SCORE: Dict[str, float] = {
    " ": 3.0, "e": 2.8, "t": 2.6, "a": 2.5, "o": 2.3, "i": 2.3,
    "n": 2.2, "s": 2.1, "h": 1.9, "r": 1.8,
    "d": 1.6, "l": 1.5, "c": 1.4, "u": 1.4, "m": 1.2,
    "w": 1.1, "f": 1.0, "g": 0.9, "y": 0.9, "p": 0.8,
    "b": 0.7, "v": 0.6, "k": 0.5, "j": 0.3, "x": 0.2, "q": 0.1, "z": 0.1,
    "0": 2.0, "1": 1.8, "2": 1.6, "3": 1.4, "4": 1.2,
    "5": 1.0, "6": 0.8, "7": 0.6, "8": 0.4, "9": 0.2,
    '"': 3.5, "{": 3.0, "}": 3.0, ":": 3.0, ",": 3.0,
    "-": 1.0, ".": 1.0, "_": 2.0,
}


def _score_token(token_str: str) -> float:
    if not token_str:
        return -10.0
    c = token_str[0]
    return LETTER_SCORE.get(c, -5.0)


def _build_default_vocab() -> Dict[str, str]:
    vocab: Dict[str, str] = {
        "0": "<unk>",
        "1": "<s>",
        "2": "</s>",
    }
    tid = 3
    for code in range(32, 127):
        vocab[str(tid)] = chr(code)
        tid += 1
    return vocab


class Small_LLM_Model:
    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        self.model_dir = str(model_dir) if model_dir else "mock"
        self.device = device
        self.vocab_cache: Optional[str] = None
        self.vocab_data: Optional[Dict[str, str]] = None

    def get_path_to_vocabulary_json(self) -> str:
        if self.vocab_cache is not None:
            return self.vocab_cache

        model_path = Path(self.model_dir)
        for candidate in ["vocab.json", "vocabulary.json"]:
            full = model_path / candidate
            if full.exists():
                self.vocab_cache = str(full.resolve())
                return self.vocab_cache

        vocab = _build_default_vocab()
        self.vocab_data = vocab
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        )
        json.dump(vocab, tmp, ensure_ascii=False)
        tmp.close()
        self.vocab_cache = tmp.name
        return self.vocab_cache

    def get_logits_from_input_ids(self, input_ids: np.ndarray) -> np.ndarray:
        vocab = self.get_vocab_dict()
        ids = sorted(int(k) for k in vocab)
        logits = np.full(max(ids) + 1, -1e9, dtype=np.float64)
        for tid in ids:
            token_str = vocab.get(str(tid), "")
            logits[tid] = _score_token(token_str)
        return logits

    def encode(self, text: str) -> List[int]:
        vocab = self.get_vocab_dict()
        id_to_token: Dict[int, str] = {
            int(k): v for k, v in vocab.items()
        }
        token_to_id: Dict[str, int] = {
            v: k for k, v in id_to_token.items()
        }
        ids: List[int] = []
        for c in text:
            tid = token_to_id.get(c, 0)
            ids.append(tid)
        return ids

    def decode(self, token_ids: List[int]) -> str:
        vocab = self.get_vocab_dict()
        id_to_token: Dict[int, str] = {
            int(k): v for k, v in vocab.items()
        }
        chars: List[str] = []
        for tid in token_ids:
            t = id_to_token.get(tid, "")
            chars.append(t)
        return "".join(chars)

    def get_vocab_dict(self) -> Dict[str, str]:
        if self.vocab_data is not None:
            return self.vocab_data
        path = self.get_path_to_vocabulary_json()
        with open(path, encoding="utf-8") as f:
            vocab: Dict[str, str] = json.load(f)
        return vocab

    @property
    def eos_token_id(self) -> int:
        return 2

    @property
    def vocab_size(self) -> int:
        vocab = self.get_vocab_dict()
        return len(vocab)

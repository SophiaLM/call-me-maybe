from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.utils import cached_file


SMALL_LLM_MODEL_ID = "Qwen/Qwen3-0.6B"


class Small_LLM_Model:
    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        model_id = model_dir if model_dir else SMALL_LLM_MODEL_ID
        self.model_dir = model_id
        self.device = device
        self.vocab_cache: Optional[str] = None
        self._tokenizer: Any = None
        self._model: Any = None

    def _lazy_init(self) -> None:
        if self._tokenizer is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_dir
        )
        self._model.eval()

    def get_path_to_vocabulary_json(self) -> str:
        if self.vocab_cache is not None:
            return self.vocab_cache
        path = cached_file(self.model_dir, "vocab.json")
        self.vocab_cache = str(path)
        return self.vocab_cache

    def get_logits_from_input_ids(
        self,
        input_ids: np.ndarray,
        past_key_values=None,
    ) -> tuple:
        self._lazy_init()
        assert self._model is not None
        if input_ids.ndim == 1:
            input_ids = input_ids.reshape(1, -1)
        inp = torch.tensor(
            input_ids, dtype=torch.long, device=self.device
        )
        with torch.no_grad():
            outputs = self._model(
                inp,
                past_key_values=past_key_values,
                use_cache=True,
            )
        logits = outputs.logits[0, -1, :].float().cpu().numpy()
        new_past = outputs.past_key_values
        return logits, new_past  # type: ignore[no-any-return]

    def encode(self, text: str) -> List[int]:
        self._lazy_init()
        assert self._tokenizer is not None
        return self._tokenizer.encode(text)  # type: ignore[no-any-return]

    def decode(self, token_ids: List[int]) -> str:
        self._lazy_init()
        assert self._tokenizer is not None
        return self._tokenizer.decode(token_ids)  # type: ignore[no-any-return]

    def get_vocab_dict(self) -> Dict[str, str]:
        path = self.get_path_to_vocabulary_json()
        with open(path, encoding="utf-8") as f:
            raw: Dict[str, int] = json.load(f)
        return {str(v): k for k, v in raw.items()}

    @property
    def eos_token_id(self) -> int:
        self._lazy_init()
        assert self._tokenizer is not None
        return self._tokenizer.eos_token_id  # type: ignore[no-any-return]

    @property
    def vocab_size(self) -> int:
        self._lazy_init()
        assert self._tokenizer is not None
        return len(self._tokenizer)

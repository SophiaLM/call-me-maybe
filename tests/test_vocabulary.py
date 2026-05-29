import json
import tempfile

import pytest

from src.vocabulary import Vocabulary
from src.errors import VocabularyError


SAMPLE_VOCAB = {
    "0": "<unk>",
    "1": "<s>",
    "2": "</s>",
    "3": " ",
    "4": "!",
    "5": "\"",
    "6": "{",
    "7": "}",
    "8": ":",
    "9": ",",
    "10": "a",
    "11": "b",
    "12": "c",
    "13": "f",
    "14": "n",
    "15": "u",
}


@pytest.fixture
def vocab_file():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(SAMPLE_VOCAB, f)
        path = f.name
    yield path
    import os
    os.unlink(path)


class TestVocabulary:
    def test_load(self, vocab_file):
        v = Vocabulary(vocab_file)
        assert v.size() == len(SAMPLE_VOCAB)

    def test_token_to_string(self, vocab_file):
        v = Vocabulary(vocab_file)
        assert v.token_to_string(0) == "<unk>"
        assert v.token_to_string(5) == '"'
        assert v.token_to_string(999) == ""

    def test_string_to_token_ids(self, vocab_file):
        v = Vocabulary(vocab_file)
        assert v.string_to_token_ids("a") == [10]
        assert v.string_to_token_ids("nonexistent") == []

    def test_get_tokens_by_first_char(self, vocab_file):
        v = Vocabulary(vocab_file)
        tokens = v.get_tokens_by_first_char("f")
        assert 13 in tokens
        tokens = v.get_tokens_by_first_char("z")
        assert tokens == []

    def test_get_tokens_by_prefix(self, vocab_file):
        v = Vocabulary(vocab_file)
        tokens = v.get_tokens_by_prefix("f")
        assert 13 in tokens
        tokens = v.get_tokens_by_prefix("fa")
        assert tokens == set()

    def test_get_tokens_by_empty_prefix(self, vocab_file):
        v = Vocabulary(vocab_file)
        assert v.get_tokens_by_prefix("") == set()

    def test_all_token_ids(self, vocab_file):
        v = Vocabulary(vocab_file)
        ids = v.all_token_ids()
        assert all(isinstance(tid, int) for tid in ids)
        assert set(ids) == set(int(k) for k in SAMPLE_VOCAB)

    def test_repr(self, vocab_file):
        v = Vocabulary(vocab_file)
        assert "Vocabulary(size=" in repr(v)

    def test_load_nonexistent_file(self):
        with pytest.raises(VocabularyError):
            Vocabulary("/nonexistent/path/vocab.json")

    def test_load_empty_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("{}")
            path = f.name
        try:
            with pytest.raises(VocabularyError):
                Vocabulary(path)
        finally:
            import os
            os.unlink(path)

    def test_load_invalid_json(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("not json")
            path = f.name
        try:
            with pytest.raises(VocabularyError):
                Vocabulary(path)
        finally:
            import os
            os.unlink(path)

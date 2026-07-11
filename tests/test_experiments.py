from tests.tiny import TinyDecoder

from jlens_scaling.experiments.common import (
    format_prompt,
    greedy_first_word,
    greedy_next_token,
)
from jlens_scaling.experiments.verbal_report import _is_degenerate


def test_is_degenerate_flags_echoes_punctuation_stopwords():
    assert _is_degenerate(' "', "beverage")
    assert _is_degenerate(" fruit", "fruit")
    assert _is_degenerate(" fruits", "fruit")
    assert _is_degenerate(" a", "instrument")
    assert not _is_degenerate(" blue", "color")
    assert not _is_degenerate(" Earth", "planet")


def test_format_prompt_raw_passthrough():
    tok = TinyDecoder().tokenizer
    assert format_prompt(tok, "hello", chat=False) == "hello"


def test_greedy_next_token_returns_valid_id():
    model = TinyDecoder(n_layers=4, d_model=8)
    tid = greedy_next_token(model, "abcdefghij" * 5)
    assert 0 <= tid < 32


def test_greedy_first_word_returns_clean_word():
    model = TinyDecoder(n_layers=4, d_model=8)
    word = greedy_first_word(model, "abcdefghij" * 5)
    assert isinstance(word, str)
    assert " " not in word
    # deterministic model -> deterministic word
    assert word == greedy_first_word(model, "abcdefghij" * 5)

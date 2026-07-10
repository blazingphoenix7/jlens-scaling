from tests.tiny import TinyDecoder

from jlens_scaling.experiments.common import format_prompt, greedy_next_token


def test_format_prompt_raw_passthrough():
    tok = TinyDecoder().tokenizer
    assert format_prompt(tok, "hello", chat=False) == "hello"


def test_greedy_next_token_returns_valid_id():
    model = TinyDecoder(n_layers=4, d_model=8)
    tid = greedy_next_token(model, "abcdefghij" * 5)
    assert 0 <= tid < 32

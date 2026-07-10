from tests.tiny import TinyDecoder

from jlens_scaling.corpus import _select


def fake_texts():
    for i in range(500):
        yield f"document {i} " + ("lorem ipsum " * (i % 40))


def test_select_deterministic_and_filtered():
    tok = TinyDecoder().tokenizer
    a = _select(fake_texts(), tok, n=10, min_tokens=64, seed=0)
    b = _select(fake_texts(), tok, n=10, min_tokens=64, seed=0)
    c = _select(fake_texts(), tok, n=10, min_tokens=64, seed=1)
    assert a == b
    assert a != c
    assert len(a) == 10
    for text in a:
        assert tok(text, max_length=4096).input_ids.shape[1] >= 64

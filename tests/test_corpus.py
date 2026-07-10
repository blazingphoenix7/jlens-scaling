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


class _ListTokenizer:
    """HF-style: __call__ without return_tensors gives input_ids as a flat list."""

    def __call__(self, text, *, truncation=True, max_length=4096, **_kw):
        from types import SimpleNamespace

        ids = [1] * min(len(text.split()), max_length)
        return SimpleNamespace(input_ids=ids)


def test_select_handles_list_input_ids():
    got = _select(fake_texts(), _ListTokenizer(), n=5, min_tokens=30, seed=0)
    assert len(got) == 5

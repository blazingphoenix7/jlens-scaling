import torch

from jlens import fit
from tests.tiny import TinyDecoder

from jlens_scaling.structure import (
    layer_cka_matrix,
    lens_concentration,
    linear_cka,
    mid_block,
)

PROMPTS = ["abcdefghijklmnopqrstuvwxyz" * 4] * 2
TEXTS = ["the quick brown fox jumps over the lazy dog " * 8] * 4


def test_cka_self_is_one():
    x = torch.randn(50, 8)
    assert abs(linear_cka(x, x) - 1.0) < 1e-5


def test_cka_orthogonal_invariance():
    x = torch.randn(60, 8)
    q, _ = torch.linalg.qr(torch.randn(8, 8))
    assert abs(linear_cka(x, x @ q) - 1.0) < 1e-4


def test_layer_cka_matrix_shape_and_diagonal():
    model = TinyDecoder(n_layers=5, d_model=8)
    cka = layer_cka_matrix(model, TEXTS, skip_first=4)
    assert cka.shape == (5, 5)
    assert torch.allclose(torch.diagonal(cka), torch.ones(5), atol=1e-4)


def test_mid_block_recovers_planted_block():
    cka = torch.full((8, 8), 0.2)
    cka[2:6, 2:6] = 0.9
    cka.fill_diagonal_(1.0)
    assert mid_block(cka) == (2, 6)


def test_lens_concentration_fraction_in_unit_interval():
    model = TinyDecoder(n_layers=6, d_model=8)
    lens = fit(model, PROMPTS, dim_batch=4)
    out = lens_concentration(lens, model, TEXTS, k_frac=0.25, skip_first=4)
    assert 0.0 < out["variance_fraction"] <= 1.0
    assert out["k"] == 2
    assert out["band"]

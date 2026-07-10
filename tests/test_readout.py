import torch

from jlens import fit
from tests.tiny import TinyDecoder

from jlens_scaling.metrics import (
    band_layers,
    first_token_id,
    min_band_rank,
    token_rank,
    token_variants,
)
from jlens_scaling.readout import RankGrid, rank_grid

PROMPTS = ["abcdefghijklmnopqrstuvwxyz" * 4, "the quick brown fox " * 10]


def _tiny_lens_and_model():
    model = TinyDecoder(n_layers=6, d_model=8)
    lens = fit(model, PROMPTS, dim_batch=4)
    return lens, model


def test_token_rank_matches_sort():
    logits = torch.tensor([0.1, 3.0, 2.0, -1.0])
    assert token_rank(logits, 1) == 1
    assert token_rank(logits, 2) == 2
    assert token_rank(logits, 3) == 4


def test_band_layers_middle_third():
    assert band_layers(12) == [4, 5, 6, 7]
    assert band_layers(28) == [9, 10, 11, 12, 13, 14, 15, 16, 17]


def test_rank_grid_shapes_and_roundtrip():
    lens, model = _tiny_lens_and_model()
    grid = rank_grid(lens, model, PROMPTS[0], target_ids=[3, 5])
    n_pos = model.encode(PROMPTS[0]).shape[1]
    assert grid.layers == lens.source_layers
    assert len(grid.tokens) == n_pos
    for tid in (3, 5):
        assert len(grid.ranks[tid]) == len(grid.layers)
        assert all(len(row) == n_pos for row in grid.ranks[tid])
        assert all(r >= 1 for row in grid.ranks[tid] for r in row)
    grid2 = RankGrid.from_json(grid.to_json())
    assert grid2.ranks == grid.ranks and grid2.layers == grid.layers


def test_min_band_rank():
    lens, model = _tiny_lens_and_model()
    grid = rank_grid(lens, model, PROMPTS[0], target_ids=[3])
    band = band_layers(model.n_layers)
    band = [l for l in band if l in set(grid.layers)]
    got = min_band_rank(grid, 3, band)
    rows = [grid.ranks[3][grid.layers.index(l)] for l in band]
    assert got == min(min(r) for r in rows)


def test_first_token_id_and_variants():
    tok = TinyDecoder().tokenizer
    assert isinstance(first_token_id(tok, "paris"), int)
    variants = token_variants(tok, "paris")
    assert len(variants) >= 1
    assert all(isinstance(v, int) for v in variants)
    assert first_token_id(tok, "paris") in variants

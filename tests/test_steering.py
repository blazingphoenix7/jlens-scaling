import torch

from jlens import fit
from tests.tiny import TinyDecoder

from jlens_scaling.steering import ResidualEdit, apply_edits, token_direction

PROMPTS = ["abcdefghijklmnopqrstuvwxyz" * 4] * 2


def _setup():
    model = TinyDecoder(n_layers=6, d_model=8)
    lens = fit(model, PROMPTS, dim_batch=4)
    return model, lens


def test_no_edits_matches_plain_forward():
    model, _ = _setup()
    ids = model.encode("abcdefghij" * 3)
    logits = apply_edits(model, ids, edits=[])
    from jlens.hooks import ActivationRecorder

    with ActivationRecorder(model.layers, at=[model.n_layers - 1]) as rec:
        model.forward(ids)
        expected = model.unembed(rec.activations[model.n_layers - 1][0].float())
    assert torch.allclose(logits, expected, atol=1e-5)


def test_addition_edit_changes_logits_only_at_and_after_position():
    model, lens = _setup()
    ids = model.encode("abcdefghij" * 3)
    d = token_direction(lens, model, token_id=3, layer=2)
    edit = ResidualEdit(
        layer=2, positions=[5], direction_out=None, direction_in=d, scale=4.0
    )
    edited = apply_edits(model, ids, edits=[edit])
    plain = apply_edits(model, ids, edits=[])
    assert not torch.allclose(edited[5], plain[5], atol=1e-5)
    assert torch.allclose(edited[:5], plain[:5], atol=1e-5)


def test_swap_edit_runs_and_preserves_shape():
    model, lens = _setup()
    ids = model.encode("abcdefghij" * 3)
    d_out = token_direction(lens, model, token_id=3, layer=2)
    d_in = token_direction(lens, model, token_id=7, layer=2)
    edit = ResidualEdit(layer=2, positions=None, direction_out=d_out, direction_in=d_in)
    edited = apply_edits(model, ids, edits=[edit])
    assert edited.shape == apply_edits(model, ids, edits=[]).shape


def test_hooks_removed_after_apply():
    model, lens = _setup()
    ids = model.encode("abcdefghij" * 3)
    d = token_direction(lens, model, token_id=3, layer=2)
    apply_edits(model, ids, edits=[ResidualEdit(2, None, None, d, 2.0)])
    a = apply_edits(model, ids, edits=[])
    b = apply_edits(model, ids, edits=[])
    assert torch.allclose(a, b)


def test_token_direction_is_unit_norm():
    model, lens = _setup()
    d = token_direction(lens, model, token_id=3, layer=2)
    assert abs(d.norm().item() - 1.0) < 1e-5

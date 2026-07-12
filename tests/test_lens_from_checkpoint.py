import torch

from jlens import fit
from tests.tiny import TinyDecoder

from scripts.lens_from_checkpoint import lens_from_checkpoint

PROMPTS = ["abcdefghijklmnopqrstuvwxyz" * 4] * 3


def test_rebuilt_lens_matches_completed_fit(tmp_path):
    model = TinyDecoder(n_layers=6, d_model=8)
    ckpt = tmp_path / "ckpt.pt"
    lens = fit(model, PROMPTS, dim_batch=4, checkpoint_path=str(ckpt))
    rebuilt = lens_from_checkpoint(str(ckpt))
    assert rebuilt.n_prompts == lens.n_prompts
    assert rebuilt.source_layers == lens.source_layers
    for layer in lens.source_layers:
        assert torch.allclose(
            rebuilt.jacobians[layer], lens.jacobians[layer], atol=1e-6
        )

import json

from tests.tiny import TinyDecoder

from jlens_scaling.config import FitConfig
from scripts.fit_lens import fit_and_save

PROMPTS = ["abcdefghijklmnopqrstuvwxyz" * 4] * 3


def test_fit_and_save_writes_artifacts(tmp_path):
    model = TinyDecoder(n_layers=6, d_model=8)
    fit_cfg = FitConfig(
        n_prompts=3, max_seq_len=128, dim_batch=4,
        corpus="wikitext-103", corpus_seed=0, min_tokens=64,
    )
    out = fit_and_save(
        model, PROMPTS, fit_cfg, out_dir=str(tmp_path),
        meta={"model_id": "tiny", "slug": "tiny"},
    )
    assert (tmp_path / "lens.pt").exists()
    meta = json.loads((tmp_path / "fit_meta.json").read_text(encoding="utf-8"))
    assert meta["n_prompts_fitted"] == 3
    assert meta["model_id"] == "tiny"
    assert out.n_prompts == 3
    assert out.source_layers == list(range(5))  # all layers below final

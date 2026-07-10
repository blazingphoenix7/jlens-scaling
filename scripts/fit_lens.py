"""Fit a Jacobian lens per the run config. Resumable; CPU-safe.

Usage:
    python scripts/fit_lens.py --config configs/gpt2-small.yaml
    python scripts/fit_lens.py --config configs/qwen3-0.6b.yaml --n-prompts 25
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

import jlens
import torch
import transformers

import jlens_scaling
from jlens_scaling.config import FitConfig, load_config
from jlens_scaling.corpus import fitting_prompts
from jlens_scaling.models import load_lens_model


def fit_and_save(model, prompts, fit_cfg: FitConfig, *, out_dir: str, meta: dict):
    os.makedirs(out_dir, exist_ok=True)
    start = time.perf_counter()
    lens = jlens.fit(
        model,
        prompts,
        dim_batch=fit_cfg.dim_batch,
        max_seq_len=fit_cfg.max_seq_len,
        checkpoint_path=os.path.join(out_dir, "ckpt.pt"),
        checkpoint_every=1,
        resume=True,
    )
    lens.save(os.path.join(out_dir, "lens.pt"))
    full_meta = {
        **meta,
        "n_prompts_requested": fit_cfg.n_prompts,
        "n_prompts_fitted": lens.n_prompts,
        "dim_batch": fit_cfg.dim_batch,
        "max_seq_len": fit_cfg.max_seq_len,
        "skip_first": jlens.fitting.SKIP_FIRST_N_POSITIONS,
        "corpus": fit_cfg.corpus,
        "corpus_seed": fit_cfg.corpus_seed,
        "min_tokens": fit_cfg.min_tokens,
        "source_layers": lens.source_layers,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "jlens_scaling_version": jlens_scaling.__version__,
        "wall_seconds": round(time.perf_counter() - start, 1),
    }
    with open(os.path.join(out_dir, "fit_meta.json"), "w", encoding="utf-8") as f:
        json.dump(full_meta, f, indent=2)
    return lens


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--n-prompts", type=int, default=None)
    parser.add_argument("--out", default="artifacts")
    args = parser.parse_args()

    cfg = load_config(args.config)
    fit_cfg = cfg.fit
    if args.n_prompts is not None:
        from dataclasses import replace

        fit_cfg = replace(fit_cfg, n_prompts=args.n_prompts)

    model = load_lens_model(cfg.model)
    prompts = fitting_prompts(fit_cfg, model.tokenizer)
    out_dir = os.path.join(args.out, cfg.model.slug)
    fit_and_save(
        model,
        prompts,
        fit_cfg,
        out_dir=out_dir,
        meta={
            "model_id": cfg.model.model_id,
            "revision_requested": cfg.model.revision,
            "revision_resolved": getattr(model, "resolved_revision", None),
            "slug": cfg.model.slug,
        },
    )
    print(f"lens written to {out_dir}/lens.pt")


if __name__ == "__main__":
    main()

"""Compute pre-registered structural metrics for a fitted rung.

Usage:
    python scripts/run_structure.py --config configs/gpt2-small.yaml --lens artifacts/gpt2-small/lens.pt
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace

from jlens import JacobianLens

from jlens_scaling.config import load_config
from jlens_scaling.corpus import fitting_prompts
from jlens_scaling.models import load_lens_model
from jlens_scaling.structure import layer_cka_matrix, lens_concentration, mid_block

HELDOUT_SEED = 1
HELDOUT_DOCS = 200


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--lens", required=True)
    parser.add_argument("--out", default="results")
    parser.add_argument("--n-docs", type=int, default=HELDOUT_DOCS)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = load_lens_model(cfg.model)
    lens = JacobianLens.from_pretrained(args.lens)

    heldout_cfg = replace(
        cfg.fit, corpus_seed=HELDOUT_SEED, n_prompts=args.n_docs
    )
    texts = fitting_prompts(heldout_cfg, model.tokenizer)

    cka = layer_cka_matrix(model, texts, max_seq_len=cfg.fit.max_seq_len)
    block = mid_block(cka)
    concentration = lens_concentration(
        lens, model, texts, max_seq_len=cfg.fit.max_seq_len
    )

    out_dir = os.path.join(args.out, cfg.model.slug)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "structure.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "heldout": {"corpus": cfg.fit.corpus, "seed": HELDOUT_SEED,
                            "n_docs": args.n_docs},
                "cka": [[round(v, 6) for v in row] for row in cka.tolist()],
                "mid_block": list(block),
                "concentration": concentration,
            },
            f,
            indent=2,
        )
    print(f"mid_block={block} concentration={concentration['variance_fraction']:.3f}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

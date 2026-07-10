"""Run a readout experiment against a fitted lens.

Usage:
    python scripts/run_experiment.py --config configs/gpt2-small.yaml \
        --experiment verbal_report --lens artifacts/gpt2-small/lens.pt
"""

from __future__ import annotations

import argparse
import os

from jlens import JacobianLens

from jlens_scaling.config import load_config
from jlens_scaling.models import load_lens_model

DATA = {
    "verbal_report": "data/anthropic/verbal-report.json",
    "two_hop": "data/anthropic/probe-swap.json",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment", required=True, choices=sorted(DATA))
    parser.add_argument("--lens", required=True)
    parser.add_argument("--out", default="results")
    parser.add_argument("--max-items", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = load_lens_model(cfg.model)
    lens = JacobianLens.from_pretrained(args.lens)
    out_dir = os.path.join(args.out, cfg.model.slug)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.experiment}.json")

    if args.experiment == "verbal_report":
        from jlens_scaling.experiments import verbal_report

        result = verbal_report.run(
            lens, model, DATA[args.experiment], chat=cfg.model.chat, out_path=out_path
        )
        print(f"hit rate (top-5 in band): {result['report_hit_rate_top5']:.2f}")
    elif args.experiment == "two_hop":
        from jlens_scaling.experiments import two_hop

        result = two_hop.run(
            lens,
            model,
            DATA[args.experiment],
            chat=cfg.model.chat,
            out_path=out_path,
            max_items=args.max_items,
        )
        print(
            f"baseline acc: {result['baseline_accuracy']:.2f}  "
            f"intermediate readable (top-5 in band, correct items): "
            f"{result['intermediate_hit_rate_top5']:.2f}"
        )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

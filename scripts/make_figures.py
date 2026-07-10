"""Render the showcase two-hop heatmap from a results JSON.

Usage:
    python scripts/make_figures.py --results results/gpt2-small/two_hop.json --out figures --slug gpt2-small
"""

from __future__ import annotations

import argparse
import json
import os

from jlens_scaling.viz import rank_heatmap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", default="figures")
    parser.add_argument("--slug", required=True)
    args = parser.parse_args()

    with open(args.results, encoding="utf-8") as f:
        result = json.load(f)
    correct = [i for i in result["per_item"] if i["baseline_correct"]]
    pool = correct or result["per_item"]
    best = min(pool, key=lambda i: i["intermediate_band_min_rank"])
    os.makedirs(args.out, exist_ok=True)
    out_png = os.path.join(args.out, f"{args.slug}-two-hop.png")
    rank_heatmap(
        best["grid"],
        best["intermediate_token_id"],
        f"{args.slug}: lens rank of silent intermediate "
        f"'{best['intermediate']}' (band min={best['intermediate_band_min_rank']})",
        out_png,
    )
    print(f"wrote {out_png} (item: {best['prompt'][:60]}...)")


if __name__ == "__main__":
    main()

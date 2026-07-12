"""Assemble per-rung results into the cross-scale table (results/scaling.json).

Implements the preregistration's capability-confound protocol: primary
metrics are reported over all items and over the capability-matched subset
(two-hop items answered correctly by every rung present), flagged
underpowered when that intersection has fewer than 10 items.
"""

from __future__ import annotations

import argparse
import json
import os

PARAMS = {
    "gpt2-small": (124_000_000, "gpt2", "base"),
    "gpt2-medium": (355_000_000, "gpt2", "base"),
    "qwen3-0.6b": (600_000_000, "qwen3", "instruct"),
    "qwen3-1.7b": (1_700_000_000, "qwen3", "instruct"),
    "qwen3-1.7b-base": (1_700_000_000, "qwen3", "base"),
}


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def aggregate(results_dir: str = "results") -> dict:
    rungs = []
    per_slug_items: dict[str, dict] = {}
    for slug in sorted(os.listdir(results_dir)):
        rung_dir = os.path.join(results_dir, slug)
        two_hop_path = os.path.join(rung_dir, "two_hop.json")
        if slug not in PARAMS or not os.path.exists(two_hop_path):
            continue
        params, family, training = PARAMS[slug]
        two_hop = _load(two_hop_path)
        verbal = _load(os.path.join(rung_dir, "verbal_report.json"))
        row = {
            "slug": slug,
            "params": params,
            "family": family,
            "training": training,
            "two_hop_accuracy": two_hop["baseline_accuracy"],
            "readability": two_hop["intermediate_hit_rate_top5"],
            "swap_top1": two_hop.get("swap_success_rate_top1"),
            "swap_top5": two_hop.get("swap_success_rate_top5"),
            "verbal_valid": verbal["n_valid_answers"],
            "verbal_hit_rate_valid": verbal["report_hit_rate_top5_valid"],
        }
        structure_path = os.path.join(rung_dir, "structure.json")
        if os.path.exists(structure_path):
            conc = _load(structure_path)["concentration"]
            row["concentration_ratio"] = (
                conc["variance_fraction"] / conc["k_frac"]
            )
        rungs.append(row)
        per_slug_items[slug] = {
            item["name"]: item
            for item in two_hop["per_item"]
            if item["baseline_correct"]
        }

    matched = {}
    if per_slug_items:
        shared = set.intersection(*(set(v) for v in per_slug_items.values()))
        shared_names = sorted(shared)
        matched = {
            "item_names": shared_names,
            "n": len(shared_names),
            "underpowered": len(shared_names) < 10,
            "readability_by_slug": {
                slug: (
                    sum(
                        items[n]["intermediate_band_min_rank"] <= 5
                        for n in shared_names
                    ) / len(shared_names)
                    if shared_names
                    else None
                )
                for slug, items in per_slug_items.items()
            },
            "swap_top1_by_slug": {
                slug: (
                    sum(
                        bool(items[n].get("swap_success_top1"))
                        for n in shared_names
                    ) / len(shared_names)
                    if shared_names
                    else None
                )
                for slug, items in per_slug_items.items()
            },
        }

    return {"rungs": rungs, "capability_matched": matched}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/scaling.json")
    args = parser.parse_args()
    out = aggregate(args.results)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    for row in out["rungs"]:
        print(row["slug"], row)
    print("matched:", out["capability_matched"].get("n"),
          "underpowered:", out["capability_matched"].get("underpowered"))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

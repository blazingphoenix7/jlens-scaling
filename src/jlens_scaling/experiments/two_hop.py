"""Two-hop latent reasoning, readout half ('silent intermediate').

For each two-hop prompt (e.g. '... the language spoken in the country where
the Amazon River ends is'), measure (a) whether the model answers correctly,
and (b) the lens rank of the *intermediate* entity (Brazil) over the band —
content the model never says. Causal swaps land in Phase 2.

Tokenizer robustness: upstream prompts end with a trailing space, so target
words may surface with or without the leading-space BPE marker. Correctness
accepts either first-token variant; ranks take the min over variants.
"""

from __future__ import annotations

import json

from jlens_scaling.experiments.common import format_prompt, greedy_next_token
from jlens_scaling.metrics import band_layers, min_band_rank, token_variants
from jlens_scaling.readout import rank_grid


def run(
    lens,
    model,
    data_path: str,
    *,
    chat: bool,
    out_path: str,
    max_items: int | None = None,
) -> dict:
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)
    items = data["items"] if isinstance(data, dict) and "items" in data else data
    if max_items is not None:
        items = items[:max_items]

    band = [l for l in band_layers(model.n_layers) if l in set(lens.source_layers)]
    per_item = []
    n_correct = 0
    n_hit = 0
    for item in items:
        prompt = format_prompt(model.tokenizer, item["prompt"], chat)
        answer_ids = token_variants(model.tokenizer, item["answer"])
        intermediate_ids = token_variants(model.tokenizer, item["intermediate"])
        greedy_id = greedy_next_token(model, prompt)
        baseline_correct = greedy_id in answer_ids

        grid = rank_grid(
            lens, model, prompt, target_ids=sorted(set(answer_ids + intermediate_ids))
        )
        inter_by_variant = {
            tid: min_band_rank(grid, tid, band) for tid in intermediate_ids
        }
        best_intermediate_id = min(inter_by_variant, key=inter_by_variant.get)
        inter_rank = inter_by_variant[best_intermediate_id]
        ans_rank = min(min_band_rank(grid, tid, band) for tid in answer_ids)
        if baseline_correct:
            n_correct += 1
            n_hit += int(inter_rank <= 5)
        per_item.append(
            {
                "name": item.get("name"),
                "category": item.get("category"),
                "prompt": item["prompt"],
                "answer": item["answer"],
                "intermediate": item["intermediate"],
                "baseline_correct": baseline_correct,
                "greedy_token": model.tokenizer.decode([greedy_id]),
                "intermediate_band_min_rank": inter_rank,
                "answer_band_min_rank": ans_rank,
                "intermediate_token_id": best_intermediate_id,
                "grid": grid.to_json(),
            }
        )

    result = {
        "experiment": "two_hop",
        "band": band,
        "n_items": len(per_item),
        "baseline_accuracy": n_correct / max(len(per_item), 1),
        "intermediate_hit_rate_top5": n_hit / max(n_correct, 1),
        "per_item": per_item,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result

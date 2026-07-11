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
from jlens_scaling.metrics import band_layers, min_band_rank, token_rank, token_variants
from jlens_scaling.readout import rank_grid
from jlens_scaling.steering import ResidualEdit, apply_edits, token_direction


def _swap_item(
    lens, model, prompt: str, band: list[int], out_word: str, in_word: str,
    swap_answer: str, *, scale: float,
) -> dict:
    """Causal arm: swap out_word->in_word lens directions across the band at
    every prompt position; grade the final-position prediction against
    swap_answer (both token variants)."""
    out_id = token_variants(model.tokenizer, out_word)[0]
    in_id = token_variants(model.tokenizer, in_word)[0]
    edits = [
        ResidualEdit(
            layer=layer,
            positions=None,
            direction_out=token_direction(lens, model, out_id, layer),
            direction_in=token_direction(lens, model, in_id, layer),
            scale=scale,
        )
        for layer in band
    ]
    input_ids = model.encode(prompt)
    logits = apply_edits(model, input_ids, edits)  # [seq, vocab]
    final = logits[-1]
    greedy_id = int(final.argmax().item())
    swap_ids = token_variants(model.tokenizer, swap_answer)
    swap_rank = min(token_rank(final, tid) for tid in swap_ids)
    return {
        "swap_greedy_token": model.tokenizer.decode([greedy_id]),
        "swap_success_top1": greedy_id in swap_ids,
        "swap_success_top5": swap_rank <= 5,
        "swap_answer_rank": swap_rank,
    }


def run(
    lens,
    model,
    data_path: str,
    *,
    chat: bool,
    out_path: str,
    max_items: int | None = None,
    swap: bool = False,
    swap_scale: float = 1.0,
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
        swap_fields: dict = {}
        if baseline_correct:
            n_correct += 1
            n_hit += int(inter_rank <= 5)
            if swap:
                swap_fields = _swap_item(
                    lens, model, prompt, band,
                    item["intermediate"], item["swap_to"], item["swap_answer"],
                    scale=swap_scale,
                )
        per_item.append(
            {
                **swap_fields,
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
    if swap:
        swapped = [i for i in per_item if "swap_success_top1" in i]
        result["swap_scale"] = swap_scale
        result["n_swapped"] = len(swapped)
        result["swap_success_rate_top1"] = (
            sum(i["swap_success_top1"] for i in swapped) / len(swapped)
            if swapped
            else 0.0
        )
        result["swap_success_rate_top5"] = (
            sum(i["swap_success_top5"] for i in swapped) / len(swapped)
            if swapped
            else 0.0
        )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result

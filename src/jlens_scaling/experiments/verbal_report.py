"""Verbal report (paper section: 'think of a {category}').

Protocol (readout half of the upstream experiment): prompt the model with the
paper's template per category, take its greedy one-word answer, then ask
whether that answer token was already readable in the J-lens band at
pre-answer positions. Phase 1 is descriptive; the causal swap half lands in
Phase 2.
"""

from __future__ import annotations

import json

from jlens_scaling.experiments.common import format_prompt, greedy_next_token
from jlens_scaling.metrics import band_layers, min_band_rank
from jlens_scaling.readout import rank_grid

TEMPLATE = "Think of a {category}. Answer in one word:"


def run(lens, model, data_path: str, *, chat: bool, out_path: str) -> dict:
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)
    candidates = data["candidates"] if "candidates" in data else data
    template = data.get("template", TEMPLATE) if isinstance(data, dict) else TEMPLATE

    band = [l for l in band_layers(model.n_layers) if l in set(lens.source_layers)]
    per_category = {}
    hits = 0
    for category in sorted(candidates):
        prompt = format_prompt(model.tokenizer, template.format(category=category), chat)
        answer_id = greedy_next_token(model, prompt)
        answer_str = model.tokenizer.decode([answer_id])
        grid = rank_grid(lens, model, prompt, target_ids=[answer_id])
        band_min = min_band_rank(grid, answer_id, band)
        hit = band_min <= 5
        hits += int(hit)
        per_category[category] = {
            "answer_token": answer_str,
            "band_min_rank": band_min,
            "hit_top5": hit,
            "grid": grid.to_json(),
        }
    result = {
        "experiment": "verbal_report",
        "template": template,
        "band": band,
        "n_categories": len(per_category),
        "report_hit_rate_top5": hits / max(len(per_category), 1),
        "per_category": per_category,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result

"""Verbal report (paper section: 'think of a {category}').

Protocol (readout half of the upstream experiment): prompt the model with the
paper's template per category, take its greedy one-word answer, then ask
whether that answer token was already readable in the J-lens band at
pre-answer positions. Phase 1 is descriptive; the causal swap half lands in
Phase 2.
"""

from __future__ import annotations

import json

from jlens_scaling.experiments.common import format_prompt, greedy_first_word
from jlens_scaling.metrics import band_layers, min_band_rank, token_variants
from jlens_scaling.readout import rank_grid

TEMPLATE = "Think of a {category}. Answer in one word:"

_STOPWORDS = {"a", "an", "the", "one", "it"}


def _is_degenerate(answer_token: str, category: str) -> bool:
    """True when the greedy 'answer' is not a real category exemplar: punctuation
    or non-alphabetic tokens, stopwords, or an echo of the category word itself
    (base models often do all three). Degenerate answers trivially self-read in
    the lens and must not count as report evidence."""
    word = answer_token.strip().lower()
    cat = category.strip().lower()
    if not word.isalpha() or len(word) < 2:
        return True
    if word in _STOPWORDS:
        return True
    return word in cat or cat in word


def run(lens, model, data_path: str, *, chat: bool, out_path: str) -> dict:
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)
    candidates = data["candidates"] if "candidates" in data else data
    template = data.get("template", TEMPLATE) if isinstance(data, dict) else TEMPLATE

    band = [l for l in band_layers(model.n_layers) if l in set(lens.source_layers)]
    per_category = {}
    hits = 0
    valid_hits = 0
    n_valid = 0
    for category in sorted(candidates):
        prompt = format_prompt(model.tokenizer, template.format(category=category), chat)
        answer_word = greedy_first_word(model, prompt)
        degenerate = _is_degenerate(answer_word or "?", category)
        target_ids = token_variants(model.tokenizer, answer_word) if answer_word else []
        if not target_ids:
            degenerate = True
            target_ids = [0]
        grid = rank_grid(lens, model, prompt, target_ids=target_ids)
        band_min = min(min_band_rank(grid, tid, band) for tid in target_ids)
        hit = band_min <= 5
        hits += int(hit)
        if not degenerate:
            n_valid += 1
            valid_hits += int(hit)
        per_category[category] = {
            "answer_word": answer_word,
            "answer_token_variants": target_ids,
            "answer_degenerate": degenerate,
            "band_min_rank": band_min,
            "hit_top5": hit,
            "grid": grid.to_json(),
        }
    result = {
        "experiment": "verbal_report",
        "template": template,
        "band": band,
        "n_categories": len(per_category),
        "n_valid_answers": n_valid,
        "report_hit_rate_top5": hits / max(len(per_category), 1),
        "report_hit_rate_top5_valid": valid_hits / n_valid if n_valid else None,
        "per_category": per_category,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result

"""Pretraining-like fitting prompts, reproducibly sampled and cached.

The paper fits on ~1000 sequences of 128 tokens from a pretraining-like
distribution and notes quality saturates quickly (~100 prompts usable). We
stream wikitext-103 (raw), filter to documents with at least ``min_tokens``
tokens, and take a seeded sample from a fixed-size leading buffer so the
selection is independent of iteration timing.
"""

from __future__ import annotations

import json
import os
import random
from collections.abc import Iterable

from jlens_scaling.config import FitConfig

_DATASETS = {
    "wikitext-103": ("Salesforce/wikitext", "wikitext-103-raw-v1", "train", "text"),
}


def _select(
    texts: Iterable[str],
    tokenizer,
    *,
    n: int,
    min_tokens: int,
    seed: int,
    buffer_size: int = 2000,
) -> list[str]:
    kept: list[str] = []
    for text in texts:
        text = text.strip()
        if not text:
            continue
        n_tok = tokenizer(text, max_length=4096).input_ids.shape[1]
        if n_tok >= min_tokens:
            kept.append(text)
        if len(kept) >= buffer_size:
            break
    if len(kept) < n:
        raise ValueError(f"only {len(kept)} usable documents, need {n}")
    return random.Random(seed).sample(kept, n)


def fitting_prompts(
    fit: FitConfig, tokenizer, cache_dir: str = "data/fitting"
) -> list[str]:
    cache = os.path.join(
        cache_dir,
        f"{fit.corpus}-seed{fit.corpus_seed}-n{fit.n_prompts}-min{fit.min_tokens}.jsonl",
    )
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            return [json.loads(line)["text"] for line in f]

    import datasets  # local import: heavy

    repo, name, split, field = _DATASETS[fit.corpus]
    stream = datasets.load_dataset(repo, name, split=split, streaming=True)
    prompts = _select(
        (row[field] for row in stream),
        tokenizer,
        n=fit.n_prompts,
        min_tokens=fit.min_tokens,
        seed=fit.corpus_seed,
    )
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache, "w", encoding="utf-8") as f:
        for text in prompts:
            f.write(json.dumps({"text": text}) + "\n")
    return prompts

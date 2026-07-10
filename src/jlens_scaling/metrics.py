"""Pre-registered readout metrics (Phase 1: descriptive only).

Definitions follow the upstream data README: a *hit* at rank k means the
target token's lens rank is <= k at some (layer, position); the *band* is a
contiguous mid-network layer range (middle third here, formalized further in
Phase 2's metrics.md before any scaling comparisons).

Tokenizer note: several upstream prompts end with a trailing space, so a
target word's first token may appear either with or without the leading-space
BPE marker. ``token_variants`` returns both ids (deduplicated); rank metrics
take the min over variants and correctness checks accept either.
"""

from __future__ import annotations

import torch


def token_rank(logits_row: torch.Tensor, token_id: int) -> int:
    """1-indexed rank of ``token_id`` in a ``[vocab]`` logits row."""
    return int((logits_row > logits_row[token_id]).sum().item()) + 1


def band_layers(n_layers: int) -> list[int]:
    """Middle third of the residual stack."""
    return list(range(n_layers // 3, (2 * n_layers) // 3))


def min_band_rank(grid, target_id: int, band: list[int]) -> int:
    """Min rank of ``target_id`` over (band layers x all positions)."""
    layer_index = {layer: i for i, layer in enumerate(grid.layers)}
    rows = [grid.ranks[target_id][layer_index[l]] for l in band if l in layer_index]
    if not rows:
        raise ValueError(f"no band layers {band} present in grid layers {grid.layers}")
    return min(min(row) for row in rows)


def _encode_word(tokenizer, text: str) -> list[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except (TypeError, AttributeError):  # toy tokenizers without .encode kwargs
        return tokenizer(text).input_ids[0].tolist()


def token_variants(tokenizer, word: str) -> list[int]:
    """First token id of ``word`` under both space conventions, deduplicated."""
    ids: list[int] = []
    for candidate in (" " + word, word):
        encoded = _encode_word(tokenizer, candidate)
        if encoded and int(encoded[0]) not in ids:
            ids.append(int(encoded[0]))
    if not ids:
        raise ValueError(f"could not tokenize {word!r}")
    return ids


def first_token_id(tokenizer, word: str) -> int:
    """First token id of ``word`` under the leading-space BPE convention."""
    return token_variants(tokenizer, word)[0]

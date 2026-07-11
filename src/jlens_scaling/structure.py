"""Structural metrics: layer-pair CKA and lens concentration.

Pre-registered definitions (docs/preregistration-draft.md, metrics 4-5):
linear CKA between residual activations of every layer pair over held-out
documents; the mechanical mid-block rule; and the activation-variance
fraction captured by the top-k right singular vectors of the band-mean J_l.
"""

from __future__ import annotations

import torch

from jlens.hooks import ActivationRecorder

from jlens_scaling.metrics import band_layers

SKIP_FIRST_DEFAULT = 16


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Centered linear CKA between [n, d1] and [n, d2] activation matrices."""
    x = (x - x.mean(dim=0, keepdim=True)).double()
    y = (y - y.mean(dim=0, keepdim=True)).double()
    numerator = ((x.T @ y) ** 2).sum()
    denominator = torch.sqrt(((x.T @ x) ** 2).sum() * ((y.T @ y) ** 2).sum())
    return float(numerator / denominator)


@torch.no_grad()
def _pooled_activations(
    model, texts: list[str], *, max_seq_len: int = 128, skip_first: int
) -> dict[int, torch.Tensor]:
    """Per-layer residuals pooled over (document, position), fitting-style
    position masking (skip attention-sink prefix and the final position)."""
    layers = list(range(model.n_layers))
    collected: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    for text in texts:
        input_ids = model.encode(text, max_length=max_seq_len)
        if input_ids.shape[1] <= skip_first + 1:
            continue
        with ActivationRecorder(model.layers, at=layers) as recorder:
            model.forward(input_ids)
            for layer in layers:
                acts = recorder.activations[layer][0, skip_first:-1].float().cpu()
                collected[layer].append(acts)
    if not collected[0]:
        raise ValueError("no document long enough for the position mask")
    return {layer: torch.cat(chunks) for layer, chunks in collected.items()}


def layer_cka_matrix(
    model,
    texts: list[str],
    *,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_DEFAULT,
) -> torch.Tensor:
    acts = _pooled_activations(
        model, texts, max_seq_len=max_seq_len, skip_first=skip_first
    )
    n_layers = model.n_layers
    cka = torch.zeros(n_layers, n_layers)
    for i in range(n_layers):
        for j in range(i, n_layers):
            value = linear_cka(acts[i], acts[j])
            cka[i, j] = cka[j, i] = value
    return cka


def mid_block(cka: torch.Tensor) -> tuple[int, int]:
    """Pre-registered mechanical rule: over contiguous windows [i, j) with
    width >= 3, excluding layer 0 and the final layer, maximize within-window
    mean CKA minus window-to-outside mean CKA; ties -> wider, then lower i."""
    n_layers = cka.shape[0]
    best_key = None
    best_window = None
    for i in range(1, n_layers - 1):
        for j in range(i + 3, n_layers):
            within = cka[i:j, i:j].mean().item()
            outside_cols = list(range(0, i)) + list(range(j, n_layers))
            outside = cka[i:j, outside_cols].mean().item()
            key = (within - outside, j - i, -i)
            if best_key is None or key > best_key:
                best_key, best_window = key, (i, j)
    if best_window is None:
        raise ValueError(f"model too shallow for mid_block: {n_layers} layers")
    return best_window


def lens_concentration(
    lens,
    model,
    texts: list[str],
    *,
    k_frac: float = 0.1,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_DEFAULT,
) -> dict:
    """Variance fraction of held-out band residuals captured by the top-k
    right singular vectors of the band-mean Jacobian."""
    band = [l for l in band_layers(model.n_layers) if l in set(lens.source_layers)]
    j_mean = torch.stack([lens.jacobians[layer] for layer in band]).mean(dim=0)
    _, _, vh = torch.linalg.svd(j_mean.double(), full_matrices=False)
    k = max(1, int(round(k_frac * model.d_model)))
    v_k = vh[:k].T  # [d, k]

    acts = _pooled_activations(
        model, texts, max_seq_len=max_seq_len, skip_first=skip_first
    )
    band_resid = torch.cat([acts[layer] for layer in band]).double()
    band_resid = band_resid - band_resid.mean(dim=0, keepdim=True)
    projected = band_resid @ v_k
    fraction = float((projected**2).sum() / (band_resid**2).sum())
    return {"k": k, "k_frac": k_frac, "variance_fraction": fraction, "band": band}

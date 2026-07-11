"""Residual-stream interventions driven by lens directions.

The lens logit of token v at layer l is (J_l^T W_U[v]) . h, so the lens-space
direction for v is d_l(v) = normalize(J_l^T W_U[v]). A *swap* removes the
component of the residual along d_out and adds an equal-magnitude component
(times scale) along d_in at every edited position — the upstream probe-swap /
verbal-report clamping protocol, expressed with lens directions instead of
trained probes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ResidualEdit:
    layer: int
    positions: list[int] | None  # None = every position
    direction_out: torch.Tensor | None  # unit vector to remove; None = pure add
    direction_in: torch.Tensor | None  # unit vector to add; None = pure ablation
    scale: float = 1.0


def _unembed_weight(model) -> torch.Tensor:
    for attr in ("_lm_head", "lm_head"):
        head = getattr(model, attr, None)
        if head is not None and hasattr(head, "weight"):
            return head.weight
    raise AttributeError("could not locate unembedding weight on model")


def token_direction(lens, model, token_id: int, layer: int) -> torch.Tensor:
    w_v = _unembed_weight(model)[token_id].detach().float().cpu()
    d = lens.jacobians[layer].T @ w_v
    return d / d.norm()


def _edit_hidden(hidden: torch.Tensor, edit: ResidualEdit) -> torch.Tensor:
    # hidden: [batch, seq, d_model]
    positions = (
        list(range(hidden.shape[1])) if edit.positions is None else edit.positions
    )
    h = hidden.clone()
    device, dtype = h.device, h.dtype
    d_out = (
        edit.direction_out.to(device=device, dtype=dtype)
        if edit.direction_out is not None
        else None
    )
    d_in = (
        edit.direction_in.to(device=device, dtype=dtype)
        if edit.direction_in is not None
        else None
    )
    for p in positions:
        vec = h[:, p, :]
        if d_out is not None:
            coeff = vec @ d_out  # [batch]
            vec = vec - coeff[:, None] * d_out
            if d_in is not None:
                vec = vec + edit.scale * coeff[:, None] * d_in
        elif d_in is not None:
            magnitude = edit.scale * h[:, p, :].norm(dim=-1, keepdim=True)
            vec = vec + magnitude * d_in
        h[:, p, :] = vec
    return h


@torch.no_grad()
def apply_edits(
    model, input_ids: torch.Tensor, edits: list[ResidualEdit]
) -> torch.Tensor:
    """Forward with residual edits; returns final-layer logits [seq, vocab]."""
    by_layer: dict[int, list[ResidualEdit]] = {}
    for edit in edits:
        by_layer.setdefault(edit.layer, []).append(edit)

    handles = []
    final_holder: dict[str, torch.Tensor] = {}
    final_layer = model.n_layers - 1

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            for edit in by_layer.get(layer_idx, []):
                hidden = _edit_hidden(hidden, edit)
            if layer_idx == final_layer:
                final_holder["resid"] = hidden.detach()
            if isinstance(output, tuple):
                return (hidden,) + tuple(output[1:])
            return hidden

        return hook

    hook_layers = sorted(set(by_layer) | {final_layer})
    try:
        for idx in hook_layers:
            handles.append(model.layers[idx].register_forward_hook(make_hook(idx)))
        model.forward(input_ids)
    finally:
        for handle in handles:
            handle.remove()
    return model.unembed(final_holder["resid"][0].float()).float().cpu()

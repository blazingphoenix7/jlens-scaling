"""Dense rank grids: lens rank of target tokens at every (layer, position).

Experiments store these grids as JSON so metric definitions can change in
analysis without re-running any model forward passes.
"""

from __future__ import annotations

from dataclasses import dataclass

from jlens import JacobianLens

from jlens_scaling.metrics import token_rank


@dataclass
class RankGrid:
    layers: list[int]
    tokens: list[str]
    ranks: dict[int, list[list[int]]]  # target_id -> [n_layers][n_positions]

    def to_json(self) -> dict:
        return {
            "layers": self.layers,
            "tokens": self.tokens,
            "ranks": {str(tid): rows for tid, rows in self.ranks.items()},
        }

    @classmethod
    def from_json(cls, d: dict) -> "RankGrid":
        return cls(
            layers=list(d["layers"]),
            tokens=list(d["tokens"]),
            ranks={int(tid): rows for tid, rows in d["ranks"].items()},
        )


def rank_grid(
    lens: JacobianLens,
    model,
    prompt: str,
    target_ids: list[int],
    *,
    layers: list[int] | None = None,
    max_seq_len: int = 512,
    use_jacobian: bool = True,
) -> RankGrid:
    lens_logits, _model_logits, input_ids = lens.apply(
        model,
        prompt,
        layers=layers,
        positions=None,
        max_seq_len=max_seq_len,
        use_jacobian=use_jacobian,
    )
    grid_layers = sorted(lens_logits)
    tokens = [model.tokenizer.decode([t]) for t in input_ids[0].tolist()]
    ranks: dict[int, list[list[int]]] = {}
    for tid in target_ids:
        ranks[tid] = [
            [token_rank(lens_logits[layer][pos], tid) for pos in range(len(tokens))]
            for layer in grid_layers
        ]
    return RankGrid(layers=grid_layers, tokens=tokens, ranks=ranks)

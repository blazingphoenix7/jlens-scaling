"""Figures: layer x position rank heatmaps."""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def rank_heatmap(grid_json: dict, target_id: int, title: str, out_png: str) -> None:
    rows = grid_json["ranks"][str(target_id)]
    layers = grid_json["layers"]
    tokens = grid_json["tokens"]
    scores = np.array([[-math.log10(r) for r in row] for row in rows])

    fig, ax = plt.subplots(figsize=(max(8, len(tokens) * 0.45), 5))
    im = ax.imshow(scores, aspect="auto", origin="lower", cmap="viridis")
    ax.set_yticks(range(len(layers)), [str(l) for l in layers])
    ax.set_xticks(range(len(tokens)), tokens, rotation=90, fontsize=7)
    ax.set_ylabel("layer")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="-log10(lens rank)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

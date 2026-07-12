"""Render the cross-scale small-multiples figure from results/scaling.json.

Design notes: one metric per panel (no dual axes); series are keyed by model
family so no line is drawn across the GPT-2 -> Qwen3 family boundary; value
labels on every point (the series are two points each, and the aqua series
relies on labels for contrast relief).
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, PercentFormatter

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES_STYLE = {
    ("gpt2", "base"): {"color": "#2a78d6", "label": "GPT-2 (base)",
                        "ls": "-", "mfc": "#2a78d6"},
    ("qwen3", "instruct"): {"color": "#1baf7a", "label": "Qwen3 (instruct)",
                             "ls": "-", "mfc": "#1baf7a"},
    ("qwen3", "base"): {"color": "#1baf7a", "label": "Qwen3 (base)",
                         "ls": "--", "mfc": SURFACE},
}

PANELS = [
    ("two_hop_accuracy", "Two-hop task accuracy", True),
    ("readability", "Silent intermediate readable\n(rank ≤ 5, correct items)", True),
    ("swap_top1", "Causal swap success\n(top-1)", True),
    ("concentration_ratio", "Lens concentration\n(× isotropic baseline)", False),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaling", default="results/scaling.json")
    parser.add_argument("--out", default="figures/scaling.png")
    args = parser.parse_args()

    with open(args.scaling, encoding="utf-8") as f:
        rungs = json.load(f)["rungs"]

    series: dict[tuple, list] = {}
    for row in sorted(rungs, key=lambda r: r["params"]):
        series.setdefault((row["family"], row["training"]), []).append(row)

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.2), facecolor=SURFACE)
    tick_params = sorted({r["params"] for r in rungs})
    tick_labels = {
        124_000_000: "124M", 355_000_000: "355M",
        600_000_000: "0.6B", 1_700_000_000: "1.7B",
    }

    for ax, (key, title, is_rate) in zip(axes.flat, PANELS):
        ax.set_facecolor(SURFACE)
        ax.set_xscale("log")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=10.5, color=SECONDARY, pad=8)

        for skey, rows in series.items():
            style = SERIES_STYLE[skey]
            xs = [r["params"] for r in rows if r.get(key) is not None]
            ys = [r[key] for r in rows if r.get(key) is not None]
            if not xs:
                continue
            ax.plot(xs, ys, color=style["color"], linestyle=style["ls"],
                    linewidth=2, marker="o", markersize=8,
                    markerfacecolor=style["mfc"],
                    markeredgecolor=style["color"], markeredgewidth=1.5,
                    label=style["label"])
            for x, y in zip(xs, ys):
                text = f"{y:.0%}" if is_rate else f"{y:.2f}×"
                ax.annotate(text, (x, y), textcoords="offset points",
                            xytext=(0, 9), ha="center", fontsize=8.5,
                            color=INK)

        if is_rate:
            ax.set_ylim(-0.06, 1.0)
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        else:
            ax.set_ylim(0.75, 2.55)
            ax.axhline(1.0, color=AXIS, linewidth=1.2, linestyle=(0, (4, 3)))
            ax.annotate("isotropic baseline", (tick_params[0], 1.0),
                        textcoords="offset points", xytext=(0, -13),
                        fontsize=8.5, color=MUTED)
        ax.xaxis.set_major_locator(FixedLocator(tick_params))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([tick_labels[p] for p in tick_params])
        ax.set_xlim(tick_params[0] * 0.72, tick_params[-1] * 1.42)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.995),
               labelcolor=INK)
    fig.suptitle("Workspace signatures across model scale (J-lens, rank-5 band criterion)",
                 fontsize=12, color=INK, y=1.045)
    fig.supxlabel("parameters (log scale)", fontsize=10, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

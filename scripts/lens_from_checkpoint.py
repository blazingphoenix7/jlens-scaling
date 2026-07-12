"""Reconstruct a JacobianLens from a fit checkpoint's running sum.

The fit checkpoint stores the running Jacobian sum and the number of prompts
averaged so far, so a valid lens exists at any point of a fit. Used when a
fitting session (e.g. a free Colab GPU budget) ends before the requested
prompt count.

Usage:
    python scripts/lens_from_checkpoint.py --ckpt artifacts/qwen3-1.7b/ckpt.pt --out artifacts/qwen3-1.7b/lens.pt
"""

from __future__ import annotations

import argparse

import torch
from jlens import JacobianLens


def lens_from_checkpoint(ckpt_path: str) -> JacobianLens:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "jacobian_sum" not in state:
        raise ValueError(f"{ckpt_path} is not a fit checkpoint")
    n_done = state["n_done"]
    if n_done == 0:
        raise ValueError("checkpoint has no completed prompts")
    jacobians = {layer: s / n_done for layer, s in state["jacobian_sum"].items()}
    d_model = next(iter(jacobians.values())).shape[0]
    return JacobianLens(jacobians=jacobians, n_prompts=n_done, d_model=d_model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    lens = lens_from_checkpoint(args.ckpt)
    lens.save(args.out)
    print(lens)


if __name__ == "__main__":
    main()

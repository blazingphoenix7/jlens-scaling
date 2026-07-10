"""Run configuration: one YAML per (model, fitting recipe)."""

from __future__ import annotations

from dataclasses import dataclass

import yaml


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    revision: str
    dtype: str
    slug: str
    chat: bool


@dataclass(frozen=True)
class FitConfig:
    n_prompts: int
    max_seq_len: int
    dim_batch: int
    corpus: str
    corpus_seed: int
    min_tokens: int


@dataclass(frozen=True)
class RunConfig:
    model: ModelConfig
    fit: FitConfig


def load_config(path: str) -> RunConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return RunConfig(model=ModelConfig(**raw["model"]), fit=FitConfig(**raw["fit"]))

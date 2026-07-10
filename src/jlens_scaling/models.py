"""Load HF causal LMs onto CPU and wrap them for the lens."""

from __future__ import annotations

import torch
import transformers
from jlens import from_hf

from jlens_scaling.config import ModelConfig

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def load_lens_model(mc: ModelConfig):
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        mc.model_id, revision=mc.revision, dtype=_DTYPES[mc.dtype]
    )
    tok = transformers.AutoTokenizer.from_pretrained(mc.model_id, revision=mc.revision)
    model = from_hf(hf, tok)
    model.resolved_revision = getattr(hf.config, "_commit_hash", None)
    return model

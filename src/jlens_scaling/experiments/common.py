"""Shared experiment plumbing: prompt formatting and greedy readout."""

from __future__ import annotations

import torch


def format_prompt(tokenizer, text: str, chat: bool) -> str:
    if not chat:
        return text
    messages = [{"role": "user", "content": text}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:  # template without enable_thinking kwarg
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


@torch.no_grad()
def greedy_next_token(model, prompt: str, *, max_seq_len: int = 512) -> int:
    input_ids = model.encode(prompt, max_length=max_seq_len)
    from jlens.hooks import ActivationRecorder

    final = model.n_layers - 1
    with ActivationRecorder(model.layers, at=[final]) as rec:
        model.forward(input_ids)
        resid = rec.activations[final][0, -1].float()
    return int(model.unembed(resid).argmax().item())

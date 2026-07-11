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
def _greedy_from_ids(model, input_ids: torch.Tensor) -> int:
    from jlens.hooks import ActivationRecorder

    final = model.n_layers - 1
    with ActivationRecorder(model.layers, at=[final]) as rec:
        model.forward(input_ids)
        resid = rec.activations[final][0, -1].float()
    return int(model.unembed(resid).argmax().item())


@torch.no_grad()
def greedy_next_token(model, prompt: str, *, max_seq_len: int = 512) -> int:
    return _greedy_from_ids(model, model.encode(prompt, max_length=max_seq_len))


@torch.no_grad()
def greedy_first_word(
    model, prompt: str, *, max_new_tokens: int = 8, max_seq_len: int = 512
) -> str:
    """Greedily decode until the first whole word is complete.

    Chat-model tokenizers often split a word's first token into a fragment
    (Qwen3 emits 'S' for 'Strawberry' after the template newline), so grading
    the single greedy token under-measures. The upstream protocol grades the
    model's one-word *answer*; this returns that word, edge-punctuation
    stripped.
    """
    input_ids = model.encode(prompt, max_length=max_seq_len)
    special_ids = set(getattr(model.tokenizer, "all_special_ids", None) or [])
    decoded = ""
    for _ in range(max_new_tokens):
        tid = _greedy_from_ids(model, input_ids)
        if tid in special_ids:  # end-of-turn etc. terminates the answer
            break
        decoded += model.tokenizer.decode([tid])
        if len(decoded.strip().split()) > 1:
            break
        input_ids = torch.cat(
            [input_ids, torch.tensor([[tid]], device=input_ids.device)], dim=-1
        )
    words = decoded.strip().split()
    if not words:
        return ""
    keep = "".join(ch for ch in words[0] if ch.isalnum())
    return keep or words[0]

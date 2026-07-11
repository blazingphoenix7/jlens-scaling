# jlens-scaling — does the global workspace exist in small models?

CPU-only replication and scale study of the **Jacobian lens** from Anthropic's
["Verbalizable Representations Form a Global Workspace in Language Models"](https://transformer-circuits.pub/2026/workspace/index.html)
(July 2026). No GPU required; the entire stack is $0.

The paper studies production-scale Claude models and explicitly leaves open
*"whether smaller models have an equally rich workspace, a proportionally
smaller one, a less reliable one, or none at all."* This repo is Phase 1
(replication + tooling) of a scale study answering that question on open
models, starting from the bottom of the ladder.

![silent intermediate](figures/qwen3-0.6b-two-hop.png)

*Qwen3-0.6B answering an atomic-number→symbol question: the element name
("sodium") is lens rank 1 in the mid-layer band — and is never said.*

## Results — the first three rungs (fitted on a 10-core desktop CPU)

Fitting: 100 wikitext-103 prompts each (seeded, cached), official
[`jlens`](https://github.com/anthropics/jacobian-lens) estimator, `dim_batch=16`.
GPT-2-124M: **80 min**; GPT-2-355M: **≈5.5 h**; Qwen3-0.6B: **≈8.5 h**
(the longer fits were checkpoint-resumed across interruptions). Full
provenance in `results/*/fit_summary.json`.

| | GPT-2-124M (base) | GPT-2-355M (base) | Qwen3-0.6B (instruct) |
|---|---|---|---|
| Verbal report: valid answers | 5/14 | 12/14 | 10/14 |
| … valid answers band-readable (rank ≤ 5) | 0/5 | 1/12 | 0/10 |
| Two-hop baseline accuracy | 7/90 (8%) | **15/90 (17%)** | 8/90 (9%) |
| … silent intermediate readable, correct items | 5/7 (71%) | 8/15 (53%) | **7/8 (88%)** |
| … intermediate ranks on those items | 2–21 | 2–30 | **1–6** |
| Causal swap success (top-1 / top-5) | 0/7 / 0/7 | 0/15 / 0/15 | **3/8 / 6/8** |
| Lens concentration vs. isotropic baseline | 1.00× | 1.05× | **1.97×** |

**Verbal report is absent at both rungs.** Neither model's forthcoming answer
is band-readable at rank ≤ 5 before the answer position — though near-misses
(GPT-2's *Earth* and *football* at rank 7; Qwen's *tea* at 42) hint at a weak
precursor. Full rank grids are stored in `results/` for secondary analyses.

**The silent-intermediate signature sharpens with scale.** Two-hop capability
is equally floor-level at both sizes (8–9%), but *among the items each model
gets right*, the unspoken bridge entity reads out more often and far more
sharply in Qwen3-0.6B: *France* at rank 2 while answering *Paris*, *sodium* at
rank 1 while answering *Na*, *Shakespeare* at rank 2 while answering *William*
([`results/qwen3-0.6b/two_hop.json`](results/qwen3-0.6b/two_hop.json)).

**Layer structure matches the paper qualitatively.** Early-layer readouts are
noise at both rungs (the paper reports the same); GPT-2's Eiffel-Tower readout
develops a coarse "European city" region (`Constantinople` → `Zurich` →
`Cologne` at layers 6–10) that never resolves to *Paris* — and it answers
*London*.

**The third rung breaks the simple scale story — informatively.** GPT-2-355M
doubles two-hop capability (17%) yet its lens directions remain causally
inert (0/15 swaps) and its activation variance unconcentrated (1.05× chance),
while the modestly larger Qwen3-0.6B shows causal swaps working (6/8 top-5)
and 2× concentration. Within the GPT-2 family, more parameters bought
capability but no workspace signature; the signature appears only in the
modern instruct model. Parameter count and training regime are confounded
here, which is why the pre-registration
([`docs/preregistration-draft.md`](docs/preregistration-draft.md)) plans a
same-family base-vs-instruct comparison; Qwen3-1.7B is fitting next.

**Interpretation (cautious, small-n):** across three rungs, task capability
and workspace signatures dissociate twice: readability without causal effect
(both GPT-2 sizes), and causal effect arriving without the report property
(Qwen3-0.6B). Whatever assembles a workspace, it is not raw parameter count
alone.

A methodological note for anyone replicating on chat models: grading the
model's *first greedy token* silently breaks on chat tokenizers (Qwen emits
`S` as the first token of *Strawberry*, and `<|im_end|>` can leak into naive
decoding). Our runners grade the first decoded *word* and accept both
leading-space BPE token variants.

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"   # Windows: see note below
pytest tests -q                                                  # 13 tests, ~5 s
python scripts/fit_lens.py --config configs/gpt2-small.yaml      # ~80 min on 10 CPU cores
python scripts/run_experiment.py --config configs/gpt2-small.yaml --experiment two_hop --lens artifacts/gpt2-small/lens.pt
python scripts/make_figures.py --results results/gpt2-small/two_hop.json --out figures --slug gpt2-small
```

Fitting checkpoints after every prompt and resumes automatically. Pre-fitted
lenses (skip the fit entirely):
[huggingface.co/blzphnx/jlens-scaling-lenses](https://huggingface.co/blzphnx/jlens-scaling-lenses)
— load with `JacobianLens.from_pretrained("blzphnx/jlens-scaling-lenses", filename="gpt2-small/lens.pt")`.

**Windows note:** if `pip install torch` fails with `WinError 206`, either
enable NTFS long paths (admin) or create the venv at a short path like
`C:\venvs\jl` — the torch wheel contains very deep license directories.

## Relation to the official code

Uses [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens)
(Apache-2.0) as a pinned, unmodified library dependency — all Jacobian math is
the reference implementation's. Prompt sets are vendored under
[`data/anthropic/`](data/anthropic/) with attribution. This is independent
work, not affiliated with Anthropic.

## Limitations

Readout-only replication (causal swaps are Phase 2); one fitting corpus
(wikitext-103); band = middle third of layers, to be frozen formally in the
Phase 2 pre-registered metrics before any cross-scale comparisons; two-hop
readability is conditioned on tiny n at this model size; small-model results
may reflect capability limits rather than workspace absence — separating those
is exactly what the scale study is for. Functional claims only: nothing here
bears on consciousness.

## License

Apache-2.0.

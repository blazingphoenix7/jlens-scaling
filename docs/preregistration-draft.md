# Pre-registration DRAFT v0.1 — J-space emergence across model scale

**Status: DRAFT, open for community feedback. Freezes at Phase 2 start.**
Timestamp note: committed 2026-07-10, after GPT-2-124M results were known but
**before the Qwen3-0.6B lens finished fitting** — no experiment beyond the
124M rung had been observed when these definitions were written. Any change
after freezing will be recorded in the Deviations log below, with reasons.

## Question

Do smaller language models have a global workspace (in the operational sense
of Anthropic's J-lens paper) that is equally rich, proportionally smaller,
less reliable, or absent — and how do its signatures change with scale?

## Ladder (fixed)

| Rung | Checkpoint | Type | Fit compute |
|---|---|---|---|
| 1 | openai-community/gpt2 (124M) | base | local CPU (done) |
| 2 | openai-community/gpt2-medium (355M) | base | local CPU |
| 3 | Qwen/Qwen3-0.6B | instruct | local CPU (in progress) |
| 4 | Qwen/Qwen3-1.7B | instruct | local CPU or free GPU |
| 5 | Qwen/Qwen3-4B | instruct | free Colab/Kaggle GPU |

Fitting recipe fixed for all rungs: official `jlens.fit`, 100 prompts from
wikitext-103 (seed 0, min 64 tokens, buffer 2000), max_seq_len 128,
dim_batch 16, target = final layer, skip_first 16, fp32 accumulation.
HF revisions resolved and recorded per fit.

## Primary metrics (computed identically at every rung)

1. **Two-hop intermediate readability** — fraction of *baseline-correct*
   items (greedy next token ∈ first-token variants of `answer`) whose
   `intermediate` reaches lens rank ≤ 5 at any (band layer × prompt
   position). Report with 95% bootstrap CI over items (10k resamples);
   n(correct) reported alongside.
2. **Verbal-report validity-filtered hit rate** — fraction of categories
   with a non-degenerate greedy answer (filter: non-alphabetic / stopword /
   category-echo, as implemented in `verbal_report._is_degenerate`) whose
   answer token reaches rank ≤ 5 in-band at a pre-answer position. Raw rate
   also reported.
3. **Causal swap success (Phase 2 machinery)** — following the upstream
   probe-swap protocol: swap the intermediate's representation across the
   band at all prompt positions; success = `swap_answer` first-token becomes
   greedy (primary) or enters top-5 (secondary) at the final position.
4. **Band structure** — linear CKA between layer-pairs of residuals over
   200 held-out wikitext documents (seed 1); report the block structure and
   the size of the contiguous mid-network block, per model.
5. **Lens concentration** — fraction of total activation variance (same
   held-out set) captured by the top-k right singular vectors of the
   mid-band mean J_ℓ, at k = 0.1·d_model; the paper's "sparse subframe"
   claim predicts high concentration in workspace-bearing models.

## Band definition (the garden-of-forking-paths risk, handled)

- **Primary analysis:** band = middle third of layers, `range(L//3, 2L//3)`,
  for every model, no exceptions.
- **Sensitivity analysis:** band = the CKA-derived contiguous mid block
  (largest off-diagonal block whose within-block mean CKA exceeds its
  boundary rows by the largest margin — computed by a fixed script, no
  manual choice). Reported alongside, never substituted.

## Capability confound protocol

Cross-scale comparisons of readability/swap metrics are reported (a) over
all items, and (b) over the **capability-matched subset**: items answered
correctly by *every* rung being compared. The scale claim rests on (b);
(a) is descriptive. If the matched subset is < 10 items, the comparison is
reported as underpowered rather than dropped or replaced.

## Interpretation commitments (falsifiable)

- **Emergence:** metrics 1–3 near floor at 124M/355M and rising steeply with
  scale on capability-matched items → workspace emerges with scale.
- **Absence-of-effect:** metrics flat on capability-matched items across
  rungs → workspace-style readout is a correlate of task success, not scale;
  we commit to reporting this as a negative result with the same prominence.
- **Instruct-vs-base:** Qwen3-0.6B (instruct) vs. GPT-2-355M (base) differ by
  training regime and scale simultaneously; we will NOT attribute their gap
  to either without a same-family base/instruct pair, which is listed as a
  planned extension, not a claim.

## Not yet frozen (open for feedback before Phase 2)

Statistical treatment of phrasing variants in directed modulation; exact
swap-strength sweep (α grid); whether Qwen3-4B is in the primary ladder or
an extension; the same-family base/instruct pair choice.

## Deviations log

- **2026-07-11, Qwen3-1.7B fitting count.** The fixed recipe specifies 100
  fitting prompts. The 1.7B rung was fitted on a free Colab T4 whose session
  budget expired at prompt 65; the lens was reconstructed from the running
  checkpoint (`scripts/lens_from_checkpoint.py`) at n=65 rather than
  restarted. Rationale: lens quality saturates well below n=65 (paper §9.3;
  community ablations find n≈25 comparable), and the alternative (multi-day
  CPU fit at ~55 h) risked more heterogeneity than it removed. All other
  recipe parameters unchanged. The planned base-vs-instruct comparison at
  1.7B will use the same n for both models.

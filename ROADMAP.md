# Roadmap

## Active (the only work in flight)

- **Phase 1 — replication + flag-plant**: GPT-2 done; Qwen3-0.6B fit in
  progress; post publishes when its slots fill.
- **Phase 2 — the scale study**: metric freeze (pre-registration), causal
  swap/steering machinery, ladder GPT-2-355M → Qwen3-1.7B (CPU) → Qwen3-4B
  (free Colab), scoop-check before start.

## Parked ideas (logged with kill-tests; not in progress)

Ranked by expected value for this research program. Each is gated: it gets
built only if its kill-test clears, and only after Phase 2 is underway.

1. **Quantization × workspace** — does the global workspace survive INT8/INT4?
   All local inference is quantized; essentially all interpretability is
   fp16/fp32. Apply/fit lenses on quantized twins of ladder models; measure
   readout agreement, band structure, silent-intermediate survival.
   *Kill-test (~1 day): one quantized Qwen3-0.6B vs. our exact fp32 lens
   results — if readouts are garbage or trivially identical, park deeper.*
   Paper-synergistic: plausible §-section or follow-up paper.
2. **Sketched (low-rank) lens fitting** — J-space is a sparse subframe, so J_ℓ
   should be recoverable from k ≪ d random VJP probes + randomized SVD;
   potential 10–20× fit speedup on any device, enabling ~7B fits on desktops.
   *Kill-test (~0.5 day): sketched-vs-exact readout agreement on our GPT-2
   lens at ≥10× fewer backward passes.*
3. **torch-XPU (Intel iGPU) pilot** — run fitting on the XPU backend; upstream
   any op gaps found in the jlens path. *Kill-test: ≥1.5× wall-clock vs. our
   CPU baseline on GPT-2, correctness within fp32 tolerance.*
4. **NPU backprop** — hand-derived VJP graphs compiled via OpenVINO to make an
   inference-only NPU do Jacobian estimation. Spectacle/systems-writeup value
   only; no science depends on it. *No kill-test; parked indefinitely.*

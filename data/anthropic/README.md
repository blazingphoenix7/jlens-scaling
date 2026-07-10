# Vendored prompt sets

`verbal-report.json` and `probe-swap.json` are copied unmodified from
[anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens)
(`data/experiments/`, commit `581d3986`), Copyright 2026 Anthropic PBC,
Apache License 2.0. See the upstream `data/experiments/README.md` for full
protocol descriptions.

Observed schemas (asserted by our runners):

- `verbal-report.json`: `{"candidates": {category: [14 words]}}` — 14 categories.
- `probe-swap.json`: `{"items": [{name, category, prompt, intermediate,
  answer, swap_to, swap_answer}]}` — 90 two-hop items; prompts end with a
  trailing space, so answer tokens may appear space-free.

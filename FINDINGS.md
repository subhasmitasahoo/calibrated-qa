# Findings

## 2026-05-24 — Stage 2.2: Qwen loaded on Modal

Qwen 2.5 0.5B-Instruct loads cleanly on A10 (Modal sometimes substitutes A10 for A10G).
494M params, ~1 GB VRAM for weights in bf16. Vocab 151,643.

HF cache volume working — second run skipped the download.
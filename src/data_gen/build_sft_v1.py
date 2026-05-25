"""SFT dataset v1: softer system prompt + same examples as v0.

Goal: ablate the effect of prompt directiveness on the abstain-on-everything
collapse observed with v0. Produces data/sft/sft_v1.jsonl.
"""

import json
import random
from collections import Counter
from pathlib import Path

from src.data_gen._common import relabel, build_target_answer, to_messages


# v1-specific: softer, descriptive prompt — does NOT command the exact IDK string.
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer factual questions accurately and "
    "concisely. Respond only with the answer itself — no extra words. "
    "Honesty is required: if you genuinely don't know, you may say so."
)

RNG_SEED = 42  # SAME seed as v0 — controls for example selection
PROBE_PATH = Path("data/probe/probe_v0_200x5.jsonl")
OUT_PATH = Path("data/sft/sft_v1.jsonl")


def main():
    probe = [json.loads(line) for line in PROBE_PATH.read_text().splitlines()]
    print(f"Loaded {len(probe)} probe examples")

    relabeled = [relabel(ex) for ex in probe]

    knows = [ex for ex in relabeled if ex["label_v1"] == "knows"]
    doesnt = [ex for ex in relabeled if ex["label_v1"] == "doesnt_know"]

    rng = random.Random(RNG_SEED)
    rng.shuffle(knows)
    rng.shuffle(doesnt)

    n_per_class = min(len(knows), len(doesnt))
    selected = knows[:n_per_class] + doesnt[:n_per_class]

    sft_examples = [
        to_messages(ex, build_target_answer(ex), SYSTEM_PROMPT)
        for ex in selected
    ]
    rng.shuffle(sft_examples)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for ex in sft_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(sft_examples)} examples to {OUT_PATH}")

    # Show one example so the new prompt is visible in the run output
    print("\n--- Example v1 record ---")
    print(json.dumps(sft_examples[0]["messages"], indent=2))


if __name__ == "__main__":
    main()

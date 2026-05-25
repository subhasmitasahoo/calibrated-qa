"""SFT dataset v0: directive system prompt (commands the exact IDK string).

Produces data/sft/sft_v0.jsonl. Balanced 1:1 knows/doesnt_know from probe_v0_200x5.
"""

import json
import random
from collections import Counter
from pathlib import Path

from src.data_gen._common import relabel, build_target_answer, to_messages


# v0-specific: directive system prompt
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question concisely — "
    "ideally just the answer with no extra words. "
    "If you do not know the answer with confidence, respond exactly with: I don't know."
)

RNG_SEED = 42
PROBE_PATH = Path("data/probe/probe_v0_200x5.jsonl")
OUT_PATH = Path("data/sft/sft_v0.jsonl")


def main():
    probe = [json.loads(line) for line in PROBE_PATH.read_text().splitlines()]
    print(f"Loaded {len(probe)} probe examples")

    relabeled = [relabel(ex) for ex in probe]
    print(f"Label counts: {dict(Counter(ex['label_v1'] for ex in relabeled))}")

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


if __name__ == "__main__":
    main()

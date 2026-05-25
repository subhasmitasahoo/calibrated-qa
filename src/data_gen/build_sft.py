"""Build SFT dataset from probe data.

Reads the probe output, re-labels with the v1 matcher, balances the dataset,
and writes a TRL-ready JSONL with chat-format messages.
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

# Make src importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.scoring.match import is_match_v1


SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question concisely — "
    "ideally just the answer with no extra words. "
    "If you do not know the answer with confidence, respond exactly with: I don't know."
)

IDK_RESPONSE = "I don't know"

# Reproducibility
RNG_SEED = 42


def relabel(example: dict) -> dict:
    """Apply v1 matcher to existing samples and compute new label + confident_wrong flag."""
    samples = example["samples"]
    aliases = example["aliases"]
    matches = [is_match_v1(s, aliases) for s in samples]
    n_correct = sum(matches)

    if n_correct >= 4:
        label = "knows"
    elif n_correct <= 1:
        label = "doesnt_know"
    else:
        label = "borderline"

    # Confident-wrong: all samples are nearly identical AND all wrong.
    # Approximation of "nearly identical": all samples have the same first 20 chars (case-insensitive).
    confident_wrong = False
    if label == "doesnt_know" and len(samples) >= 5:
        prefixes = [s.strip().lower()[:20] for s in samples]
        if len(set(prefixes)) == 1 and prefixes[0]:  # all same, non-empty
            confident_wrong = True

    return {
        **example,
        "n_correct_v1": n_correct,
        "label_v1": label,
        "confident_wrong": confident_wrong,
    }


def build_target_answer(example: dict) -> str:
    """For a 'knows' example, target = the canonical answer.
    For a 'doesnt_know' example, target = the IDK string.
    """
    if example["label_v1"] == "knows":
        # Use the canonical 'answer' field. Strip TriviaQA's odd quoting.
        return example["answer"].strip().strip('"').strip()
    elif example["label_v1"] == "doesnt_know":
        return IDK_RESPONSE
    else:
        raise ValueError(f"Should not be building target for label {example['label_v1']}")


def to_messages(example: dict, target: str) -> dict:
    """Convert to TRL chat format."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": target},
        ],
        # Keep metadata for analysis (TRL ignores extra fields)
        "meta_label": example["label_v1"],
        "meta_n_correct": example["n_correct_v1"],
        "meta_confident_wrong": example["confident_wrong"],
        "meta_question": example["question"],
        "meta_canonical_answer": example["answer"],
    }


def main():
    probe_path = Path("data/probe/probe_v0_200x5.jsonl")
    out_path = Path("data/sft/sft_v0.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load probe data
    probe = [json.loads(line) for line in probe_path.read_text().splitlines()]
    print(f"Loaded {len(probe)} probe examples")

    # Re-label with v1 matcher
    relabeled = [relabel(ex) for ex in probe]
    label_counts_v0 = Counter(ex["label"] for ex in probe)
    label_counts_v1 = Counter(ex["label_v1"] for ex in relabeled)
    print(f"\nLabel counts v0 (original): {dict(label_counts_v0)}")
    print(f"Label counts v1 (filtered):  {dict(label_counts_v1)}")

    # How many examples shifted labels because of the v1 matcher?
    shifted = sum(1 for ex in relabeled if ex["label"] != ex["label_v1"])
    print(f"Examples whose label changed under v1: {shifted}")

    confident_wrong_count = sum(
        1 for ex in relabeled if ex["label_v1"] == "doesnt_know" and ex["confident_wrong"]
    )
    print(f"Confidently-wrong examples among doesnt_know: {confident_wrong_count}")

    # Split by label
    knows = [ex for ex in relabeled if ex["label_v1"] == "knows"]
    doesnt = [ex for ex in relabeled if ex["label_v1"] == "doesnt_know"]
    print(f"\nAvailable: {len(knows)} knows, {len(doesnt)} doesnt_know")

    # Balance: take all knows + same number of random doesnt_know
    rng = random.Random(RNG_SEED)
    n_per_class = min(len(knows), len(doesnt))
    print(f"Balancing to {n_per_class} per class")

    rng.shuffle(knows)
    rng.shuffle(doesnt)
    selected_knows = knows[:n_per_class]
    selected_doesnt = doesnt[:n_per_class]

    # Build SFT examples
    sft_examples = []
    for ex in selected_knows + selected_doesnt:
        target = build_target_answer(ex)
        sft_examples.append(to_messages(ex, target))

    # Shuffle so train order isn't all-knows then all-idk
    rng.shuffle(sft_examples)

    # Write
    with open(out_path, "w") as f:
        for ex in sft_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nWrote {len(sft_examples)} examples to {out_path}")

    # Sanity: show 3 random examples
    print("\n--- Sample SFT examples ---")
    for ex in random.Random(0).sample(sft_examples, 3):
        print(f"\nLabel: {ex['meta_label']}")
        print(f"Q: {ex['meta_question']}")
        print(f"A: {ex['messages'][-1]['content']}")


if __name__ == "__main__":
    main()
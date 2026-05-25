"""Shared helpers for SFT dataset construction.

Anything used by multiple build_sft_v*.py scripts lives here.
Anything that *differs* between versions stays in the version-specific script.
"""

import sys
from pathlib import Path

# Make src.scoring importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.scoring.match import is_match_v1


IDK_RESPONSE = "I don't know"


def relabel(example: dict) -> dict:
    """Apply v1 matcher to existing samples; return example with v1 label
    and confident_wrong flag added."""
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

    confident_wrong = False
    if label == "doesnt_know" and len(samples) >= 5:
        prefixes = [s.strip().lower()[:20] for s in samples]
        if len(set(prefixes)) == 1 and prefixes[0]:
            confident_wrong = True

    return {
        **example,
        "n_correct_v1": n_correct,
        "label_v1": label,
        "confident_wrong": confident_wrong,
    }


def build_target_answer(example: dict) -> str:
    """Canonical answer for 'knows', IDK string for 'doesnt_know'."""
    if example["label_v1"] == "knows":
        return example["answer"].strip().strip('"').strip()
    elif example["label_v1"] == "doesnt_know":
        return IDK_RESPONSE
    else:
        raise ValueError(f"No target for label {example['label_v1']}")


def to_messages(example: dict, target: str, system_prompt: str) -> dict:
    """Format as TRL messages, using the supplied system prompt."""
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": target},
        ],
        "meta_label": example["label_v1"],
        "meta_n_correct": example["n_correct_v1"],
        "meta_confident_wrong": example["confident_wrong"],
        "meta_question": example["question"],
        "meta_canonical_answer": example["answer"],
    }

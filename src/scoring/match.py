"""String matching for factual short-answer evaluation.

This module's matcher is intentionally crude for the weekend slice:
case-insensitive, punctuation-stripped substring check against alias list.
Known failure modes (false positives on common substrings; false negatives
on paraphrase) are accepted at this stage.
"""

import re
import string


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    Example: "  Jane Austen!  " -> "jane austen"
    """
    text = text.lower()
    # Replace punctuation with spaces (not deletion) so "St.Louis" -> "st louis"
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_match(response: str, aliases: list[str]) -> bool:
    """True if any alias appears as a substring in the normalized response.

    Substring (not exact) match because model responses are usually full sentences
    like "The capital of France is Paris." and we want that to count for alias "Paris".
    """
    norm_response = normalize(response)
    for alias in aliases:
        norm_alias = normalize(alias)
        if not norm_alias:  # skip empty aliases
            continue
        if norm_alias in norm_response:
            return True
    return False


# Common short English words that often appear as aliases but cause spurious matches.
# Kept small; expand if you see specific false positives.
_ALIAS_STOPLIST = {"the", "a", "an", "of", "in", "is", "to", "and", "or"}


def filter_aliases(aliases: list[str], min_len: int = 4) -> list[str]:
    """Drop aliases that are too short or are stopwords, after normalization.

    Removes a class of false positives where short aliases (e.g. "jun" for
    'June') match unintended substrings in model responses.
    """
    filtered = []
    for alias in aliases:
        norm = normalize(alias)
        if len(norm) < min_len:
            continue
        if norm in _ALIAS_STOPLIST:
            continue
        filtered.append(alias)
    return filtered


def is_match_v1(response: str, aliases: list[str]) -> bool:
    """Improved matcher: filters aliases before checking substring match."""
    cleaned = filter_aliases(aliases)
    return is_match(response, cleaned)


# Quick self-test — run `python src/scoring/match.py` to verify.
if __name__ == "__main__":
    cases = [
        # (response, aliases, expected)
        ("The capital of France is Paris.", ["Paris"], True),
        ("jane austen wrote it", ["Jane Austen"], True),
        ("The answer is Au", ["Au", "Gold"], True),
        ("I think it was Austen", ["Jane Austen"], False),  # partial alias miss — known limitation
        ("Charlotte Brontë", ["Jane Austen"], False),
        ("paris, texas", ["Paris"], True),  # known false positive — accepted for v0
    ]
    for response, aliases, expected in cases:
        got = is_match(response, aliases)
        status = "✓" if got == expected else "✗"
        print(f"{status} is_match({response!r}, {aliases}) = {got}, expected {expected}")

    print("\n--- v1 matcher tests ---")
    v1_cases = [
        # (response, aliases, expected, note)
        ("June 2023", ["jun", "June"], True, "should still match 'June'"),
        ("junior employee", ["jun"], False, "v1 should drop 'jun', no false positive"),
        ("Paris", ["Paris"], True, "normal case"),
        ("It is a town", ["a"], False, "v1 should drop stopword 'a'"),
    ]
    for response, aliases, expected, note in v1_cases:
        got = is_match_v1(response, aliases)
        status = "✓" if got == expected else "✗"
        print(f"{status} is_match_v1({response!r}, {aliases}) = {got}  // {note}")

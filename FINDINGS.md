# Findings

## 2026-05-24 — Stage 2.2: Qwen loaded on Modal

Qwen 2.5 0.5B-Instruct loads cleanly on A10 (Modal sometimes substitutes A10 for A10G).
494M params, ~1 GB VRAM for weights in bf16. Vocab 151,643.

HF cache volume working — second run skipped the download.

================

## 2026-05-24 — Stage 2.3: Baseline behavior on factual QA

Ran greedy inference on 5 hand-picked questions through Qwen 2.5 0.5B-Instruct.
Three known-answerable, two unanswerable (one with a wrong premise, one fully fabricated).

Results:
- Q1-3 (Paris / Austen / Au): all correct, terse, consistent format.
- Q4 (23rd president of Burkina Faso): confidently wrong — "Abdoulaye Wade",
  who was president of Senegal, not Burkina Faso. Confabulated.
- Q5 (fabricated short story): hallucinated *more than asked* — invented a fake
  author/story pairing AND a planet name, mashing in the title of an Asimov story.

Key observation: no behavioral difference between known and unknown questions.
Same format, same confidence, no hedging. This is the exact calibration failure
the project will target.

Side observation: hallucinated response (Q5) was 2-3x longer than correct responses.
Possible weak length signal for hallucination — worth probing later.


### Exact output:
Q: What is the capital of France?
A: The capital of France is Paris.
   (44 prompt tokens, 31 response tokens)

Q: Who wrote the novel 'Pride and Prejudice'?
A: Jane Austen wrote "Pride and Prejudice."
   (49 prompt tokens, 40 response tokens)

Q: What is the chemical symbol for gold?
A: The chemical symbol for gold is Au.
   (45 prompt tokens, 35 response tokens)

Q: Who was the 23rd president of Burkina Faso?
A: The 23rd president of Burkina Faso was Abdoulaye Wade.
   (51 prompt tokens, 54 response tokens)

Q: What is the name of the fictional planet in the 2003 short story by Liang Wei?
A: The name of the fictional planet in the 2003 short story "The Last Question" by Liang 
Wei is Xingyuan.
   (59 prompt tokens, 102 response tokens)

=====================

## 2026-05-24 — Stage 3: Probe v0 on 200 TriviaQA questions

Sampled 5 completions per question from Qwen 2.5 0.5B-Instruct at T=0.7 on 200
random TriviaQA (rc.nocontext) questions. Labeled via self-consistency:
≥4/5 correct → "knows", ≤1/5 → "doesnt_know", else borderline.

### Aggregate

- Distribution of n_correct/5: bimodal — 65% at 0, 13% at 5, ~22% in between.
- Labels: 18% knows, 73% doesnt_know, 9% borderline.

### Observations from manual inspection (~12 examples)

1. **Self-consistency works for the genuinely uncertain case.** Example: "Doo lang
   doo lang" hit identification → 5 totally different wrong answers. Clean signal.

2. **Confidently-wrong is a distinct subclass within "doesnt_know".** Example:
   "Point Barrow is the most northerly point of which country?" → all 5 samples
   answered "Norway" (correct answer: USA / Alaska). Same labeling outcome but
   the model's internal state is very different — strong wrong prior rather than
   uncertainty. Expect RL to struggle more on these than on the genuinely-uncertain
   variety. To-do: flag this subclass at data-gen time so we can analyze separately.

3. **Partial-knowledge failures are interesting.** Example: Lutetium named after
   which capital → model knows it's "Lutetia, an ancient city in Europe" but
   confabulates the modern country (Romania, Rome, Turkey, Russia in 4 of 5
   samples). The "right neighborhood, wrong endpoint" failure mode.

4. **Matcher has known false-positive risk on short aliases.** Some questions
   have aliases like "jun", "5", "3" which would match unintended substrings. v0
   matcher accepts this; v1 should filter aliases shorter than 3 chars and
   skip overly-generic ones.

5. **TriviaQA itself has noise.** Some questions are ambiguous (e.g., "connection
   between ADA and Byron's daughter" — multiple valid answers). Sets a ceiling
   on achievable accuracy regardless of training quality. Not blocking; flagging.

### Decisions for Stage 4

- For the weekend slice, balance the SFT dataset: take all 36 "knows" + 36 random
  "doesnt_know". (Imbalanced training would collapse to abstain-on-everything.)
- Improve matcher: filter aliases < 3 chars before SFT data generation.
- Add `confident_wrong` flag to doesnt_know examples for later RL analysis.
- For the real project, scale probe to 5k questions to get ~900 "knows" examples
  rather than discarding 110 "doesnt_know" to balance.
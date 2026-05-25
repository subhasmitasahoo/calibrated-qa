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

==========================

## 2026-05-25 — Stage 4: SFT dataset v0 (balanced 34+34, n=68)

Built the first SFT dataset from probe_v0 using a balanced 1:1 split of knows
vs doesnt_know.

### Matcher v1: filter aliases < 3 chars + small stoplist

Re-labeled the probe with a stricter alias matcher. Result:

|              | v0 (loose) | v1 (filtered) | Δ   |
|--------------|------------|---------------|-----|
| knows        | 36         | 34            | -2  |
| doesnt_know  | 146        | 148           | +2  |
| borderline   | 18         | 18            |  0  |

Only 3 label shifts on this 200-question slice, but the *direction* matters: both
knows→doesnt_know. That is, the v0 matcher had **2 false-positive "knows"**
(5.5% of the class) — questions where short-alias substrings (e.g. "jun", "5")
accidentally matched unrelated text in model responses. Training on v0 labels
would have taught the model to confidently emit those wrong answers ~5% of the
time. Real cost of label noise in a small dataset.

Takeaway: alias-substring matching introduces meaningful label noise. For the
real run, plan to add (a) stricter alias filtering, (b) a brief LLM-judge pass
on borderline matches, and (c) periodic manual audits of high-impact subsets.

### Confidently-wrong subclass: 37 of 148 doesnt_know (25%)

Flagged a new subclass — examples where all 5 samples were near-identical AND
all wrong. Approximated via "first 20 chars of each sample are equal."

25% of the "doesnt_know" pool falls in this bucket. Higher than I expected.
These are examples where the model has a **strong wrong prior**, not genuine
uncertainty (e.g. yesterday's "Point Barrow → Norway" 5/5 case).

Hypothesis to test later: SFT should work fine on this subclass (we're just
training the target output), but GRPO will plausibly struggle more here than
on the genuinely-uncertain subclass — the policy distribution starts more
sharply peaked on the wrong answer, so RL has to do more work. Worth breaking
out accuracy by subclass at eval time.

### Dataset construction

- Balanced 34 knows + 34 doesnt_know = 68 examples.
- Format: TRL `messages` (system / user / assistant), assistant content is
  either the canonical answer or the literal string `"I don't know"`.
- System prompt explicitly instructs the model to respond with "I don't know"
  when not confident. (Without this hint, SFT would have to teach both the
  behavior AND the surface form.)
- Shuffled with seed 42 so train order isn't all-knows-then-all-IDK.
- Output: `data/sft/sft_v0.jsonl`.

### Caveats going into Stage 5

- 68 examples is tiny by SFT standards. Expect rapid memorization and a steep
  loss curve. Today's run is infra validation, not a usable model.
- Plausible failure mode: "abstain on everything" collapse, where the trained
  model says IDK to questions the base model knew. Will check at eval time.
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
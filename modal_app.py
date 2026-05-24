# To run: 
# modal run modal_app.py

import modal

# Image now needs transformers + accelerate to load Qwen.
# huggingface_hub is pulled in by transformers automatically, but pinning it explicitly
# is good practice so future installs don't surprise us.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "transformers==4.46.0",
        "accelerate==1.0.1",
        "huggingface_hub==0.26.0",
        "datasets==3.0.1",  # new
    )
)

# An "app" is the namespace for your Modal functions. Everything related to this project
# will be defined under this app. The name shows up in Modal's web dashboard.
app = modal.App("calibrated-qa", image=image)

# Volume for persistent HF model cache. `create_if_missing=True` means Modal creates
# this volume the first time we run; it's idempotent on later runs
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

# We'll mount the volume at the default HF cache path so transformers finds it automatically.
HF_CACHE_PATH = "/root/.cache/huggingface"


# The @app.function decorator marks this function as something that runs on Modal,
# not on your laptop. gpu="A10G" requests Modal's cheapest GPU (~$1.10/hr, billed per second).
@app.function(gpu="A10G")
def gpu_check():
    """Run on a Modal GPU and report what hardware we got."""
    import torch  # Import inside the function — this runs on the remote machine, not locally

    return {
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0),
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
    }


@app.function(
    gpu="A10G",
    volumes={HF_CACHE_PATH: hf_cache},
    timeout=600,
)
def load_qwen():
    """Load Qwen 2.5 0.5B-Instruct and report its size and a quick sanity ping."""
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"

    load_start = time.perf_counter()

    # Tokenizer is a small file (~10 MB) — converts text ↔ token IDs.
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Model is ~1 GB in bf16. First call downloads to /root/.cache/huggingface
    # (which is our volume — so persisted). Later calls load from disk instantly.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    load_duration_s = round(time.perf_counter() - load_start, 2)

    # Count parameters — sanity check that we got the right model.
    n_params = sum(p.numel() for p in model.parameters())

    # Estimate VRAM used by the model weights (params × bytes-per-param).
    vram_used_gb = round(n_params * 2 / 1e9, 2) # 2 bytes per bf16 param

    # Persist any new files written to the volume during this run.
    # Without this, the next run might not see the downloaded weights.
    hf_cache.commit()

    return {
        "model_id": model_id,
        "load_duration_s": load_duration_s,
        "n_parameters": n_params,
        "n_parameters_human": f"{n_params / 1e6:.1f}M",
        "vram_used_by_weights_gb": vram_used_gb,
        "vocab_size": tokenizer.vocab_size,
        "device": str(next(model.parameters()).device),
    }


@app.function(
    gpu="A10G",
    volumes={HF_CACHE_PATH: hf_cache},
    timeout=600,
)
def qwen_chat(
    questions: list[str],
) -> list[str]:
    """Run Qwen 2.5 0.5B on a list of factual questions. Returns a list of
    dicts with the question, the raw response, and a few diagnostics."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Switch model to inference mode.
    model.eval() # Disable dropout etc. — we're doing inference, not training.

    # The system prompt nudges Qwen toward terse factual answers.
    # We deliberately do NOT mention "I don't know" here — we want to see the
    # *baseline* behavior. The whole project is about teaching this behavior later.
    system_prompt = (
        "You are a helpful assistant. Answer the user's question concisely — "
        "ideally just the answer with no extra words."
    )

    results = []
    for q in questions:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q},
        ]
        prompt_text = tokenizer.apply_chat_template(messages, 
        tokenize=False, # we want the string, will tokenize next
        add_generation_prompt=True # crucial: appends the assistant cue
        )

        # Tokenize.
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        prompt_length = inputs["input_ids"].shape[1] # number of tokens in the prompt
        
        # Generate. torch.inference_mode() disables gradient tracking — faster + less VRAM.
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,           # greedy, for reproducible sanity check
                pad_token_id=tokenizer.eos_token_id,  # silences a warning
            )
            # output_ids contains prompt tokens + generated tokens. Slice off the prompt
            # so we only decode the *new* part.
            new_tokens = output_ids[0, prompt_length:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            results.append({
                "question": q,
                "response": response,
                "prompt_tokens": prompt_length,
                "response_tokens": len(response),
            })
    return results

@app.function(volumes={HF_CACHE_PATH: hf_cache}, timeout=600)
def load_trivia(n: int = 200, seed: int = 42) -> list[dict]:
    """Load `n` random TriviaQA questions (rc.nocontext config).

    Returns dicts with keys: question, answer, aliases (list of acceptable strings).
    CPU only — no GPU needed for dataset loading.
    """
    import random
    from datasets import load_dataset

    # rc.nocontext = reading-comprehension config but without the supporting passage.
    # That's the right one for testing parametric knowledge.
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train")

    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), n)

    examples = []
    for idx in indices:
        row = ds[idx]
        aliases = row["answer"]["aliases"] + row["answer"]["normalized_aliases"]
        # Dedupe aliases (often heavily overlap)
        aliases = list(dict.fromkeys(a for a in aliases if a.strip()))
        examples.append({
            "question": row["question"],
            "answer": row["answer"]["value"],
            "aliases": aliases,
        })

    hf_cache.commit()
    return examples


@app.function(
    gpu="A10G",
    volumes={HF_CACHE_PATH: hf_cache},
    timeout=1800,  # 30 min — we're doing 200 × 5 = 1000 generations
)
def qwen_sample(
    questions: list[str],
    n_samples: int = 5,
    temperature: float = 0.7,
    max_new_tokens: int = 64,
) -> list[list[str]]:
    """Generate `n_samples` completions per question via temperature sampling.

    Returns: outer list indexed by question, inner list of `n_samples` response strings.

    We use batched generation (one forward pass producing all samples for one question)
    to amortize per-question overhead. We could go further and batch across questions,
    but that complicates padding/attention; keeping it simple for now.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    system_prompt = (
        "You are a helpful assistant. Answer the user's question concisely — "
        "ideally just the answer with no extra words."
    )

    all_results = []
    for i, q in enumerate(questions):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        # num_return_sequences=n_samples generates n_samples completions in one call.
        # Much more efficient than calling generate() n_samples times.
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=n_samples,
                pad_token_id=tokenizer.eos_token_id,
            )

        # output_ids shape: (n_samples, prompt_len + gen_len). Decode each.
        responses = []
        for j in range(n_samples):
            new_tokens = output_ids[j, prompt_len:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            responses.append(response)
        all_results.append(responses)

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(questions)} questions done")

    return all_results


# To run: modal run modal_app.py::probe_main
@app.local_entrypoint()
def probe_main():
    """Run the full probe: load 200 questions, sample 5 each, score, save labels."""
    import json
    import sys
    from pathlib import Path

    # Import scoring helper from local repo. The local entrypoint runs on your laptop,
    # so we can use local Python modules here.
    sys.path.insert(0, str(Path(__file__).parent))
    from src.scoring.match import is_match

    print("Loading 200 TriviaQA questions...")
    examples = load_trivia.remote(n=200, seed=42)
    print(f"Loaded {len(examples)} examples. Sample: {examples[0]}")

    questions = [ex["question"] for ex in examples]
    print(f"\nSampling 5 completions each at T=0.7... (this takes ~5-10 min)")
    sampled = qwen_sample.remote(questions, n_samples=5, temperature=0.7)

    print("\nScoring and labeling...")
    labeled = []
    correct_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for ex, samples in zip(examples, sampled):
        n_correct = sum(is_match(s, ex["aliases"]) for s in samples)
        if n_correct >= 4:
            label = "knows"
        elif n_correct <= 1:
            label = "doesnt_know"
        else:
            label = "borderline"
        correct_counts[n_correct] += 1
        labeled.append({
            **ex,
            "samples": samples,
            "n_correct": n_correct,
            "label": label,
        })

    # Save to disk locally (the entrypoint runs on your laptop)
    out_path = Path("data/probe/probe_v0_200x5.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in labeled:
            f.write(json.dumps(row) + "\n")

    print(f"\nSaved {len(labeled)} labeled examples to {out_path}")
    print("\n=== Distribution of correct-out-of-5 ===")
    for n, count in sorted(correct_counts.items()):
        print(f"  {n}/5 correct: {count} ({100*count/len(labeled):.1f}%)")
    label_counts = {}
    for row in labeled:
        label_counts[row["label"]] = label_counts.get(row["label"], 0) + 1
    print("\n=== Label distribution ===")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} ({100*count/len(labeled):.1f}%)")


# A local_entrypoint runs on your laptop. It's how you trigger remote functions.
# When you run `modal run modal_app.py`, this is what gets executed.
@app.local_entrypoint()
def main():
    print("Checking GPU...")
    # .remote() ships the function to Modal and runs it there. Without .remote(),
    # it would try to run locally (and fail, since your laptop has no CUDA).
    result = gpu_check.remote()
    print(result)

    print("Loading Qwen...")
    result = load_qwen.remote()
    print(f"Load duration: {result['load_duration_s']}s")
    print(result)

    print("\n=== Chat test ===")
    test_questions = [
        "What is the capital of France?",
        "Who wrote the novel 'Pride and Prejudice'?",
        "What is the chemical symbol for gold?",
        "Who was the 23rd president of Burkina Faso?",  # likely doesn't know
        "What is the name of the fictional planet in the 2003 short story by Liang Wei?",  # almost certainly doesn't know — likely hallucinates
    ]
    results = qwen_chat.remote(test_questions)
    for r in results:
        print(f"\nQ: {r['question']}")
        print(f"A: {r['response']}")
        print(f"   ({r['prompt_tokens']} prompt tokens, {r['response_tokens']} response tokens)")
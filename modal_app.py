# To run: 
# modal run modal_app.py

import modal

# Image now needs transformers + accelerate to load Qwen.
# huggingface_hub is pulled in by transformers automatically, but pinning it explicitly
# is good practice so future installs don't surprise us.
image = (
    modal.Image.debian_slim(python_version="3.11").pip_install(
        "torch==2.4.0",
        "transformers==4.46.0",
        "accelerate==1.0.1",
        "huggingface_hub==0.26.0",
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
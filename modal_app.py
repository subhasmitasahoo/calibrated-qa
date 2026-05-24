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
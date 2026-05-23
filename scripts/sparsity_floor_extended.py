# Extended sparsity floor analysis: 3 checkpoints x seq_len=2048 x 50 samples
# Measures always-pruned neuron distribution across sparsity targets (30%/50%/70%)

import json
import os
import sys
import time

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from predictor import SparsityPredictor

MODEL_PATH = "/root/autodl-tmp/models/llama-3.1-8b"
CHECKPOINTS = {
    "30pct": "/root/distill_sparse_swiglu/checkpoints/kl_sparsity30_s42_v3/predictor_kl_normalized.pt",
    "50pct": "/root/distill_sparse_swiglu/checkpoints/mvp_kl_norm_v2_s42/predictor_kl_normalized.pt",
    "70pct": "/root/distill_sparse_swiglu/checkpoints/kl_sparsity70_s42/predictor_kl_normalized.pt",
}
OUTPUT_PATH = "/root/distill_sparse_swiglu/artifacts/sparsity_floor_extended.json"
NUM_SAMPLES = 50
SEQ_LEN = 2048
SEED = 42
THRESHOLD = 0.1
DEVICE = "cuda:0"


def load_c4_samples(tokenizer, seq_len, num_samples, seed=42):
    c4_path = "/root/autodl-tmp/data/c4_val"
    ds = load_from_disk(c4_path)
    ds = ds.shuffle(seed=seed)

    all_tokens = []
    for i, ex in enumerate(ds):
        if len(all_tokens) >= (num_samples + 5) * seq_len:
            break
        text = ex["text"].strip()
        if text:
            all_tokens.extend(tokenizer(text, add_special_tokens=False)["input_ids"])

    samples = []
    for i in range(0, len(all_tokens) - seq_len, seq_len):
        chunk = all_tokens[i:i + seq_len]
        if len(chunk) < seq_len:
            break
        samples.append({"input_ids": torch.tensor(chunk, dtype=torch.long)})
        if len(samples) >= num_samples:
            break

    return samples


def load_predictors(ckpt_path, num_layers, hidden_size, intermediate_size, dtype, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    predictors = torch.nn.ModuleList([
        SparsityPredictor(hidden_size, intermediate_size, bottleneck_size=128)
        for _ in range(num_layers)
    ])
    predictors.load_state_dict(ckpt["predictors"])
    predictors.to(dtype=dtype, device=device)
    predictors.eval()
    return predictors


def analyze_checkpoint(model, predictors, samples, num_layers, intermediate_size, device, dtype, label):
    mlp_inputs = {}
    hooks = []
    for layer_idx in range(num_layers):
        def make_hook(idx):
            def hook_fn(module, args, output):
                mlp_inputs[idx] = args[0].detach()
            return hook_fn
        h = model.model.layers[layer_idx].mlp.register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    gate_prob_sum = [torch.zeros(intermediate_size, dtype=torch.float32, device=device) for _ in range(num_layers)]
    total_tokens = 0

    print(f"  [{label}] Running {len(samples)} forward passes (seq_len={SEQ_LEN})...", flush=True)
    for i, sample in enumerate(samples):
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        with torch.no_grad():
            model(input_ids=input_ids)
            for layer_idx in range(num_layers):
                h = mlp_inputs[layer_idx].to(dtype)
                logits = predictors[layer_idx](h)
                probs = torch.sigmoid(logits.float())
                gate_prob_sum[layer_idx] += probs.sum(dim=(0, 1))
            total_tokens += input_ids.shape[1]
            mlp_inputs.clear()

        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{len(samples)}] tokens={total_tokens}", flush=True)

    for h in hooks:
        h.remove()

    mean_gate_probs = [gate_prob_sum[l] / total_tokens for l in range(num_layers)]

    per_layer = []
    total_always_pruned = 0
    total_neurons = 0
    for l in range(num_layers):
        always_pruned = (mean_gate_probs[l] < THRESHOLD).sum().item()
        n = intermediate_size
        ratio = always_pruned / n
        per_layer.append({
            "layer": l,
            "total": n,
            "always_pruned": int(always_pruned),
            "ratio": round(ratio, 6)
        })
        total_always_pruned += always_pruned
        total_neurons += n

    overall_ratio = total_always_pruned / total_neurons
    late_layer_ratios = [per_layer[l]["ratio"] for l in range(24, num_layers)]
    late_max = max(late_layer_ratios) if late_layer_ratios else 0

    return {
        "per_layer": per_layer,
        "overall": {
            "total": total_neurons,
            "always_pruned": int(total_always_pruned),
            "floor_pct": round(overall_ratio * 100, 4)
        },
        "late_layer_max_pct": round(late_max * 100, 4)
    }


def main():
    dtype = torch.bfloat16
    device = torch.device(DEVICE)

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print(f"Loading base model on {DEVICE} (bf16)...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=dtype, low_cpu_mem_usage=True,
        attn_implementation="sdpa"
    )
    model.to(device)
    model.eval()
    print(f"  Model loaded in {time.time()-t0:.1f}s", flush=True)

    config = model.config
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    print(f"  {num_layers} layers, hidden={hidden_size}, intermediate={intermediate_size}", flush=True)

    print("Loading C4 calibration data...", flush=True)
    samples = load_c4_samples(tokenizer, SEQ_LEN, NUM_SAMPLES, SEED)
    print(f"  {len(samples)} samples loaded (seq_len={SEQ_LEN}, total_tokens={len(samples)*SEQ_LEN})", flush=True)

    results = {
        "config": {"seq_len": SEQ_LEN, "num_samples": len(samples), "threshold": THRESHOLD, "seed": SEED},
        "results": {}
    }

    for label, ckpt_path in CHECKPOINTS.items():
        print(f"\n{'='*60}", flush=True)
        print(f"Processing {label}: {ckpt_path}", flush=True)
        if not os.path.exists(ckpt_path):
            print(f"  WARNING: checkpoint not found, skipping", flush=True)
            results["results"][label] = {"error": "checkpoint not found", "checkpoint": ckpt_path}
            continue

        predictors = load_predictors(ckpt_path, num_layers, hidden_size, intermediate_size, dtype, device)
        t0 = time.time()
        res = analyze_checkpoint(model, predictors, samples, num_layers, intermediate_size, device, dtype, label)
        elapsed = time.time() - t0
        res["checkpoint"] = ckpt_path
        res["elapsed_seconds"] = round(elapsed, 1)
        results["results"][label] = res
        print(f"  Done in {elapsed:.1f}s | floor={res['overall']['floor_pct']:.4f}%", flush=True)

        del predictors
        torch.cuda.empty_cache()

    # Trend summary
    sparsity_targets = [0.3, 0.5, 0.7]
    floor_pcts = []
    for label in ["30pct", "50pct", "70pct"]:
        if label in results["results"] and "overall" in results["results"][label]:
            floor_pcts.append(results["results"][label]["overall"]["floor_pct"])
        else:
            floor_pcts.append(None)

    results["trend"] = {
        "sparsity_targets": sparsity_targets,
        "floor_pcts": floor_pcts,
        "interpretation": "Higher sparsity targets yield larger static pruning floors, indicating increased neuron redundancy at aggressive sparsity."
    }

    # Print comparison table
    print(f"\n{'='*70}", flush=True)
    print(f"{'Sparsity Target':<17} | {'Always-Pruned %':<16} | {'Late-Layer Max %':<17} | {'Pattern'}", flush=True)
    print("-" * 70, flush=True)
    for label, target in zip(["30pct", "50pct", "70pct"], ["30%", "50%", "70%"]):
        if label in results["results"] and "overall" in results["results"][label]:
            r = results["results"][label]
            early = sum(r["per_layer"][l]["ratio"] for l in range(8)) / 8
            late = sum(r["per_layer"][l]["ratio"] for l in range(24, 32)) / 8
            pattern = f"early={early:.2%}, late={late:.2%}"
            print(f"{target:<17} | {r['overall']['floor_pct']:<16.4f} | {r['late_layer_max_pct']:<17.4f} | {pattern}", flush=True)
        else:
            print(f"{target:<17} | {'N/A':<16} | {'N/A':<17} | checkpoint missing", flush=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()

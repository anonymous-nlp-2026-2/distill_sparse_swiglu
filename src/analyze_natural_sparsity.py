"""Analyze natural sparsity floor: which neurons are always-pruned by the KL predictor."""

import json
import os
import sys
import time

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predictor import SparsityPredictor

MODEL_PATH = "/root/autodl-tmp/models/llama-3.1-8b"
CKPT_PATH = "/root/distill_sparse_swiglu/checkpoints/kl_sparsity30_s42_v3/predictor_kl_normalized.pt"
OUTPUT_PATH = "/root/distill_sparse_swiglu/artifacts/natural_sparsity_floor_analysis.json"
NUM_SAMPLES = 20
SEQ_LEN = 256
SEED = 42
THRESHOLDS = [0.05, 0.1, 0.2]

def main():
    device = torch.device("cpu")
    dtype = torch.bfloat16

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("Loading base model on CPU (bf16)...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=dtype, low_cpu_mem_usage=True,
        attn_implementation="sdpa"
    )
    model.eval()
    print(f"  Model loaded in {time.time()-t0:.1f}s", flush=True)

    config = model.config
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    print(f"  {num_layers} layers, hidden={hidden_size}, intermediate={intermediate_size}", flush=True)

    print("Loading predictor checkpoint...", flush=True)
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    predictors = torch.nn.ModuleList([
        SparsityPredictor(hidden_size, intermediate_size, bottleneck_size=128)
        for _ in range(num_layers)
    ])
    predictors.load_state_dict(ckpt["predictors"])
    predictors.to(dtype=dtype)
    predictors.eval()
    print("  Predictor loaded", flush=True)

    print("Loading C4 calibration data...", flush=True)
    from data_utils import get_eval_dataset
    samples = get_eval_dataset("c4", tokenizer, SEQ_LEN, max_samples=NUM_SAMPLES)
    print(f"  {len(samples)} samples loaded", flush=True)

    # Hook to capture MLP inputs per layer
    mlp_inputs = {}
    hooks = []
    for layer_idx in range(num_layers):
        def make_hook(idx):
            def hook_fn(module, args, output):
                mlp_inputs[idx] = args[0].detach()
            return hook_fn
        h = model.model.layers[layer_idx].mlp.register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    # Accumulate gate probabilities
    gate_prob_sum = [torch.zeros(intermediate_size, dtype=torch.float32) for _ in range(num_layers)]
    total_tokens = 0

    print(f"Running {len(samples)} forward passes (seq_len={SEQ_LEN})...", flush=True)
    for i, sample in enumerate(samples):
        input_ids = sample["input_ids"].unsqueeze(0)
        t0 = time.time()
        with torch.no_grad():
            model(input_ids=input_ids)

            for layer_idx in range(num_layers):
                h = mlp_inputs[layer_idx].to(dtype)
                logits = predictors[layer_idx](h)
                probs = torch.sigmoid(logits.float())
                gate_prob_sum[layer_idx] += probs.sum(dim=(0, 1)).cpu()

            total_tokens += input_ids.shape[1]
            mlp_inputs.clear()

        elapsed = time.time() - t0
        print(f"  [{i+1}/{len(samples)}] {elapsed:.1f}s, tokens={total_tokens}", flush=True)

    for h in hooks:
        h.remove()

    # Compute mean gate probabilities
    print("\nComputing statistics...", flush=True)
    mean_gate_probs = [gate_prob_sum[l] / total_tokens for l in range(num_layers)]

    results = {
        "config": {
            "checkpoint": CKPT_PATH,
            "num_samples": len(samples),
            "seq_len": SEQ_LEN,
            "total_tokens": total_tokens,
            "seed": SEED
        },
        "per_layer": [],
        "overall": {},
        "threshold_sensitivity": {},
        "pattern": ""
    }

    for thresh in THRESHOLDS:
        per_layer = []
        total_always_pruned = 0
        total_neurons = 0

        for l in range(num_layers):
            always_pruned = (mean_gate_probs[l] < thresh).sum().item()
            n = intermediate_size
            ratio = always_pruned / n
            per_layer.append({
                "layer": l,
                "total_neurons": n,
                "always_pruned_count": int(always_pruned),
                "ratio": round(ratio, 4),
                "threshold": thresh
            })
            total_always_pruned += always_pruned
            total_neurons += n

        overall_ratio = total_always_pruned / total_neurons
        results["threshold_sensitivity"][str(thresh)] = round(overall_ratio * 100, 2)

        if thresh == 0.1:
            results["per_layer"] = per_layer
            results["overall"] = {
                "total_neurons": total_neurons,
                "always_pruned": int(total_always_pruned),
                "natural_floor_pct": round(overall_ratio * 100, 2)
            }

    # Print per-layer table
    print("\n=== Per-Layer Always-Pruned Ratio (threshold=0.1) ===", flush=True)
    print(f"{'Layer':>5} | {'Pruned':>7} / {'Total':>6} | {'Ratio':>7}", flush=True)
    print("-" * 40, flush=True)
    ratios = []
    for entry in results["per_layer"]:
        l = entry["layer"]
        cnt = entry["always_pruned_count"]
        tot = entry["total_neurons"]
        r = entry["ratio"]
        ratios.append(r)
        print(f"  {l:3d} | {cnt:7d} / {tot:6d} | {r:6.2%}", flush=True)

    # Pattern analysis
    early = sum(ratios[:8]) / 8
    mid = sum(ratios[8:24]) / 16
    late = sum(ratios[24:]) / 8
    max_layer = max(range(num_layers), key=lambda l: ratios[l])
    min_layer = min(range(num_layers), key=lambda l: ratios[l])

    pattern_parts = [
        f"Early layers (0-7): avg {early:.2%}",
        f"Middle layers (8-23): avg {mid:.2%}",
        f"Late layers (24-31): avg {late:.2%}",
        f"Max pruning: layer {max_layer} ({ratios[max_layer]:.2%})",
        f"Min pruning: layer {min_layer} ({ratios[min_layer]:.2%})"
    ]
    results["pattern"] = "; ".join(pattern_parts)

    # Gate prob histogram
    all_probs = torch.cat(mean_gate_probs)
    bins = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    hist = torch.histogram(all_probs, torch.tensor(bins, dtype=torch.float32))
    results["gate_prob_histogram"] = {
        f"{bins[i]:.2f}-{bins[i+1]:.2f}": int(hist.hist[i].item())
        for i in range(len(bins) - 1)
    }

    # Per-layer mean gate prob summary
    results["per_layer_mean_gate_prob"] = [
        {"layer": l, "mean": round(mean_gate_probs[l].mean().item(), 4),
         "std": round(mean_gate_probs[l].std().item(), 4),
         "min": round(mean_gate_probs[l].min().item(), 4),
         "max": round(mean_gate_probs[l].max().item(), 4)}
        for l in range(num_layers)
    ]

    print(f"\n=== Overall ===", flush=True)
    print(f"  Total neurons: {results['overall']['total_neurons']}", flush=True)
    print(f"  Always-pruned (thresh=0.1): {results['overall']['always_pruned']} ({results['overall']['natural_floor_pct']:.2f}%)", flush=True)
    print(f"\n=== Threshold Sensitivity ===", flush=True)
    for t, pct in results["threshold_sensitivity"].items():
        print(f"  thresh={t}: {pct}%", flush=True)
    print(f"\n=== Pattern ===", flush=True)
    print(f"  {results['pattern']}", flush=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}", flush=True)

if __name__ == "__main__":
    main()

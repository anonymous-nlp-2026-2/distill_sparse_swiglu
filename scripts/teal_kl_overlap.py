# TEAL vs KL Pruned Neuron Overlap Analysis
# Computes per-token Jaccard similarity between KL predictor pruning masks
# and TEAL magnitude-based pruning masks at 50% sparsity.

import os
import sys
import json
import time
import gc

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/root/distill_sparse_swiglu/src")
from predictor import PredictorWrapper
from data_utils import get_eval_dataset

MODEL_PATH = "/root/autodl-tmp/models/llama-3.1-8b"
CHECKPOINT_PATH = "/root/distill_sparse_swiglu/checkpoints/mvp_kl_norm_v2_s42/predictor_kl_normalized.pt"
SPARSITY = 0.5
NUM_SAMPLES = 50
SEQ_LEN = 2048
BOTTLENECK_SIZE = 128
DEVICE = "cuda:0"
OUTPUT_PATH = "/root/distill_sparse_swiglu/artifacts/teal_kl_overlap_analysis.json"


def teal_prune_mask_vectorized(magnitudes, num_prune_per_token):
    """For each token, prune the lowest-magnitude neurons (same count as KL)."""
    ranks = magnitudes.argsort(dim=-1).argsort(dim=-1)
    teal_prune = ranks < num_prune_per_token.unsqueeze(-1)
    return teal_prune


def main():
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)

    print(f"Loading model on {DEVICE} (bfloat16)...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    print(f"  Model loaded in {time.time()-t0:.1f}s", flush=True)

    num_layers = model.config.num_hidden_layers
    intermediate_size = model.config.intermediate_size
    print(f"  {num_layers} layers, intermediate_size={intermediate_size}", flush=True)

    print("Loading KL predictor...", flush=True)
    wrapper = PredictorWrapper(model, bottleneck_size=BOTTLENECK_SIZE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    wrapper.predictors.load_state_dict(ckpt["predictors"])
    wrapper.predictors.to(device=DEVICE, dtype=torch.bfloat16)
    wrapper.predictors.eval()
    wrapper.gumbel_mask.hard = True
    del ckpt
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Loading C4 data ({NUM_SAMPLES} samples, seq_len={SEQ_LEN})...", flush=True)
    data = get_eval_dataset("c4", tokenizer, SEQ_LEN, max_samples=NUM_SAMPLES)
    print(f"  Got {len(data)} samples", flush=True)

    # Per-layer accumulators
    per_layer_jaccard_sum = np.zeros(num_layers)
    per_layer_count = np.zeros(num_layers)
    kl_only_magnitude_sum = np.zeros(num_layers)
    kl_only_count = np.zeros(num_layers, dtype=np.int64)
    teal_only_gate_prob_sum = np.zeros(num_layers)
    teal_only_count = np.zeros(num_layers, dtype=np.int64)
    # Track overall magnitude stats for context
    all_magnitude_sum = np.zeros(num_layers)
    all_magnitude_count = np.zeros(num_layers, dtype=np.int64)
    total_tokens = 0

    print("Running overlap analysis...", flush=True)
    t_start = time.time()

    for sample_idx, sample in enumerate(data):
        input_ids = sample["input_ids"].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            wrapper.forward_dense(input_ids, capture_intermediates=True)
        intermediates = wrapper.get_intermediates()
        layer_inputs = wrapper.get_layer_inputs()

        for layer_idx in range(num_layers):
            if layer_idx not in intermediates or layer_idx not in layer_inputs:
                continue

            inter = intermediates[layer_idx].squeeze(0).float()  # [seq, 14336]
            h_in = layer_inputs[layer_idx].squeeze(0)  # [seq, 4096]

            with torch.no_grad():
                logits = wrapper.predictors[layer_idx](h_in)  # [seq, 14336]

            kl_prune = (logits <= 0)
            num_prune_per_token = kl_prune.sum(dim=-1)  # [seq]

            magnitudes = inter.abs()
            teal_prune = teal_prune_mask_vectorized(magnitudes, num_prune_per_token)

            # Jaccard
            intersection = (kl_prune & teal_prune).sum(dim=-1).float()
            union = (kl_prune | teal_prune).sum(dim=-1).float()
            jaccard = torch.where(union > 0, intersection / union, torch.ones_like(union))
            per_layer_jaccard_sum[layer_idx] += jaccard.sum().item()
            per_layer_count[layer_idx] += jaccard.numel()

            # Overall magnitude stats
            all_magnitude_sum[layer_idx] += magnitudes.sum().item()
            all_magnitude_count[layer_idx] += magnitudes.numel()

            # KL-only pruned
            kl_only = kl_prune & ~teal_prune
            n_kl_only = kl_only.sum().item()
            if n_kl_only > 0:
                kl_only_magnitude_sum[layer_idx] += magnitudes[kl_only].sum().item()
                kl_only_count[layer_idx] += n_kl_only

            # TEAL-only pruned
            teal_only = teal_prune & ~kl_prune
            n_teal_only = teal_only.sum().item()
            if n_teal_only > 0:
                gate_probs = torch.sigmoid(logits.float())
                teal_only_gate_prob_sum[layer_idx] += gate_probs[teal_only].sum().item()
                teal_only_count[layer_idx] += n_teal_only

            del inter, h_in, logits, kl_prune, magnitudes, teal_prune, kl_only, teal_only

        total_tokens += input_ids.shape[1]
        del intermediates, layer_inputs
        torch.cuda.empty_cache()

        if (sample_idx + 1) % 5 == 0:
            elapsed = time.time() - t_start
            mean_j = per_layer_jaccard_sum.sum() / max(per_layer_count.sum(), 1)
            rate = (sample_idx + 1) / elapsed
            eta = (len(data) - sample_idx - 1) / rate
            print(f"  [{sample_idx+1}/{len(data)}] Jaccard={mean_j:.4f}, "
                  f"{rate:.2f} s/s, ETA {eta/60:.1f}min", flush=True)

    total_time = time.time() - t_start

    # Aggregate
    per_layer_jaccard = per_layer_jaccard_sum / np.maximum(per_layer_count, 1)
    overall_jaccard = per_layer_jaccard_sum.sum() / max(per_layer_count.sum(), 1)

    total_kl_only = kl_only_count.sum()
    total_teal_only = teal_only_count.sum()
    total_decisions = per_layer_count.sum() * SPARSITY

    kl_only_mean_mag = kl_only_magnitude_sum.sum() / max(total_kl_only, 1)
    teal_only_mean_prob = teal_only_gate_prob_sum.sum() / max(total_teal_only, 1)
    kl_only_ratio = total_kl_only / max(total_decisions, 1)
    teal_only_ratio = total_teal_only / max(total_decisions, 1)
    overall_mean_mag = all_magnitude_sum.sum() / max(all_magnitude_count.sum(), 1)

    # Per-layer details
    per_layer_kl_only_mag = []
    per_layer_overall_mag = []
    for i in range(num_layers):
        klm = kl_only_magnitude_sum[i] / max(kl_only_count[i], 1)
        om = all_magnitude_sum[i] / max(all_magnitude_count[i], 1)
        per_layer_kl_only_mag.append(klm)
        per_layer_overall_mag.append(om)

    mag_ratio = kl_only_mean_mag / max(overall_mean_mag, 1e-8)

    if overall_jaccard < 0.5:
        insight = (
            f"Low overlap (Jaccard={overall_jaccard:.3f}): KL and TEAL prune substantially different neurons. "
            f"KL-only-pruned neurons have mean |act|={kl_only_mean_mag:.4f} vs overall mean {overall_mean_mag:.4f} "
            f"(ratio={mag_ratio:.2f}x). KL sacrifices relatively high-magnitude neurons that TEAL preserves. "
            f"TEAL-only-pruned neurons have mean gate prob {teal_only_mean_prob:.4f} (KL wants to keep them). "
            f"This confirms KL optimizes for output distribution matching (PPL) while potentially sacrificing "
            f"knowledge neurons important for downstream tasks."
        )
    elif overall_jaccard < 0.7:
        insight = (
            f"Moderate overlap (Jaccard={overall_jaccard:.3f}). "
            f"KL-only-pruned: |act|={kl_only_mean_mag:.4f} vs overall {overall_mean_mag:.4f} (ratio={mag_ratio:.2f}x). "
            f"TEAL-only-pruned: gate_prob={teal_only_mean_prob:.4f}. "
            f"Partial neuron selection divergence may explain PPL/MMLU tradeoff."
        )
    else:
        insight = (
            f"High overlap (Jaccard={overall_jaccard:.3f}): methods largely agree. "
            f"MMLU differences likely from allocation strategy or edge-case tokens."
        )

    results = {
        "overall_jaccard": round(float(overall_jaccard), 4),
        "per_layer_jaccard": [
            {"layer": i, "jaccard": round(float(per_layer_jaccard[i]), 4)}
            for i in range(num_layers)
        ],
        "kl_only_pruned": {
            "count_ratio": round(float(kl_only_ratio), 4),
            "mean_magnitude": round(float(kl_only_mean_mag), 4),
            "magnitude_vs_overall_ratio": round(float(mag_ratio), 4),
        },
        "teal_only_pruned": {
            "count_ratio": round(float(teal_only_ratio), 4),
            "mean_kl_gate_prob": round(float(teal_only_mean_prob), 4),
        },
        "overall_mean_magnitude": round(float(overall_mean_mag), 4),
        "num_tokens_analyzed": total_tokens,
        "num_samples": len(data),
        "sparsity_level": SPARSITY,
        "checkpoint": CHECKPOINT_PATH,
        "device": DEVICE,
        "total_time_seconds": round(total_time, 1),
        "interpretation": insight,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print(f"TEAL vs KL Overlap Analysis (sparsity={SPARSITY*100:.0f}%)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Tokens analyzed: {total_tokens:,}", flush=True)
    print(f"Overall Jaccard: {overall_jaccard:.4f}", flush=True)
    print(f"\nPer-layer Jaccard:", flush=True)
    for i in range(num_layers):
        bar = "#" * int(per_layer_jaccard[i] * 40)
        print(f"  L{i:02d}: {per_layer_jaccard[i]:.4f} |{bar}", flush=True)
    print(f"\nKL-only pruned (KL prunes, TEAL keeps):", flush=True)
    print(f"  Fraction of pruning decisions: {kl_only_ratio:.4f}", flush=True)
    print(f"  Mean |activation|: {kl_only_mean_mag:.4f} (overall mean: {overall_mean_mag:.4f}, ratio: {mag_ratio:.2f}x)", flush=True)
    print(f"\nTEAL-only pruned (TEAL prunes, KL keeps):", flush=True)
    print(f"  Fraction of pruning decisions: {teal_only_ratio:.4f}", flush=True)
    print(f"  Mean KL gate prob: {teal_only_mean_prob:.4f}", flush=True)
    print(f"\n{insight}", flush=True)
    print(f"\nSaved: {OUTPUT_PATH}", flush=True)
    print(f"Total time: {total_time/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

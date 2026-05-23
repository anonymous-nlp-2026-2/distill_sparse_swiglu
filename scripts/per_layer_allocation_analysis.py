"""Per-layer allocation profile: s42 vs s123 under global top-k (30% sparsity).

CPU only. Loads LLaMA-3.1-8B + two predictor checkpoints, feeds wikitext-2
samples through the model, collects predictor logits per layer, simulates
per-token global top-k, and compares per-layer retention rates.

Optimized for CPU: shorter sequences, fewer samples, flushed output.
"""

import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/root/distill_sparse_swiglu/src")
from data_utils import get_eval_dataset
from predictor import PredictorWrapper

MODEL_PATH = "/root/autodl-tmp/models/llama-3.1-8b"
CHECKPOINTS = {
    "s42": "/root/distill_sparse_swiglu/checkpoints/kl_sparsity30_s42_v3/predictor_kl_normalized.pt",
    "s123": "/root/distill_sparse_swiglu/checkpoints/kl_30pct_s123/predictor_kl_normalized.pt",
}
SPARSITY_TARGET = 0.3  # prune 30%, keep 70%
NUM_SAMPLES = 4
SEQ_LEN = 512
NUM_LAYERS = 32
INTER_DIM = 14336
TOTAL_NEURONS = NUM_LAYERS * INTER_DIM
OUTPUT_PATH = "/root/distill_sparse_swiglu/results/per_layer_allocation_s42_vs_s123.json"

P = lambda *a, **kw: print(*a, **kw, flush=True)


@torch.no_grad()
def per_token_global_topk_retention(wrapper, samples):
    """For each token, compute predictor logits across all layers, apply global top-k.
    
    Returns per-layer retention rates averaged across all tokens.
    """
    per_layer_retain_sum = torch.zeros(NUM_LAYERS)
    total_tokens = 0

    for si, sample in enumerate(samples):
        input_ids = sample["input_ids"].unsqueeze(0)  # (1, seq_len)
        t0 = time.time()

        wrapper.forward_dense(input_ids, capture_intermediates=True)
        layer_inputs = wrapper.get_layer_inputs()

        P(f"  Sample {si}: forward done in {time.time()-t0:.0f}s", end="")
        t1 = time.time()

        all_logits = []
        for li in range(NUM_LAYERS):
            logits = wrapper.predictors[li](layer_inputs[li])  # (1, seq_len, inter_dim)
            all_logits.append(logits.squeeze(0))  # (seq_len, inter_dim)

        all_logits = torch.stack(all_logits, dim=0)  # (num_layers, seq_len, inter_dim)
        seq_len = all_logits.shape[1]

        num_keep = int(TOTAL_NEURONS * (1.0 - SPARSITY_TARGET))

        for ti in range(seq_len):
            token_logits = all_logits[:, ti, :]  # (num_layers, inter_dim)
            flat = token_logits.reshape(-1)  # (total_neurons,)
            thr = torch.topk(flat, num_keep).values[-1]
            for li in range(NUM_LAYERS):
                kept = (token_logits[li] >= thr).sum().item()
                per_layer_retain_sum[li] += kept
            total_tokens += 1

        P(f", topk done in {time.time()-t1:.0f}s, total_tokens={total_tokens}")

    per_layer_retention = per_layer_retain_sum / (total_tokens * INTER_DIM)
    return per_layer_retention.tolist()


def main():
    t0 = time.time()

    P("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    P(f"Loading wikitext-2 eval data (seq_len={SEQ_LEN}, max={NUM_SAMPLES})...")
    samples = get_eval_dataset("wikitext2", tokenizer, SEQ_LEN, max_samples=NUM_SAMPLES)
    P(f"  Got {len(samples)} samples")

    P("Loading LLaMA-3.1-8B on CPU (float32)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float32, device_map="cpu",
        attn_implementation="sdpa",
    )
    model.eval()
    P(f"  Model loaded in {time.time()-t0:.0f}s")

    results = {}
    for seed_name, ckpt_path in CHECKPOINTS.items():
        P(f"\n{'='*60}")
        P(f"Processing {seed_name}: {os.path.basename(ckpt_path)}")
        P(f"{'='*60}")

        wrapper = PredictorWrapper(model, bottleneck_size=128)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        wrapper.predictors.load_state_dict(ckpt["predictors"])
        wrapper.predictors.eval()
        wrapper.predictors.float()

        per_layer_ret = per_token_global_topk_retention(wrapper, samples)

        actual_mean = sum(per_layer_ret) / len(per_layer_ret)
        P(f"  Mean retention: {actual_mean:.4f} (target: {1-SPARSITY_TARGET:.2f})")

        results[seed_name] = per_layer_ret
        del wrapper

    # Comparison
    comparison = []
    P(f"\n{'='*70}")
    P("PER-LAYER RETENTION: s42 vs s123 (global top-k, 30% sparsity)")
    P(f"{'='*70}")
    P(f"{'Layer':<6} {'s42 ret':<10} {'s123 ret':<10} {'delta':<10} {'flag'}")
    P("-" * 60)

    for li in range(NUM_LAYERS):
        r42 = round(results["s42"][li], 4)
        r123 = round(results["s123"][li], 4)
        delta = round(r123 - r42, 4)
        flag = ""
        if r42 < 0.50:
            flag += "s42_starved "
        if r123 < 0.50:
            flag += "s123_starved "
        if abs(delta) > 0.10:
            flag += "BIG_DELTA "
        flag = flag.strip()
        comparison.append({
            "layer": li, "s42_retention": r42, "s123_retention": r123,
            "delta": delta, "flag": flag
        })
        P(f"  {li:<4} {r42:<10.4f} {r123:<10.4f} {delta:>+8.4f}   {flag}")

    s42_r = [c["s42_retention"] for c in comparison]
    s123_r = [c["s123_retention"] for c in comparison]

    def avg(lst, a, b): return round(sum(lst[a:b])/(b-a), 4)

    s42_starved_10 = [c["layer"] for c in comparison if c["s42_retention"] < 0.10]
    s123_starved_10 = [c["layer"] for c in comparison if c["s123_retention"] < 0.10]
    s42_starved_50 = [c["layer"] for c in comparison if c["s42_retention"] < 0.50]
    s123_starved_50 = [c["layer"] for c in comparison if c["s123_retention"] < 0.50]
    s42_only = [li for li in s42_starved_50 if li not in s123_starved_50]
    s123_only = [li for li in s123_starved_50 if li not in s42_starved_50]
    max_delta = max(comparison, key=lambda c: abs(c["delta"]))

    summary = {
        "s42_starved_below_10pct": s42_starved_10,
        "s123_starved_below_10pct": s123_starved_10,
        "s42_starved_below_50pct": s42_starved_50,
        "s123_starved_below_50pct": s123_starved_50,
        "s42_only_starved": s42_only,
        "s123_only_starved": s123_only,
        "max_abs_delta_layer": max_delta["layer"],
        "max_abs_delta_value": max_delta["delta"],
        "s42_early_avg": avg(s42_r, 0, 8), "s42_mid_avg": avg(s42_r, 8, 24),
        "s42_knowledge_avg": avg(s42_r, 12, 21), "s42_late_avg": avg(s42_r, 24, 32),
        "s123_early_avg": avg(s123_r, 0, 8), "s123_mid_avg": avg(s123_r, 8, 24),
        "s123_knowledge_avg": avg(s123_r, 12, 21), "s123_late_avg": avg(s123_r, 24, 32),
    }

    P(f"\n--- Summary ---")
    P(f"s42  starved (<10%): {len(s42_starved_10)} layers {s42_starved_10}")
    P(f"s123 starved (<10%): {len(s123_starved_10)} layers {s123_starved_10}")
    P(f"s42  starved (<50%): {len(s42_starved_50)} layers {s42_starved_50}")
    P(f"s123 starved (<50%): {len(s123_starved_50)} layers {s123_starved_50}")
    P(f"Starved only in s42: {s42_only}")
    P(f"Starved only in s123: {s123_only}")
    P(f"Max |delta|: layer {max_delta['layer']} ({max_delta['delta']:+.4f})")
    P(f"\nRegion averages:")
    P(f"  {'Region':<20} {'s42':<10} {'s123':<10} {'delta'}")
    for name, a, b in [("Early (0-7)", 0, 8), ("Mid (8-23)", 8, 24),
                        ("Knowledge (12-20)", 12, 21), ("Late (24-31)", 24, 32)]:
        s42_a, s123_a = avg(s42_r, a, b), avg(s123_r, a, b)
        P(f"  {name:<20} {s42_a:<10} {s123_a:<10} {s123_a - s42_a:+.4f}")

    output = {
        "config": {"sparsity_target": SPARSITY_TARGET, "num_samples": len(samples),
                    "seq_len": SEQ_LEN, "model": MODEL_PATH, "method": "per_token_global_topk"},
        "comparison": comparison, "summary": summary,
        "raw": {"s42": [round(r,4) for r in results["s42"]],
                "s123": [round(r,4) for r in results["s123"]]},
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    P(f"\nResults saved to {OUTPUT_PATH}")
    P(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

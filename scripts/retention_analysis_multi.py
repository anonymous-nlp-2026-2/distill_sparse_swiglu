"""Per-layer retention analysis for multiple sparsity levels (30%/50%/70%)."""

import argparse
import importlib.util
import json
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/root/distill_sparse_swiglu/src")
from data_utils import get_eval_dataset
from predictor import PredictorWrapper

MODEL_PATH = "/root/autodl-tmp/models/llama-3.1-8b"
NUM_CALIB = 50
SEQ_LEN = 2048
DEVICE = "cuda:0"


def best_attn_impl():
    if importlib.util.find_spec("flash_attn"):
        return "flash_attention_2"
    return "sdpa"


def global_topk_masks(layer_scores, sparsity_target):
    items = sorted(layer_scores.items())
    normalized = []
    for li, s in items:
        s_f = s.float()
        mu = s_f.mean()
        normalized.append((li, s_f / mu if mu > 1e-12 else s_f))
    all_s = torch.cat([s for _, s in normalized])
    num_keep = int(len(all_s) * (1.0 - sparsity_target))
    thr = torch.topk(all_s, num_keep).values[-1]
    return {li: (s >= thr).to(torch.bfloat16) for li, s in normalized}


@torch.no_grad()
def compute_layer_scores(wrapper, input_ids, calib_batch_size=4):
    accum_scores = {}
    n_batches = 0
    for start in range(0, input_ids.size(0), calib_batch_size):
        batch = input_ids[start:start + calib_batch_size]
        wrapper.forward_dense(batch, capture_intermediates=True)
        intermediates = wrapper.get_intermediates()
        layer_inputs = wrapper.get_layer_inputs()
        with torch.no_grad():
            for layer_idx in range(len(wrapper.predictors)):
                if layer_idx not in layer_inputs or layer_idx not in intermediates:
                    continue
                inp = layer_inputs[layer_idx]
                logits = wrapper.predictors[layer_idx](inp.to(wrapper.predictors[layer_idx].down.weight.dtype))
                hard_mask = (logits > 0).float()
                mag = intermediates[layer_idx].abs().float()
                scores = (mag * hard_mask).mean(dim=(0, 1))
                if layer_idx not in accum_scores:
                    accum_scores[layer_idx] = scores
                else:
                    accum_scores[layer_idx] = accum_scores[layer_idx] + scores
        n_batches += 1
        if start == 0:
            print(f"  Dry-run batch 0 OK, layers={len(accum_scores)}, "
                  f"score shape={accum_scores[0].shape}")
    layer_scores = {li: accum_scores[li] / n_batches for li in sorted(accum_scores)}
    return layer_scores


def analyze_retention(masks, sparsity_target, n_layers=32, inter_dim=14336):
    per_layer = []
    uniform_baseline = 1.0 - sparsity_target
    for li in range(n_layers):
        mask = masks[li]
        retained = mask.sum().item()
        ratio = retained / inter_dim
        per_layer.append({
            "layer": li,
            "retained_neurons": int(retained),
            "mean_retention": round(ratio, 4),
        })

    retentions = [p["mean_retention"] for p in per_layer]
    early = retentions[:8]
    middle = retentions[8:24]
    knowledge = retentions[12:21]
    late = retentions[24:]

    min_layer = min(per_layer, key=lambda x: x["mean_retention"])
    max_layer = max(per_layer, key=lambda x: x["mean_retention"])

    starved = [p for p in per_layer if p["mean_retention"] < uniform_baseline * 0.7]

    return {
        "per_layer_retention": per_layer,
        "starved_layers": starved,
        "knowledge_layer_avg_retention": round(sum(knowledge) / len(knowledge), 4),
        "early_layer_avg_retention": round(sum(early) / len(early), 4),
        "middle_layer_avg_retention": round(sum(middle) / len(middle), 4),
        "late_layer_avg_retention": round(sum(late) / len(late), 4),
        "min_retention_layer": {"layer": min_layer["layer"], "retention": min_layer["mean_retention"]},
        "max_retention_layer": {"layer": max_layer["layer"], "retention": max_layer["mean_retention"]},
        "uniform_baseline_retention": uniform_baseline,
        "sparsity_target": sparsity_target,
        "num_calib_samples": NUM_CALIB,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparsity_target", type=float, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    sparsity_target = args.sparsity_target
    predictor_path = args.checkpoint
    output_path = args.output

    print(f"=== Retention Analysis: sparsity={sparsity_target}, checkpoint={predictor_path} ===")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        attn_implementation=best_attn_impl(),
    )
    model.eval()

    print("Loading predictor...")
    wrapper = PredictorWrapper(model, bottleneck_size=128)
    ckpt = torch.load(predictor_path, map_location=DEVICE, weights_only=True)
    wrapper.predictors.load_state_dict(ckpt["predictors"])
    wrapper = wrapper.to(DEVICE)
    wrapper.eval()

    print("Preparing calibration data...")
    calib_examples = get_eval_dataset("c4", tokenizer, seq_len=SEQ_LEN, max_samples=NUM_CALIB)
    calib_ids = torch.stack([ex["input_ids"] for ex in calib_examples]).to(DEVICE)
    n_actual = calib_ids.shape[0]
    print(f"  Calibration shape: {calib_ids.shape} ({n_actual} samples)")

    print("Computing layer scores...")
    layer_scores = compute_layer_scores(wrapper, calib_ids, calib_batch_size=4)

    print("Applying global top-k...")
    global_masks = global_topk_masks(layer_scores, sparsity_target)

    total_n = sum(m.numel() for m in global_masks.values())
    total_kept = sum(m.sum().item() for m in global_masks.values())
    actual_sparsity = 1 - total_kept / total_n
    print(f"  Actual global sparsity: {actual_sparsity:.4f}")

    result = analyze_retention(global_masks, sparsity_target)
    result["actual_global_sparsity"] = round(actual_sparsity, 4)
    result["num_calib_samples_actual"] = n_actual

    uniform = 1.0 - sparsity_target
    print(f"\n{'='*60}")
    print(f"PER-LAYER RETENTION (global top-k, {sparsity_target*100:.0f}% target)")
    print(f"{'='*60}")
    print(f"{'Layer':<6} {'Retention':<10} {'Neurons Kept':<14} {'Status'}")
    print("-" * 50)
    for p in result["per_layer_retention"]:
        status = ""
        if p["mean_retention"] < uniform * 0.7:
            status = "STARVED"
        if p["mean_retention"] < uniform * 0.5:
            status = "CRITICAL"
        print(f"  {p['layer']:<4} {p['mean_retention']:<10.4f} "
              f"{p['retained_neurons']:<14} {status}")

    print(f"\n--- Summary ---")
    print(f"Early layers (0-7):       avg retention = {result['early_layer_avg_retention']:.4f}")
    print(f"Middle layers (8-23):     avg retention = {result['middle_layer_avg_retention']:.4f}")
    print(f"Knowledge layers (12-20): avg retention = {result['knowledge_layer_avg_retention']:.4f}")
    print(f"Late layers (24-31):      avg retention = {result['late_layer_avg_retention']:.4f}")
    print(f"Min retention: layer {result['min_retention_layer']['layer']} ({result['min_retention_layer']['retention']:.4f})")
    print(f"Max retention: layer {result['max_retention_layer']['layer']} ({result['max_retention_layer']['retention']:.4f})")
    print(f"Starved layers (<{uniform*0.7:.2f}): {len(result['starved_layers'])}")
    print(f"Uniform baseline: every layer = {uniform:.2f}")

    score_stats = []
    for li in sorted(layer_scores):
        s = layer_scores[li].float()
        score_stats.append({
            "layer": li,
            "mean_score": round(s.mean().item(), 6),
            "std_score": round(s.std().item(), 6),
            "max_score": round(s.max().item(), 6),
            "min_score": round(s.min().item(), 6),
        })
    result["raw_score_stats"] = score_stats

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

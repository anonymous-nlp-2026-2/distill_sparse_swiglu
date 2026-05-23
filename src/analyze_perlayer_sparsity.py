# Analyze per-layer sparsity distribution under global top-k allocation.
# Loads predictor + base model, runs calibration data, computes global top-k masks,
# and reports the actual sparsity achieved at each layer.
# Input: predictor checkpoint, model path, sparsity target
# Output: per-layer sparsity JSON in --output_path
# Dependencies: transformers, torch, datasets

import argparse
import importlib.util
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import get_eval_dataset
from predictor import PredictorWrapper


def _best_attn_impl():
    if importlib.util.find_spec("flash_attn"):
        return "flash_attention_2"
    return "sdpa"


def compute_layer_scores(wrapper, input_ids, calib_batch_size=4):
    """Run calibration data through predictor and accumulate importance scores per layer."""
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
                logits = wrapper.predictors[layer_idx](layer_inputs[layer_idx])
                hard_mask = (logits > 0).float()
                mag = intermediates[layer_idx].abs().float()
                scores = (mag * hard_mask).mean(dim=(0, 1))
                if layer_idx not in accum_scores:
                    accum_scores[layer_idx] = scores
                else:
                    accum_scores[layer_idx] = accum_scores[layer_idx] + scores
        n_batches += 1

    return {li: accum_scores[li] / n_batches for li in sorted(accum_scores)}


def global_topk_masks(layer_scores, sparsity_target):
    """Global top-k: normalize per-layer means, concat, single threshold."""
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


def analyze_masks(masks):
    """Compute per-layer sparsity stats from binary masks."""
    per_layer = {}
    for idx in sorted(masks):
        sp = 1.0 - masks[idx].float().mean().item()
        per_layer[idx] = sp

    values = list(per_layer.values())
    n_layers = len(values)
    total_n = sum(m.numel() for m in masks.values())
    total_z = sum((m == 0).sum().item() for m in masks.values())

    first_half = values[:n_layers // 2]
    second_half = values[n_layers // 2:]

    stats = {
        "per_layer": {str(k): round(v, 6) for k, v in per_layer.items()},
        "summary": {
            "mean": round(sum(values) / len(values), 6),
            "std": round((sum((v - sum(values)/len(values))**2 for v in values) / len(values)) ** 0.5, 6),
            "min": round(min(values), 6),
            "max": round(max(values), 6),
            "min_layer": min(per_layer, key=per_layer.get),
            "max_layer": max(per_layer, key=per_layer.get),
            "model_level": round(total_z / total_n, 6),
            "first_half_mean": round(sum(first_half) / len(first_half), 6),
            "second_half_mean": round(sum(second_half) / len(second_half), 6),
        },
    }
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Analyze per-layer sparsity distribution under global top-k allocation."
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to base LLM (HF format)")
    parser.add_argument("--predictor_path", type=str, required=True,
                        help="Path to predictor checkpoint (.pt)")
    parser.add_argument("--sparsity_target", type=float, default=0.5,
                        help="Global sparsity target (default: 0.5)")
    parser.add_argument("--calibration_samples", type=int, default=256,
                        help="Number of calibration sequences (default: 256)")
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--calib_batch_size", type=int, default=4)
    parser.add_argument("--bottleneck_size", type=int, default=128)
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output JSON path (default: results/perlayer_sparsity_{target}.json)")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Loading model from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation=_best_attn_impl(),
    )
    model.eval()

    print(f"Loading predictor from {args.predictor_path} ...")
    wrapper = PredictorWrapper(model, bottleneck_size=args.bottleneck_size)
    ckpt = torch.load(args.predictor_path, map_location=device, weights_only=True)
    wrapper.predictors.load_state_dict(ckpt["predictors"])
    wrapper.predictors.to(device=device, dtype=torch.bfloat16)
    wrapper.predictors.eval()
    wrapper.gumbel_mask.hard = True

    print("Loading calibration data (WikiText-2) ...")
    calib_data = get_eval_dataset("wikitext2", tokenizer, args.seq_len)
    n_calib = min(args.calibration_samples, len(calib_data))
    calib_ids = torch.stack([calib_data[i]["input_ids"] for i in range(n_calib)]).to(device)
    print(f"  Using {n_calib} calibration sequences")

    print(f"Computing layer scores ...")
    layer_scores = compute_layer_scores(wrapper, calib_ids, calib_batch_size=args.calib_batch_size)

    print(f"Computing global top-k masks (sparsity={args.sparsity_target}) ...")
    masks = global_topk_masks(layer_scores, args.sparsity_target)

    stats = analyze_masks(masks)
    stats["config"] = {
        "model_path": args.model_path,
        "predictor_path": args.predictor_path,
        "sparsity_target": args.sparsity_target,
        "calibration_samples": n_calib,
        "seq_len": args.seq_len,
    }

    print(f"\nPer-layer sparsity (global top-k, target={args.sparsity_target}):")
    for idx_str in sorted(stats["per_layer"], key=lambda x: int(x)):
        sp = stats["per_layer"][idx_str]
        bar = "#" * int(sp * 50)
        print(f"  Layer {int(idx_str):2d}: {sp:.4f}  {bar}")

    s = stats["summary"]
    print(f"\nSummary:")
    print(f"  Mean:  {s['mean']:.4f}")
    print(f"  Std:   {s['std']:.4f}")
    print(f"  Min:   {s['min']:.4f} (layer {s['min_layer']})")
    print(f"  Max:   {s['max']:.4f} (layer {s['max_layer']})")
    print(f"  First half mean:  {s['first_half_mean']:.4f}")
    print(f"  Second half mean: {s['second_half_mean']:.4f}")
    print(f"  Model-level:      {s['model_level']:.4f}")

    output_path = args.output_path
    if output_path is None:
        pct = int(args.sparsity_target * 100)
        output_path = f"results/perlayer_sparsity_{pct}pct.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()

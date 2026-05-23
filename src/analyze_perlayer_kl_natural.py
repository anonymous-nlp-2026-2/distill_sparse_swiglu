# Per-layer sparsity analysis: KL predictor's NATURAL output vs. magnitude baseline.
#
# Motivation: Reviewer asked for mechanistic explanation of KL's MMLU/GSM8K regression.
# Hypothesis: KL's mode-covering objective concentrates pruning in middle layers
# (knowledge storage, L8-L23 per Geva et al. 2023), causing factual knowledge loss.
#
# Method:
#   KL natural sparsity   = 1 - (predictor_logits > 0).mean() per layer.
#                           No per-layer enforcement; this is what the KL-trained
#                           predictor *actually* decides to prune.
#   Magnitude baseline    = global top-k over |gate * up| activation magnitude,
#                           matched to KL's overall sparsity (TEAL-greedy policy).
#
# Output: per-layer table + early/middle/late aggregates + JSON.

import argparse
import gc
import importlib.util
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import get_eval_dataset
from predictor import SparsityPredictor
from transformers import AutoModelForCausalLM, AutoTokenizer


def _best_attn_impl():
    if importlib.util.find_spec("flash_attn"):
        return "flash_attention_2"
    return "sdpa"


@torch.no_grad()
def compute_kl_logits_and_mag(model, ckpt_path, calib_ids, device, bottleneck_size=128):
    config = model.config
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    predictor_sd = ckpt["predictors"]

    kl_keep_sum = {li: torch.zeros(intermediate_size, dtype=torch.float64) for li in range(num_layers)}
    kl_keep_count = {li: 0 for li in range(num_layers)}
    mag_sum = {li: torch.zeros(intermediate_size, dtype=torch.float64) for li in range(num_layers)}
    mag_count = {li: 0 for li in range(num_layers)}

    saved_fwds = {}
    for li, layer in enumerate(model.model.layers):
        saved_fwds[li] = layer.mlp.forward

    def make_patched(idx, mlp, predictor):
        gp, up, dp, af = mlp.gate_proj, mlp.up_proj, mlp.down_proj, mlp.act_fn

        def fwd(x):
            with torch.no_grad():
                logits = predictor(x)
                keep = (logits > 0).float()
                kl_keep_sum[idx].add_(keep.sum(dim=(0, 1)).double().cpu())
                kl_keep_count[idx] += keep.shape[0] * keep.shape[1]

                h = af(gp(x)) * up(x)
                mag_sum[idx].add_(h.abs().float().sum(dim=(0, 1)).double().cpu())
                mag_count[idx] += h.shape[0] * h.shape[1]
            return dp(h)

        return fwd

    predictors = []
    for li in range(num_layers):
        pred = SparsityPredictor(hidden_size, intermediate_size, bottleneck_size)
        prefix = f"{li}."
        sd = {k[len(prefix):]: v for k, v in predictor_sd.items() if k.startswith(prefix)}
        pred.load_state_dict(sd)
        pred.to(device=device, dtype=torch.bfloat16).eval()
        predictors.append(pred)

    for li, layer in enumerate(model.model.layers):
        layer.mlp.forward = make_patched(li, layer.mlp, predictors[li])

    for start in range(calib_ids.size(0)):
        model(calib_ids[start:start + 1])
        torch.cuda.empty_cache()

    for li, fwd in saved_fwds.items():
        model.model.layers[li].mlp.forward = fwd

    for p in predictors:
        del p
    del predictors
    gc.collect()
    torch.cuda.empty_cache()

    kl_retention_per_neuron = {li: (kl_keep_sum[li] / kl_keep_count[li]) for li in range(num_layers)}
    mag_mean_per_neuron = {li: (mag_sum[li] / mag_count[li]) for li in range(num_layers)}
    return kl_retention_per_neuron, mag_mean_per_neuron


def kl_natural_per_layer_sparsity(kl_retention_per_neuron):
    return {li: 1.0 - kl_retention_per_neuron[li].mean().item()
            for li in kl_retention_per_neuron}


def magnitude_global_topk(mag_mean_per_neuron, target_sparsity):
    items = sorted(mag_mean_per_neuron.items())
    normalized = []
    for li, s in items:
        s_f = s.float()
        mu = s_f.mean()
        normalized.append((li, s_f / mu if mu > 1e-12 else s_f))
    all_s = torch.cat([s for _, s in normalized])
    num_keep = int(len(all_s) * (1.0 - target_sparsity))
    thr = torch.topk(all_s, num_keep).values[-1]
    return {li: ((s < thr).float().mean().item()) for li, s in normalized}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/root/autodl-tmp/models/llama-3.1-8b")
    ap.add_argument("--predictor_path", default="checkpoints/kl_sparsity30_s42_v3/predictor_kl_normalized.pt")
    ap.add_argument("--calib_samples", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--bottleneck_size", type=int, default=128)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--output_path", default="results/perlayer_kl_natural_vs_magnitude.json")
    args = ap.parse_args()

    device = torch.device(args.device)

    print(f"Loading model: {args.model_path}")
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
    for p in model.parameters():
        p.requires_grad = False

    print("Loading WikiText-2 calibration ...")
    calib = get_eval_dataset("wikitext2", tokenizer, args.seq_len)
    n = min(args.calib_samples, len(calib))
    calib_ids = torch.stack([calib[i]["input_ids"] for i in range(n)]).to(device)
    print(f"  {n} sequences x {args.seq_len} tokens")

    print(f"Loading KL predictor: {args.predictor_path}")
    kl_keep, mag_mean = compute_kl_logits_and_mag(
        model, args.predictor_path, calib_ids, device, bottleneck_size=args.bottleneck_size
    )

    num_layers = len(kl_keep)
    kl_sparsity = kl_natural_per_layer_sparsity(kl_keep)
    overall_kl_sparsity = sum(kl_sparsity.values()) / num_layers
    print(f"\nKL natural overall sparsity: {overall_kl_sparsity:.4f}")

    print(f"Computing magnitude-greedy allocation matched to KL overall ({overall_kl_sparsity:.4f})")
    mag_sparsity = magnitude_global_topk(mag_mean, overall_kl_sparsity)

    delta = {li: kl_sparsity[li] - mag_sparsity[li] for li in range(num_layers)}

    early = list(range(0, 8))
    middle = list(range(8, 24))
    late = list(range(24, num_layers))

    def avg(idxs, d):
        return sum(d[i] for i in idxs) / len(idxs)

    summary = {
        "overall_kl_sparsity": round(overall_kl_sparsity, 6),
        "overall_mag_sparsity": round(sum(mag_sparsity.values()) / num_layers, 6),
        "early_kl_mean": round(avg(early, kl_sparsity), 6),
        "middle_kl_mean": round(avg(middle, kl_sparsity), 6),
        "late_kl_mean": round(avg(late, kl_sparsity), 6),
        "early_mag_mean": round(avg(early, mag_sparsity), 6),
        "middle_mag_mean": round(avg(middle, mag_sparsity), 6),
        "late_mag_mean": round(avg(late, mag_sparsity), 6),
        "early_delta_mean": round(avg(early, delta), 6),
        "middle_delta_mean": round(avg(middle, delta), 6),
        "late_delta_mean": round(avg(late, delta), 6),
    }

    print(f"\n{'='*68}")
    print(f"{'Layer':>6} | {'KL Sparsity':>12} | {'Mag Sparsity':>13} | {'Delta':>12}")
    print('-' * 68)
    for li in range(num_layers):
        d = delta[li]
        print(f"  L{li:<3} | {kl_sparsity[li]:>12.4f} | {mag_sparsity[li]:>13.4f} | {d:>+12.4f}")
    print('=' * 68)
    print(f"\nRegion averages (KL natural | Magnitude greedy | Delta):")
    print(f"  Early   (L0-L7)   : {summary['early_kl_mean']:.4f} | {summary['early_mag_mean']:.4f} | {summary['early_delta_mean']:+.4f}")
    print(f"  Middle  (L8-L23)  : {summary['middle_kl_mean']:.4f} | {summary['middle_mag_mean']:.4f} | {summary['middle_delta_mean']:+.4f}")
    print(f"  Late    (L24-L31) : {summary['late_kl_mean']:.4f} | {summary['late_mag_mean']:.4f} | {summary['late_delta_mean']:+.4f}")

    results = {
        "config": {
            "model_path": args.model_path,
            "predictor_path": args.predictor_path,
            "calib_samples": n,
            "seq_len": args.seq_len,
            "bottleneck_size": args.bottleneck_size,
        },
        "per_layer": {
            str(li): {
                "kl_sparsity": round(kl_sparsity[li], 6),
                "mag_sparsity": round(mag_sparsity[li], 6),
                "delta": round(delta[li], 6),
            }
            for li in range(num_layers)
        },
        "summary": summary,
    }
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output_path}")


if __name__ == "__main__":
    main()

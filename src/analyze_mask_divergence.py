"""Per-layer Jaccard divergence between KL-predicted and magnitude-based masks.

Usage:
  cd /root/distill_sparse_swiglu
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/root/autodl-tmp/.hf_cache \
    python3 src/analyze_mask_divergence.py \
      --model_path /root/autodl-tmp/models/qwen2.5-14b \
      --predictor_path checkpoints/qwen14b_kl30_s42/predictor_kl_normalized.pt \
      --model_name qwen14b --sparsity 0.3 --device cuda:0
"""

import argparse
import gc
import importlib.util
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predictor import SparsityPredictor
from data_utils import get_eval_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def _best_attn_impl():
    if importlib.util.find_spec("flash_attn"):
        return "flash_attention_2"
    return "sdpa"


@torch.no_grad()
def get_magnitude_scores(model, calib_ids, batch_size=1):
    """Per-neuron mean |act(gate)*up| across calibration data."""
    accum = {}
    n_batches = 0
    saved = {}

    for li, layer in enumerate(model.model.layers):
        saved[li] = layer.mlp.forward
        mlp = layer.mlp
        gp, up, dp, af = mlp.gate_proj, mlp.up_proj, mlp.down_proj, mlp.act_fn
        def _make(idx, _gp, _up, _dp, _af):
            def fwd(x):
                h = _af(_gp(x)) * _up(x)
                mag = h.abs().mean(dim=(0, 1))
                if idx in accum:
                    accum[idx] = accum[idx] + mag
                else:
                    accum[idx] = mag
                return _dp(h)
            return fwd
        layer.mlp.forward = _make(li, gp, up, dp, af)

    for start in range(0, calib_ids.size(0), batch_size):
        model(calib_ids[start:start + batch_size])
        n_batches += 1
        torch.cuda.empty_cache()

    for li, fwd in saved.items():
        model.model.layers[li].mlp.forward = fwd

    return {li: (accum[li] / n_batches).cpu() for li in sorted(accum)}


@torch.no_grad()
def get_kl_scores(model, ckpt_path, calib_ids, device, bottleneck_size=128):
    """Per-layer mean sigmoid scores from KL predictor.

    Processes one sample at a time and captures layer inputs via hooks
    instead of PredictorWrapper to minimize memory usage.
    """
    config = model.config
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    predictor_sd = ckpt["predictors"]

    layer_scores = {}
    n_samples = 0

    for start in range(calib_ids.size(0)):
        batch = calib_ids[start:start + 1]

        layer_inputs = {}
        hooks = []
        for li, layer in enumerate(model.model.layers):
            def _make_hook(idx):
                def hook_fn(module, args, kwargs=None):
                    inp = args[0] if args else kwargs.get("hidden_states")
                    layer_inputs[idx] = inp.detach()
                return hook_fn
            h = layer.register_forward_pre_hook(_make_hook(li))
            hooks.append(h)

        model(batch)

        for h in hooks:
            h.remove()

        for li in range(num_layers):
            if li not in layer_inputs:
                continue
            pred = SparsityPredictor(hidden_size, intermediate_size, bottleneck_size)
            pred_key_prefix = f"{li}."
            pred_sd = {k[len(pred_key_prefix):]: v for k, v in predictor_sd.items() if k.startswith(pred_key_prefix)}
            pred.load_state_dict(pred_sd)
            pred.to(device=device, dtype=torch.bfloat16)
            pred.eval()

            logits = pred(layer_inputs[li].to(device))
            sig = torch.sigmoid(logits.float()).mean(dim=(0, 1))
            if li not in layer_scores:
                layer_scores[li] = sig.cpu()
            else:
                layer_scores[li] = layer_scores[li] + sig.cpu()

            del pred, logits, sig
            del layer_inputs[li]

        del layer_inputs
        torch.cuda.empty_cache()
        n_samples += 1
        if n_samples % 8 == 0:
            print(f"    KL scoring: {n_samples}/{calib_ids.size(0)} samples")

    return {li: layer_scores[li] / n_samples for li in sorted(layer_scores)}


def scores_to_mask(scores, sparsity):
    s = scores.float()
    n = s.numel()
    k = int(n * (1.0 - sparsity))
    thr = torch.topk(s, k).values[-1]
    return (s >= thr).int()


def jaccard_distance(mask_a, mask_b):
    a = mask_a.bool()
    b = mask_b.bool()
    intersection = (a & b).sum().float()
    union = (a | b).sum().float()
    if union == 0:
        return 0.0
    return 1.0 - (intersection / union).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--predictor_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--sparsity", type=float, default=0.3)
    parser.add_argument("--calib_samples", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--bottleneck_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="/root/distill_sparse_swiglu/results")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=_best_attn_impl(),
        device_map={"": device},
    )
    model.eval()

    print("Loading calibration data (WikiText-2)...")
    data = get_eval_dataset("wikitext2", tokenizer, seq_len=args.seq_len, max_samples=args.calib_samples)
    n = min(args.calib_samples, len(data))
    calib_ids = torch.stack([data[i]["input_ids"] for i in range(n)]).to(device)
    print(f"  {n} sequences, shape {calib_ids.shape}")

    print("Computing magnitude scores...")
    mag_scores = get_magnitude_scores(model, calib_ids, batch_size=1)
    gc.collect()
    torch.cuda.empty_cache()
    num_layers = len(mag_scores)
    print(f"  {num_layers} layers, dim={mag_scores[0].numel()}")

    print(f"Computing KL predictor scores from {args.predictor_path}...")
    kl_scores = get_kl_scores(model, args.predictor_path, calib_ids, device,
                              bottleneck_size=args.bottleneck_size)
    print(f"  {len(kl_scores)} layers")

    sparsity = args.sparsity
    print(f"\nComputing per-layer Jaccard distance at {sparsity*100:.0f}% sparsity...")
    per_layer_jaccard = {}
    per_layer_kl_retention = {}
    per_layer_mag_retention = {}

    for li in range(num_layers):
        kl_mask = scores_to_mask(kl_scores[li], sparsity)
        mag_mask = scores_to_mask(mag_scores[li], sparsity)
        jd = jaccard_distance(kl_mask, mag_mask)
        kl_ret = kl_mask.float().mean().item()
        mag_ret = mag_mask.float().mean().item()
        per_layer_jaccard[li] = round(jd, 6)
        per_layer_kl_retention[li] = round(kl_ret, 6)
        per_layer_mag_retention[li] = round(mag_ret, 6)

    jaccard_vals = list(per_layer_jaccard.values())
    mean_jd = sum(jaccard_vals) / len(jaccard_vals)
    std_jd = (sum((v - mean_jd)**2 for v in jaccard_vals) / len(jaccard_vals)) ** 0.5
    max_layer = max(per_layer_jaccard, key=per_layer_jaccard.get)
    min_layer = min(per_layer_jaccard, key=per_layer_jaccard.get)

    first_q = jaccard_vals[:num_layers//4]
    last_q = jaccard_vals[3*num_layers//4:]
    mid = jaccard_vals[num_layers//4:3*num_layers//4]

    results = {
        "model_name": args.model_name,
        "model_path": args.model_path,
        "predictor_path": args.predictor_path,
        "sparsity": sparsity,
        "num_layers": num_layers,
        "calib_samples": n,
        "seq_len": args.seq_len,
        "per_layer_jaccard": {str(k): v for k, v in per_layer_jaccard.items()},
        "per_layer_kl_retention": {str(k): v for k, v in per_layer_kl_retention.items()},
        "per_layer_mag_retention": {str(k): v for k, v in per_layer_mag_retention.items()},
        "summary": {
            "mean_jaccard": round(mean_jd, 6),
            "std_jaccard": round(std_jd, 6),
            "max_jaccard": round(max(jaccard_vals), 6),
            "max_jaccard_layer": max_layer,
            "min_jaccard": round(min(jaccard_vals), 6),
            "min_jaccard_layer": min_layer,
            "first_quarter_mean": round(sum(first_q)/len(first_q), 6) if first_q else None,
            "middle_half_mean": round(sum(mid)/len(mid), 6) if mid else None,
            "last_quarter_mean": round(sum(last_q)/len(last_q), 6) if last_q else None,
        },
    }

    print(f"\n{'='*60}")
    print(f"Per-layer Jaccard distance ({args.model_name}, {sparsity*100:.0f}% sparsity)")
    print(f"{'='*60}")
    for li in range(num_layers):
        jd = per_layer_jaccard[li]
        bar = "#" * int(jd * 50)
        print(f"  Layer {li:2d}: {jd:.4f}  {bar}")

    print(f"\nSummary:")
    print(f"  Mean Jaccard:  {mean_jd:.4f}")
    print(f"  Std Jaccard:   {std_jd:.4f}")
    print(f"  Max: {max(jaccard_vals):.4f} (layer {max_layer})")
    print(f"  Min: {min(jaccard_vals):.4f} (layer {min_layer})")
    if results['summary']['first_quarter_mean'] is not None:
        print(f"  First quarter mean: {results['summary']['first_quarter_mean']:.4f}")
    if results['summary']['middle_half_mean'] is not None:
        print(f"  Middle half mean:   {results['summary']['middle_half_mean']:.4f}")
    if results['summary']['last_quarter_mean'] is not None:
        print(f"  Last quarter mean:  {results['summary']['last_quarter_mean']:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    sp_pct = int(sparsity * 100)
    out_path = os.path.join(args.output_dir, f"{args.model_name}_mask_divergence_{sp_pct}pct.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

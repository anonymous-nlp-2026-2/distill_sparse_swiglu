# lm-evaluation-harness with uniform per-layer sparsity allocation.
# Extends lm_eval_sparse.py with --allocation flag:
#   global  — original behavior: single threshold across all layers (uneven per-layer sparsity)
#   uniform — each layer independently does top-k, so every layer has exactly sparsity_target
# The uniform approach prevents sparsity from concentrating in later layers, which causes
# downstream task collapse (good PPL but random MMLU/GSM8K).
# Input: predictor checkpoint, model path, task list, allocation mode
# Output: lm-eval results JSON in --output_path
# Dependencies: lm-eval >= 0.4, transformers, torch, datasets

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


def _global_topk_masks(layer_scores, sparsity_target):
    """Original global allocation: normalize per-layer means, single threshold across all layers."""
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


def _uniform_topk_masks(layer_scores, sparsity_target):
    """Uniform per-layer allocation: each layer independently keeps top-(1-sparsity) neurons.

    Unlike global top-k which applies a single threshold across all layers (causing
    sparsity to concentrate in later layers), this ensures every layer has exactly
    the same sparsity ratio equal to sparsity_target.
    """
    masks = {}
    for li in sorted(layer_scores):
        s = layer_scores[li].float()
        num_keep = int(len(s) * (1.0 - sparsity_target))
        if num_keep >= len(s):
            masks[li] = torch.ones_like(s, dtype=torch.bfloat16)
        elif num_keep <= 0:
            masks[li] = torch.zeros_like(s, dtype=torch.bfloat16)
        else:
            thr = torch.topk(s, num_keep).values[-1]
            masks[li] = (s >= thr).to(torch.bfloat16)
        actual_sparsity = 1 - masks[li].float().mean()
        assert abs(actual_sparsity - sparsity_target) < 0.02, f"Layer {li}: sparsity {actual_sparsity:.3f} deviates from target {sparsity_target:.3f}"
    return masks


def teal_masks(wrapper, input_ids, sparsity_target, allocation="uniform", calib_batch_size=4):
    """Compute static masks from predictor scores.

    Args:
        allocation: "global" for original single-threshold, "uniform" for per-layer independent top-k.
    """
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
                scores = torch.sigmoid(logits).mean(dim=(0, 1))
                if layer_idx not in accum_scores:
                    accum_scores[layer_idx] = scores
                else:
                    accum_scores[layer_idx] = accum_scores[layer_idx] + scores
        n_batches += 1

    layer_scores = {li: accum_scores[li] / n_batches for li in sorted(accum_scores)}

    if allocation == "global":
        return _global_topk_masks(layer_scores, sparsity_target)
    elif allocation == "uniform":
        return _uniform_topk_masks(layer_scores, sparsity_target)
    else:
        raise ValueError(f"Unknown allocation: {allocation}. Use 'global' or 'uniform'.")


def apply_static_masks(model, global_masks, device):
    for layer_idx, layer in enumerate(model.model.layers):
        if layer_idx not in global_masks:
            continue
        mask = global_masks[layer_idx].to(device=device, dtype=layer.mlp.gate_proj.weight.dtype)
        mlp = layer.mlp

        def make_masked_fwd(gp, up, dp, af, m):
            def fwd(x):
                intermediate = af(gp(x)) * up(x)
                return dp(intermediate * m.unsqueeze(0).unsqueeze(0))
            return fwd

        layer.mlp.forward = make_masked_fwd(
            mlp.gate_proj, mlp.up_proj, mlp.down_proj, mlp.act_fn, mask
        )


def print_mask_stats(name, masks):
    print(f"\nPer-layer sparsity ({name}):")
    for idx in sorted(masks):
        sp = 1.0 - masks[idx].mean().item()
        print(f"  Layer {idx:2d}: {sp:.3f}")
    total_n = sum(m.numel() for m in masks.values())
    total_z = sum((m == 0).sum().item() for m in masks.values())
    sp = total_z / total_n
    print(f"  Model-level: {sp:.3f}")
    return sp


def main():
    parser = argparse.ArgumentParser(
        description="lm-eval with sparse Llama (global or uniform per-layer sparsity allocation)"
    )
    parser.add_argument("--model_name_or_path", type=str,
                        default="/root/autodl-tmp/models/llama-3.1-8b")
    parser.add_argument("--predictor_path", type=str, required=True)
    parser.add_argument("--sparsity_target", type=float, default=0.5)
    parser.add_argument("--allocation", type=str, default="uniform",
                        choices=["global", "uniform"],
                        help="Sparsity allocation: 'global' (single threshold) or 'uniform' (per-layer top-k)")
    parser.add_argument("--calibration_samples", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--tasks", type=str, required=True,
                        help="Comma-separated lm-eval task names")
    parser.add_argument("--num_fewshot", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit samples per task (for dry-run)")
    parser.add_argument("--batch_size", type=str, default="auto")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Loading model from {args.model_name_or_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation=_best_attn_impl(),
    )
    model.eval()

    print(f"Loading predictor from {args.predictor_path} ...")
    wrapper = PredictorWrapper(model, bottleneck_size=128)
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

    print(f"Computing masks (allocation={args.allocation}, sparsity={args.sparsity_target}) ...")
    masks = teal_masks(wrapper, calib_ids, args.sparsity_target,
                       allocation=args.allocation)
    actual_sp = print_mask_stats(f"TEAL-{args.allocation}", masks)

    # Restore original MLP forwards before applying static masks
    for layer_idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        gp, up, dp, af = mlp.gate_proj, mlp.up_proj, mlp.down_proj, mlp.act_fn
        def _make_orig(g, u, d, a):
            def fwd(x):
                return d(a(g(x)) * u(x))
            return fwd
        layer.mlp.forward = _make_orig(gp, up, dp, af)

    apply_static_masks(model, masks, device)
    print("Static masks applied to model.")

    del wrapper, ckpt, calib_ids, calib_data
    torch.cuda.empty_cache()

    # --- Run lm-eval ---
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    task_list = [t.strip() for t in args.tasks.split(",")]
    print(f"\nlm-eval tasks: {task_list}")

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
    )

    eval_kwargs = dict(
        model=lm,
        tasks=task_list,
        batch_size=args.batch_size,
    )
    if args.num_fewshot is not None:
        eval_kwargs["num_fewshot"] = args.num_fewshot
    if args.limit is not None:
        eval_kwargs["limit"] = args.limit

    results = lm_eval.simple_evaluate(**eval_kwargs)

    # --- Print and save results ---
    print(f"\n{'='*60}")
    print(f"Sparse Llama-3.1-8B (allocation={args.allocation}, sparsity={args.sparsity_target})")
    print(f"{'='*60}")
    if "results" in results:
        for task_name, task_res in results["results"].items():
            print(f"\n  {task_name}:")
            for metric, val in task_res.items():
                if isinstance(val, (int, float)):
                    print(f"    {metric}: {val:.4f}")
                else:
                    print(f"    {metric}: {val}")

    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)
        out_file = os.path.join(args.output_path, "results.json")
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {out_file}")

        summary = {}
        if "results" in results:
            for task_name, task_res in results["results"].items():
                summary[task_name] = {
                    k: v for k, v in task_res.items()
                    if isinstance(v, (int, float))
                }
        summary["_meta"] = {
            "allocation": args.allocation,
            "sparsity_target": args.sparsity_target,
            "actual_model_sparsity": actual_sp,
        }
        summary_file = os.path.join(args.output_path, "summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    main()

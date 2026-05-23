# Vanilla TEAL (training-free) downstream eval via lm-evaluation-harness.
# No predictor needed - uses activation magnitude global top-k masking.

import argparse
import importlib.util
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import get_eval_dataset


def _best_attn_impl():
    if importlib.util.find_spec("flash_attn"):
        return "flash_attention_2"
    return "sdpa"


@torch.no_grad()
def vanilla_teal_global_masks(model, input_ids, sparsity_target, calib_batch_size=4):
    """Activation-magnitude TEAL: rank neurons by mean |intermediate|, global top-k."""
    accum_magnitudes = {}
    n_batches = 0

    saved_forwards = {}
    for layer_idx, layer in enumerate(model.model.layers):
        saved_forwards[layer_idx] = layer.mlp.forward
        mlp = layer.mlp
        gp, up, dp, af = mlp.gate_proj, mlp.up_proj, mlp.down_proj, mlp.act_fn

        def make_capturing_fwd(lidx, _gp, _up, _dp, _af):
            def fwd(x):
                intermediate = _af(_gp(x)) * _up(x)
                mag = intermediate.abs().float().mean(dim=(0, 1))
                if lidx not in accum_magnitudes:
                    accum_magnitudes[lidx] = mag
                else:
                    accum_magnitudes[lidx] = accum_magnitudes[lidx] + mag
                return _dp(intermediate)
            return fwd

        layer.mlp.forward = make_capturing_fwd(layer_idx, gp, up, dp, af)

    for start in range(0, input_ids.size(0), calib_batch_size):
        batch = input_ids[start:start + calib_batch_size]
        model(batch)
        n_batches += 1

    for layer_idx, fwd in saved_forwards.items():
        model.model.layers[layer_idx].mlp.forward = fwd

    # Per-layer mean normalization (same as lm_eval_sparse.py)
    layer_scores = {}
    for layer_idx in sorted(accum_magnitudes):
        s = accum_magnitudes[layer_idx] / n_batches
        mu = s.mean()
        layer_scores[layer_idx] = s / mu if mu > 1e-12 else s

    all_s = torch.cat(list(layer_scores.values()))
    num_keep = int(len(all_s) * (1.0 - sparsity_target))
    thr = torch.topk(all_s, num_keep).values[-1]
    return {li: (layer_scores[li] >= thr).to(torch.bfloat16) for li in layer_scores}


def apply_static_masks(model, global_masks, device):
    for layer_idx, layer in enumerate(model.model.layers):
        if layer_idx not in global_masks:
            continue
        mask = global_masks[layer_idx].to(device=device, dtype=layer.mlp.gate_proj.weight.dtype)
        mlp = layer.mlp
        gp, up, dp, af = mlp.gate_proj, mlp.up_proj, mlp.down_proj, mlp.act_fn

        def make_masked_fwd(_gp, _up, _dp, _af, m):
            def fwd(x):
                intermediate = _af(_gp(x)) * _up(x)
                return _dp(intermediate * m.unsqueeze(0).unsqueeze(0))
            return fwd

        layer.mlp.forward = make_masked_fwd(gp, up, dp, af, mask)


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str,
                        default="/root/distill_sparse_swiglu/models/llama-3.1-8b")
    parser.add_argument("--sparsity_target", type=float, default=0.5)
    parser.add_argument("--calibration_samples", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--tasks", type=str, default="arc_challenge,winogrande,mmlu,gsm8k")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_fewshot", type=int, default=None)
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--output_path", type=str,
                        default="/root/distill_sparse_swiglu/results/lm_eval_teal_50pct/")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    print(f"Device: {device}")

    print(f"Loading model {args.model_name_or_path} ...")
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

    # --- Compute vanilla TEAL masks ---
    print("Loading calibration data (WikiText-2) ...")
    calib_data = get_eval_dataset("wikitext2", tokenizer, args.seq_len)
    n_calib = min(args.calibration_samples, len(calib_data))
    calib_ids = torch.stack([calib_data[i]["input_ids"] for i in range(n_calib)]).to(device)
    print(f"  Using {n_calib} calibration sequences")

    print(f"Computing vanilla TEAL global masks (sparsity={args.sparsity_target}) ...")
    global_masks = vanilla_teal_global_masks(model, calib_ids, args.sparsity_target)
    actual_sp = print_mask_stats("vanilla-TEAL", global_masks)

    # --- Apply static masks ---
    apply_static_masks(model, global_masks, device)
    print("Static masks applied to model.")

    del calib_ids, calib_data
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
    print(f"Vanilla TEAL Llama-3.1-8B (sparsity={args.sparsity_target})")
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

        summary = {"method": "vanilla_teal", "sparsity": args.sparsity_target,
                    "actual_sparsity": actual_sp}
        if "results" in results:
            for task_name, task_res in results["results"].items():
                summary[task_name] = {
                    k: v for k, v in task_res.items()
                    if isinstance(v, (int, float))
                }
        summary_file = os.path.join(args.output_path, "summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    main()

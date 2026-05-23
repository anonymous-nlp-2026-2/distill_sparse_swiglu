"""Benchmark evaluation for sparsity predictor checkpoints via lm-evaluation-harness.

Input:
  --checkpoint PATH       Predictor .pt file (required for predictor modes)
  --tasks LIST            Comma-separated: wikitext2,arc_challenge,winogrande,mmlu,gsm8k
  --sparsity_mode MODE    "per_token" (dynamic predictor mask) or "teal_global" (static calibrated mask)
  --baseline_mode MODE    Training-free: teal_vanilla,teal_greedy,wina,wina_greedy,rsparse
  --dense                 Evaluate unsparsified model
  --exp_id ID             Experiment identifier -> results/benchmark/{exp_id}/results.json

Output:
  results/benchmark/{exp_id}/results.json

Dependencies:
  lm-eval >= 0.4.0, transformers, torch, datasets
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import (
    _best_attn_impl,
    _make_original_fwd,
    evaluate_perplexity,
    evaluate_true_dense,
    teal_global_masks,
    vanilla_teal_global_masks,
    _collect_swiglu_magnitudes,
    _print_mask_stats,
    teal_activation_magnitude_masks,
    wina_global_masks,
    rsparse_global_masks,
    collect_layer_inputs,
)
from data_utils import get_eval_dataset
from predictor import PredictorWrapper, CompensationNetwork
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = "/root/distill_sparse_swiglu"
DEFAULT_MODEL = "/root/autodl-tmp/models/llama-3.1-8b"

# Default few-shot counts per task (following standard evaluation practice)
LM_EVAL_DEFAULTS = {
    # LongBench tasks (generation-based, 0-shot)
    "longbench": {"num_fewshot": 0},
    "longbench_single": {"num_fewshot": 0},
    "longbench_multi": {"num_fewshot": 0},
    "longbench_summarization": {"num_fewshot": 0},
    "longbench_fewshot": {"num_fewshot": 0},
    "longbench_narrativeqa": {"num_fewshot": 0},
    "longbench_qasper": {"num_fewshot": 0},
    "longbench_gov_report": {"num_fewshot": 0},
    "longbench_multi_news": {"num_fewshot": 0},
    "arc_challenge": {"num_fewshot": 25},
    "winogrande": {"num_fewshot": 5},
    "mmlu": {"num_fewshot": 5},
    "gsm8k": {"num_fewshot": 8},
    "hellaswag": {"num_fewshot": 10},
    "piqa": {"num_fewshot": 0},
    "boolq": {"num_fewshot": 0},
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark evaluation for sparse SwiGLU predictors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Per-token predictor eval on ARC + WikiText-2
  python benchmark_eval.py --exp_id kl_s42_pertoken \\
      --checkpoint checkpoints/mvp_kl_norm_v2_s42/predictor_kl_normalized.pt \\
      --sparsity_mode per_token --tasks wikitext2,arc_challenge

  # TEAL global mask from predictor
  python benchmark_eval.py --exp_id kl_s42_teal \\
      --checkpoint checkpoints/mvp_kl_norm_v2_s42/predictor_kl_normalized.pt \\
      --sparsity_mode teal_global --tasks wikitext2,arc_challenge,mmlu

  # Training-free baseline
  python benchmark_eval.py --exp_id vanilla_teal_50 \\
      --baseline_mode teal_vanilla --tasks wikitext2,arc_challenge

  # Dense baseline
  python benchmark_eval.py --exp_id dense \\
      --dense --tasks wikitext2,arc_challenge,mmlu
""",
    )
    p.add_argument("--model_name_or_path", type=str, default=DEFAULT_MODEL)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to predictor .pt checkpoint")
    p.add_argument("--tasks", type=str, default="wikitext2,arc_challenge,winogrande",
                   help="Comma-separated task list")
    p.add_argument("--sparsity_mode", type=str, default="per_token",
                   choices=["per_token", "teal_global"],
                   help="per_token: dynamic predictor mask; teal_global: static calibrated masks")
    p.add_argument("--baseline_mode", type=str, default=None,
                   choices=["teal_vanilla", "teal_greedy", "wina", "wina_greedy", "rsparse"],
                   help="Training-free baseline mode (no checkpoint needed)")
    p.add_argument("--dense", action="store_true",
                   help="Evaluate dense (no sparsity) model")
    p.add_argument("--exp_id", type=str, required=True,
                   help="Experiment identifier for output directory")
    p.add_argument("--sparsity_target", type=float, default=0.5)
    p.add_argument("--allocation", type=str, default="global_topk",
                   choices=["global_topk", "uniform"],
                   help="Sparsity budget allocation: global_topk (single threshold) or uniform (per-layer independent top-k)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--multi_gpu", action="store_true",
                   help="Use device_map=auto for multi-GPU inference")
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--calibration_samples", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=8,
                   help="Batch size for lm-eval tasks")
    p.add_argument("--bottleneck_size", type=int, default=128)
    p.add_argument("--svd_rank", type=int, default=256,
                   help="SVD rank for R-Sparse baseline")
    p.add_argument("--max_eval_samples", type=int, default=None,
                   help="Max samples for WikiText-2 PPL eval")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_path, device, multi_gpu=False):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        attn_implementation=_best_attn_impl(),
    )
    if multi_gpu:
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = {"": device}
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.eval()
    return model, tokenizer


def get_calibration_ids(tokenizer, seq_len, num_samples, device):
    """Load WikiText-2 calibration sequences for mask computation."""
    data = get_eval_dataset("wikitext2", tokenizer, seq_len, max_samples=num_samples)
    return torch.stack([ex["input_ids"] for ex in data[:num_samples]]).to(device)


# ---------------------------------------------------------------------------
# Sparsity setup
# ---------------------------------------------------------------------------

def setup_predictor(model, checkpoint_path, bottleneck_size, device):
    """Load predictor checkpoint into PredictorWrapper (patches model MLPs in-place).

    Auto-detects and loads compensation network if present in checkpoint.
    Returns the wrapper with gumbel_mask.hard=True for inference.
    """
    wrapper = PredictorWrapper(model, bottleneck_size=bottleneck_size)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    wrapper.predictors.load_state_dict(ckpt["predictors"])
    wrapper.predictors.to(device=device, dtype=torch.bfloat16)
    wrapper.predictors.eval()
    wrapper.gumbel_mask.hard = True

    if "comp_network" in ckpt:
        num_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size
        comp = CompensationNetwork(num_layers, hidden_size)
        comp.load_state_dict(ckpt["comp_network"])
        comp.to(device=device, dtype=torch.bfloat16)
        comp.eval()
        wrapper.comp_network = comp
        wrapper.use_compensation = True
        print("  Compensation network loaded from checkpoint")

    return wrapper


def compute_baseline_masks(model, baseline_mode, calib_ids, sparsity_target, svd_rank=256, allocation="global_topk"):
    """Compute static masks for a training-free baseline."""
    if baseline_mode in ("teal_vanilla", "wina", "rsparse"):
        avg_mag = _collect_swiglu_magnitudes(model, calib_ids)
    else:
        avg_mag = None

    if baseline_mode == "teal_vanilla":
        return vanilla_teal_global_masks(model, calib_ids, sparsity_target, avg_mag=avg_mag, allocation=allocation)
    elif baseline_mode == "teal_greedy":
        return teal_activation_magnitude_masks(model, calib_ids, sparsity_target)
    elif baseline_mode == "wina":
        return wina_global_masks(model, calib_ids, sparsity_target, avg_mag=avg_mag, allocation=allocation)
    elif baseline_mode == "wina_greedy":
        from baselines import wina_greedy_allocation_masks
        device = next(model.parameters()).device
        return wina_greedy_allocation_masks(model, calib_ids, sparsity_target, device=device)
    elif baseline_mode == "rsparse":
        return rsparse_global_masks(
            model, calib_ids, sparsity_target, svd_rank=svd_rank, avg_mag=avg_mag, allocation=allocation)
    else:
        raise ValueError(f"Unknown baseline: {baseline_mode}")


def apply_static_masks(model, masks, device):
    """Patch model MLPs with fixed binary masks. Returns saved forwards for restore."""
    saved = {}
    for layer_idx, layer in enumerate(model.model.layers):
        if layer_idx not in masks:
            continue
        saved[layer_idx] = layer.mlp.forward
        layer_device = layer.mlp.gate_proj.weight.device
        mask = masks[layer_idx].to(device=layer_device, dtype=layer.mlp.gate_proj.weight.dtype)
        mlp = layer.mlp

        def _make(gp, up, dp, af, m):
            def fwd(x):
                return dp(af(gp(x)) * up(x) * m.unsqueeze(0).unsqueeze(0))
            return fwd

        layer.mlp.forward = _make(
            mlp.gate_proj, mlp.up_proj, mlp.down_proj, mlp.act_fn, mask)
    return saved


def restore_forwards(model, saved):
    for idx, fwd in saved.items():
        model.model.layers[idx].mlp.forward = fwd


# ---------------------------------------------------------------------------
# Evaluation runners
# ---------------------------------------------------------------------------

def run_wikitext2_ppl(model, tokenizer, seq_len, device, max_samples=None):
    wt2 = get_eval_dataset("wikitext2", tokenizer, seq_len, max_samples=max_samples)
    print(f"  WikiText-2: {len(wt2)} sequences, seq_len={seq_len}")
    return evaluate_perplexity(model, wt2, device)


def run_c4_ppl(model, tokenizer, seq_len, device, max_samples=None):
    c4 = get_eval_dataset("c4", tokenizer, seq_len, max_samples=max_samples)
    print(f"  C4: {len(c4)} sequences, seq_len={seq_len}")
    return evaluate_perplexity(model, c4, device)


def run_lm_eval_tasks(model, tokenizer, tasks, batch_size, device):
    """Run lm-evaluation-harness tasks. Returns dict[task_name -> metric_dict]."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    lm_obj = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)

    # Group tasks by num_fewshot to minimize lm_eval invocations
    by_fewshot = {}
    for t in tasks:
        fs = LM_EVAL_DEFAULTS.get(t, {}).get("num_fewshot", 0)
        by_fewshot.setdefault(fs, []).append(t)

    all_results = {}
    for num_fewshot, task_group in by_fewshot.items():
        print(f"  Running {task_group} (num_fewshot={num_fewshot}) ...")
        out = lm_eval.simple_evaluate(
            model=lm_obj,
            tasks=task_group,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
        )
        for task_name in task_group:
            if task_name in out.get("results", {}):
                all_results[task_name] = out["results"][task_name]

    return all_results


def extract_key_metrics(lm_results):
    """Flatten lm-eval results into a concise {task: {metric: value}} dict."""
    summary = {}
    for task_name, metrics in lm_results.items():
        condensed = {}
        for k, v in metrics.items():
            if k.startswith("alias"):
                continue
            # lm-eval v0.4 uses "metric,filter" format
            short_key = k.split(",")[0] if "," in k else k
            if short_key.endswith("_stderr"):
                condensed[short_key] = v
            elif isinstance(v, (int, float)):
                condensed[short_key] = round(v, 4) if isinstance(v, float) else v
        summary[task_name] = condensed
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")

    if not args.dense and args.checkpoint is None and args.baseline_mode is None:
        print("Error: specify --checkpoint, --baseline_mode, or --dense")
        sys.exit(1)
    if args.checkpoint and args.baseline_mode:
        print("Error: --checkpoint and --baseline_mode are mutually exclusive")
        sys.exit(1)

    tasks = [t.strip() for t in args.tasks.split(",")]
    lm_eval_tasks = [t for t in tasks if t in LM_EVAL_DEFAULTS]
    do_wikitext2 = "wikitext2" in tasks
    do_c4 = "c4" in tasks

    unknown = [t for t in tasks if t not in ("wikitext2", "c4") and t not in LM_EVAL_DEFAULTS]
    if unknown:
        print(f"Warning: unknown tasks ignored: {unknown}")

    out_dir = Path(BASE_DIR) / "results" / "benchmark" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model ----
    print(f"Loading model: {args.model_name_or_path}")
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path, device, multi_gpu=args.multi_gpu)

    # ---- Set up sparsity mode ----
    wrapper = None
    saved_forwards = None
    mode_info = {}

    if args.dense:
        mode_info = {"mode": "dense"}
        print("Mode: dense (no sparsity)")

    elif args.baseline_mode:
        mode_info = {"mode": f"baseline_{args.baseline_mode}",
                     "sparsity_target": args.sparsity_target,
                     "allocation": args.allocation}
        print(f"Mode: baseline ({args.baseline_mode}), sparsity_target={args.sparsity_target}, allocation={args.allocation}")
        calib_ids = get_calibration_ids(
            tokenizer, args.seq_len, args.calibration_samples, device)
        print("Computing baseline masks ...")
        masks = compute_baseline_masks(
            model, args.baseline_mode, calib_ids, args.sparsity_target, args.svd_rank,
            allocation=args.allocation)
        _print_mask_stats(args.baseline_mode, masks)
        saved_forwards = apply_static_masks(model, masks, device)

    elif args.checkpoint:
        ckpt_path = args.checkpoint
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(BASE_DIR, ckpt_path)
        mode_info = {"mode": args.sparsity_mode, "checkpoint": ckpt_path,
                     "sparsity_target": args.sparsity_target,
                     "allocation": args.allocation}

        if args.sparsity_mode == "per_token":
            print(f"Mode: per-token hard mask")
            print(f"  checkpoint: {ckpt_path}")
            wrapper = setup_predictor(model, ckpt_path, args.bottleneck_size, device)
            # MLPs are now patched — predictor runs on every forward pass

        elif args.sparsity_mode == "teal_global":
            print(f"Mode: TEAL global mask from predictor (allocation={args.allocation})")
            print(f"  checkpoint: {ckpt_path}")
            calib_ids = get_calibration_ids(
                tokenizer, args.seq_len, args.calibration_samples, device)
            wrapper = setup_predictor(model, ckpt_path, args.bottleneck_size, device)
            print("Computing TEAL global masks from predictor ...")
            masks = teal_global_masks(wrapper, calib_ids, args.sparsity_target, allocation=args.allocation)
            _print_mask_stats("TEAL-predictor", masks)
            # Switch from per-token to static masks
            wrapper.sparse_mode = False
            saved_forwards = apply_static_masks(model, masks, device)

    # ---- Run evaluations ----
    results = {
        "exp_id": args.exp_id,
        "config": {k: v for k, v in vars(args).items()},
        "mode": mode_info,
        "metrics": {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if do_wikitext2:
        print("\n--- WikiText-2 PPL ---")
        t0 = time.time()
        ppl = run_wikitext2_ppl(model, tokenizer, args.seq_len, device, args.max_eval_samples)
        elapsed = time.time() - t0
        print(f"  PPL = {ppl:.4f}  ({elapsed:.1f}s)")
        results["metrics"]["wikitext2_ppl"] = round(ppl, 4)

    if do_c4:
        print("\n--- C4 PPL ---")
        t0 = time.time()
        ppl = run_c4_ppl(model, tokenizer, args.seq_len, device, args.max_eval_samples)
        elapsed = time.time() - t0
        print(f"  PPL = {ppl:.4f}  ({elapsed:.1f}s)")
        results["metrics"]["c4_ppl"] = round(ppl, 4)

    if lm_eval_tasks:
        print(f"\n--- lm-eval: {lm_eval_tasks} ---")
        t0 = time.time()
        raw = run_lm_eval_tasks(model, tokenizer, lm_eval_tasks, args.batch_size, device)
        elapsed = time.time() - t0
        summary = extract_key_metrics(raw)
        results["metrics"]["lm_eval"] = summary
        results["metrics"]["lm_eval_raw"] = raw
        print(f"  Completed in {elapsed:.1f}s")
        for task_name, m in summary.items():
            line = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                            for k, v in m.items() if not k.endswith("_stderr"))
            print(f"  {task_name}: {line}")

    # ---- Cleanup & save ----
    if saved_forwards:
        restore_forwards(model, saved_forwards)

    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

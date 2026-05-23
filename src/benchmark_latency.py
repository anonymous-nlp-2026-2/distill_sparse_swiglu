"""Inference latency benchmark: dense vs KL-sparse (mask-and-multiply).

Measures wall-clock tokens/sec, ms/token, GPU memory, and predictor overhead
for dense and predictor-masked forward passes. No sparse kernel is used;
sparsity is applied as element-wise mask on the intermediate activation
(mask-and-multiply), so no actual FLOP reduction occurs in down_proj.

Usage:
  python src/benchmark_latency.py --help
  python src/benchmark_latency.py --device cuda:0 --batch_sizes 1,8 --runs 3
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predictor import SparsityPredictor, GumbelSigmoidMask
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Defaults ──────────────────────────────────────────────────────────────
DEFAULT_MODEL = "/root/autodl-tmp/models/llama-3.1-8b"
BASE_DIR = "/root/distill_sparse_swiglu"
CKPT_30 = os.path.join(BASE_DIR, "checkpoints/kl_sparsity30_s42_v3/predictor_kl_normalized.pt")
CKPT_50 = os.path.join(BASE_DIR, "checkpoints/mvp_kl_norm_v2_s42/predictor_kl_normalized.pt")

HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 14336
NUM_LAYERS = 32
BOTTLENECK_SIZE = 128


def parse_args():
    p = argparse.ArgumentParser(
        description="Inference latency benchmark: dense vs KL-sparse (mask-and-multiply)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model path")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--batch_sizes", type=str, default="1,8",
                   help="Comma-separated batch sizes")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100, help="Timing iterations per measurement")
    p.add_argument("--runs", type=int, default=3, help="Independent runs for mean/std")
    p.add_argument("--ckpt_30", type=str, default=CKPT_30)
    p.add_argument("--ckpt_50", type=str, default=CKPT_50)
    p.add_argument("--output", type=str, default=None,
                   help="JSON output path (default: results/benchmark_latency.json)")
    p.add_argument("--dry_run", action="store_true", help="Just validate setup, no timing")
    return p.parse_args()


# ── Timing ────────────────────────────────────────────────────────────────
def measure_latency(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def gpu_memory_mb(device):
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


# ── Model loading ─────────────────────────────────────────────────────────
def load_model(model_path, device):
    import importlib.util
    attn = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        attn_implementation=attn,
        device_map={"": device},
    )
    model.eval()
    return model


def load_predictors(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    predictors = torch.nn.ModuleList([
        SparsityPredictor(HIDDEN_SIZE, INTERMEDIATE_SIZE, BOTTLENECK_SIZE)
        for _ in range(NUM_LAYERS)
    ])
    predictors.load_state_dict(ckpt["predictors"])
    predictors = predictors.half().to(device).eval()
    return predictors


# ── MLP patching ──────────────────────────────────────────────────────────
def patch_mlps_sparse(model, predictors):
    hooks = []
    for layer_idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        gp, up, dp, af = mlp.gate_proj, mlp.up_proj, mlp.down_proj, mlp.act_fn
        pred = predictors[layer_idx]
        original_fwd = mlp.forward

        def make_fwd(_gp, _up, _dp, _af, _pred):
            def fwd(x):
                gate = _af(_gp(x))
                up_out = _up(x)
                intermediate = gate * up_out
                mask = (_pred(x) > 0).to(x.dtype)
                return _dp(intermediate * mask)
            return fwd

        mlp.forward = make_fwd(gp, up, dp, af, pred)
        hooks.append((mlp, original_fwd))
    return hooks


def restore_mlps(hooks):
    for mlp, orig in hooks:
        mlp.forward = orig


# ── Predictor-only overhead ───────────────────────────────────────────────
def measure_predictor_overhead(predictors, device, batch_size, seq_len, warmup, iters):
    x = torch.randn(batch_size, seq_len, HIDDEN_SIZE, dtype=torch.float16, device=device)

    def pred_fwd():
        for p in predictors:
            _ = p(x)

    return measure_latency(pred_fwd, warmup, iters)


# ── Single condition benchmark ────────────────────────────────────────────
def bench_condition(model, input_ids, warmup, iters):
    @torch.no_grad()
    def fwd():
        _ = model(input_ids)
    return measure_latency(fwd, warmup, iters)


# ── Main ──────────────────────────────────────────────────────────────────
def run_benchmark(args):
    device = torch.device(args.device)

    # Ensure CUDA is initialized before calling memory APIs
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    print(f"Loading model from {args.model} ...")
    torch.cuda.reset_peak_memory_stats(device)
    model = load_model(args.model, device)
    model_mem = gpu_memory_mb(device)
    print(f"  Model memory: {model_mem:.0f} MB")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Loading predictors ...")
    pred_30 = load_predictors(args.ckpt_30, device)
    pred_50 = load_predictors(args.ckpt_50, device)
    print(f"  Predictor params per layer: {sum(p.numel() for p in pred_30[0].parameters()):,}")
    print(f"  Total predictor params (32 layers): {sum(p.numel() for p in pred_30.parameters()):,}")

    if args.dry_run:
        print("\n[dry_run] Setup validated. Exiting.")
        return

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    all_results = {}

    conditions = [
        ("dense", None),
        ("kl_sparse_30pct", pred_30),
        ("kl_sparse_50pct", pred_50),
    ]

    for bs in batch_sizes:
        print(f"\n{'='*70}")
        print(f"Batch size = {bs}, seq_len = {args.seq_len}")
        print(f"{'='*70}")

        input_ids = torch.randint(
            0, tokenizer.vocab_size, (bs, args.seq_len), device=device
        )
        tokens = bs * args.seq_len

        bs_results = {}

        for cond_name, preds in conditions:
            hooks = None
            if preds is not None:
                hooks = patch_mlps_sparse(model, preds)

            torch.cuda.reset_peak_memory_stats(device)

            run_tps_list = []
            run_ms_list = []
            for run_i in range(args.runs):
                times = bench_condition(model, input_ids, args.warmup, args.iters)
                mean_t = np.mean(times)
                tps = tokens / mean_t
                ms_per_tok = mean_t * 1000 / tokens
                run_tps_list.append(tps)
                run_ms_list.append(ms_per_tok)

            peak_mem = gpu_memory_mb(device)

            if hooks is not None:
                restore_mlps(hooks)

            tps_mean = np.mean(run_tps_list)
            tps_std = np.std(run_tps_list)
            ms_mean = np.mean(run_ms_list)
            ms_std = np.std(run_ms_list)

            bs_results[cond_name] = {
                "tokens_per_sec_mean": round(float(tps_mean), 1),
                "tokens_per_sec_std": round(float(tps_std), 1),
                "ms_per_token_mean": round(float(ms_mean), 4),
                "ms_per_token_std": round(float(ms_std), 4),
                "peak_memory_mb": round(float(peak_mem), 0),
                "runs": args.runs,
                "iters_per_run": args.iters,
                "warmup": args.warmup,
            }

            print(f"  {cond_name}: {tps_mean:.1f} +/- {tps_std:.1f} tok/s, "
                  f"{ms_mean:.4f} +/- {ms_std:.4f} ms/tok, "
                  f"mem={peak_mem:.0f} MB")

        # Predictor-only overhead
        pred_times_30 = measure_predictor_overhead(
            pred_30, device, bs, args.seq_len, args.warmup, args.iters)
        pred_times_50 = measure_predictor_overhead(
            pred_50, device, bs, args.seq_len, args.warmup, args.iters)
        bs_results["predictor_overhead_30pct_ms"] = round(float(np.mean(pred_times_30)) * 1000, 2)
        bs_results["predictor_overhead_50pct_ms"] = round(float(np.mean(pred_times_50)) * 1000, 2)

        all_results[f"batch_{bs}"] = bs_results

    # ── Compute speedup vs dense ──────────────────────────────────────
    for bs_key, bs_res in all_results.items():
        dense_tps = bs_res["dense"]["tokens_per_sec_mean"]
        for cond_name in ["kl_sparse_30pct", "kl_sparse_50pct"]:
            if cond_name in bs_res:
                cond_tps = bs_res[cond_name]["tokens_per_sec_mean"]
                bs_res[cond_name]["speedup_vs_dense"] = round(cond_tps / dense_tps, 4)

    # ── Output JSON ───────────────────────────────────────────────────
    output = {
        "benchmark": "inference_latency",
        "note": "mask-and-multiply (no sparse kernel). Sparsity mask is applied "
                "element-wise on intermediate activation; gate_proj, up_proj, and "
                "down_proj all execute as full dense matmuls. No FLOP savings.",
        "model": args.model,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "seq_len": args.seq_len,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoints": {
            "30pct": args.ckpt_30,
            "50pct": args.ckpt_50,
        },
        "results": all_results,
    }

    out_path = args.output or os.path.join(BASE_DIR, "results/benchmark_latency.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nJSON saved to {out_path}")

    # ── Markdown table ────────────────────────────────────────────────
    print("\n## Inference Latency Benchmark")
    print(f"*mask-and-multiply, no sparse kernel* | seq_len={args.seq_len} | "
          f"{args.runs} runs x {args.iters} iters | "
          f"GPU: {torch.cuda.get_device_name(device)}\n")

    for bs_key, bs_res in all_results.items():
        bs_num = bs_key.split("_")[1]
        print(f"### batch_size={bs_num}\n")
        print("| Condition | tokens/s | ms/token | Mem (MB) | vs Dense |")
        print("|-----------|----------|----------|----------|----------|")

        for cond_name in ["dense", "kl_sparse_30pct", "kl_sparse_50pct"]:
            r = bs_res[cond_name]
            speedup = r.get("speedup_vs_dense", 1.0)
            label = {"dense": "Dense", "kl_sparse_30pct": "KL Sparse 30%",
                     "kl_sparse_50pct": "KL Sparse 50%"}[cond_name]
            print(f"| {label} | {r['tokens_per_sec_mean']:.1f} +/- {r['tokens_per_sec_std']:.1f} "
                  f"| {r['ms_per_token_mean']:.4f} +/- {r['ms_per_token_std']:.4f} "
                  f"| {r['peak_memory_mb']:.0f} | {speedup:.4f}x |")

        print(f"\nPredictor overhead (32 layers, isolated): "
              f"30%={bs_res['predictor_overhead_30pct_ms']:.2f} ms, "
              f"50%={bs_res['predictor_overhead_50pct_ms']:.2f} ms\n")


def main():
    args = parse_args()
    with torch.no_grad():
        run_benchmark(args)


if __name__ == "__main__":
    main()

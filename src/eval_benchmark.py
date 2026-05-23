"""lm-eval-harness + PPL evaluation for all sparsity conditions (plan_007).

Produces the main results table: PPL on WikiText-2/C4 + downstream accuracy
(ARC-c, WinoGrande, MMLU, GSM8K) under different sparsity methods at 50%
SwiGLU sparsity on Llama-3.1-8B.

Conditions: dense, KL, BCE, BCE+comp, STE, SPON, TEAL, WINA, R-Sparse.

Usage:
    python src/eval_benchmark.py --condition dense --tasks arc_challenge
    python src/eval_benchmark.py --condition kl_s42 --tasks all --gpu 0
    python src/eval_benchmark.py --condition all --tasks ppl --gpu 0
    python src/eval_benchmark.py --list-conditions
"""

import argparse
import gc
import importlib.util
import json
import os
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/.hf_cache")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import sys
import time
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predictor import SparsityPredictor, CompensationNetwork
from spon import SPONBiasVectors
from data_utils import get_eval_dataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = "/root/distill_sparse_swiglu"
MODEL_PATH = "/root/autodl-tmp/models/llama-3.1-8b"
SEQ_LEN = 2048
SPARSITY_TARGET = 0.5
CALIB_SAMPLES = 32
RESULTS_DIR = os.path.join(BASE_DIR, "results", "plan_007")

_CKPT = lambda name: os.path.join(BASE_DIR, "checkpoints", name)

CONDITIONS = OrderedDict([
    ("dense", {
        "mode": "dense",
    }),
    ("kl_s42", {
        "mode": "predictor",
        "checkpoint": _CKPT("mvp_kl_norm_v2_s42/predictor_kl_normalized.pt"),
    }),
    ("bce_s42", {
        "mode": "predictor",
        "checkpoint": _CKPT("mvp_bce_s42/predictor_bce.pt"),
    }),
    ("bce_comp_s42", {
        "mode": "predictor_comp",
        "checkpoint": _CKPT("bce_comp_staged_s42/predictor_bce_comp.pt"),
    }),
    ("ste_s42", {
        "mode": "predictor",
        "checkpoint": _CKPT("ste_baseline_s42/predictor_kl_normalized.pt"),
    }),
    ("spon_s42", {
        "mode": "spon",
        "predictor_checkpoint": _CKPT("mvp_kl_norm_v2_s42/predictor_kl_normalized.pt"),
        "spon_checkpoint": _CKPT("spon_2048_s42/spon_biases.pt"),
    }),
    ("teal_vanilla", {
        "mode": "static_mask",
        "method": "teal",
    }),
    ("wina", {
        "mode": "static_mask",
        "method": "wina",
    }),
    ("rsparse", {
        "mode": "static_mask",
        "method": "rsparse",
    }),
    ("uniform_50", {
        "mode": "static_mask",
        "method": "uniform",
    }),
    ("mistral_dense", {
        "mode": "dense",
        "model_path": "/root/autodl-tmp/models/mistral-7b",
    }),
    ("mistral_kl_s42", {
        "mode": "predictor",
        "checkpoint": _CKPT("mistral_kl_s42/predictor_kl_normalized.pt"),
        "model_path": "/root/autodl-tmp/models/mistral-7b",
    }),
])

# lm-eval task -> standard num_fewshot
LM_EVAL_TASKS = OrderedDict([
    ("arc_challenge", 25),
    ("winogrande", 5),
    ("mmlu", 5),
    ("gsm8k", 5),
])

PPL_DATASETS = ["wikitext2", "c4"]


def _best_attn_impl():
    if importlib.util.find_spec("flash_attn"):
        return "flash_attention_2"
    return "sdpa"


# ---------------------------------------------------------------------------
# PPL computation (project-native, consistent with existing evaluate.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_ppl(model, dataset, device):
    total_loss = 0.0
    total_tokens = 0
    for sample in dataset:
        ids = sample["input_ids"].unsqueeze(0).to(device)
        labels = sample["labels"].unsqueeze(0).to(device)
        logits = model(input_ids=ids).logits
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += labels.numel()
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()


# ---------------------------------------------------------------------------
# Model patching
# ---------------------------------------------------------------------------

def unpatch_model(model):
    """Remove instance-level MLP forward overrides, restoring class defaults."""
    for layer in model.model.layers:
        if "forward" in layer.mlp.__dict__:
            del layer.mlp.__dict__["forward"]


def _load_predictors(config, checkpoint_path, device):
    predictors = nn.ModuleList([
        SparsityPredictor(config.hidden_size, config.intermediate_size, 128)
        for _ in range(config.num_hidden_layers)
    ])
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    predictors.load_state_dict(ckpt["predictors"])
    predictors.to(device=device, dtype=torch.bfloat16).eval()
    return predictors, ckpt


def patch_predictor(model, checkpoint_path, device):
    predictors, _ = _load_predictors(model.config, checkpoint_path, device)
    for idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        gp, up, dp, af = mlp.gate_proj, mlp.up_proj, mlp.down_proj, mlp.act_fn
        pred = predictors[idx]

        def fwd(x, _gp=gp, _up=up, _dp=dp, _af=af, _pred=pred):
            inter = _af(_gp(x)) * _up(x)
            with torch.no_grad():
                mask = (_pred(x) > 0).to(inter.dtype)
            return _dp(inter * mask)

        mlp.forward = fwd
    return {"predictors": predictors}


def patch_predictor_comp(model, checkpoint_path, device):
    cfg = model.config
    predictors, ckpt = _load_predictors(cfg, checkpoint_path, device)
    comp_net = CompensationNetwork(cfg.num_hidden_layers, cfg.hidden_size, 256)
    comp_net.load_state_dict(ckpt["comp_network"])
    comp_net.to(device=device, dtype=torch.bfloat16).eval()

    for idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        gp, up, dp, af = mlp.gate_proj, mlp.up_proj, mlp.down_proj, mlp.act_fn
        pred, comp = predictors[idx], comp_net.heads[idx]

        def fwd(x, _gp=gp, _up=up, _dp=dp, _af=af, _pred=pred, _comp=comp):
            inter = _af(_gp(x)) * _up(x)
            with torch.no_grad():
                mask = (_pred(x) > 0).to(inter.dtype)
            return _dp(inter * mask) + _comp(x)

        mlp.forward = fwd
    return {"predictors": predictors, "comp_net": comp_net}


def patch_spon(model, predictor_ckpt_path, spon_ckpt_path, device):
    cfg = model.config
    predictors, _ = _load_predictors(cfg, predictor_ckpt_path, device)

    spon = SPONBiasVectors(cfg.num_hidden_layers, cfg.intermediate_size)
    spon_ckpt = torch.load(spon_ckpt_path, map_location=device, weights_only=True)
    spon.load_state_dict(spon_ckpt["spon_biases"])
    spon.to(device=device, dtype=torch.bfloat16).eval()

    for idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        gp, up, dp, af = mlp.gate_proj, mlp.up_proj, mlp.down_proj, mlp.act_fn
        pred, bias = predictors[idx], spon.biases[idx]

        def fwd(x, _gp=gp, _up=up, _dp=dp, _af=af, _pred=pred, _bias=bias):
            inter = _af(_gp(x)) * _up(x)
            with torch.no_grad():
                mask = (_pred(x) > 0).to(inter.dtype)
            return _dp(inter * mask + _bias)

        mlp.forward = fwd
    return {"predictors": predictors, "spon": spon}


def patch_static_masks(model, masks, device):
    for idx, layer in enumerate(model.model.layers):
        if idx not in masks:
            continue
        mask = masks[idx].to(device=device, dtype=torch.bfloat16).unsqueeze(0).unsqueeze(0)
        mlp = layer.mlp
        gp, up, dp, af = mlp.gate_proj, mlp.up_proj, mlp.down_proj, mlp.act_fn

        def fwd(x, _gp=gp, _up=up, _dp=dp, _af=af, _m=mask):
            return _dp(_af(_gp(x)) * _up(x) * _m)

        mlp.forward = fwd


def compute_static_masks(model, calib_ids, method, sparsity_target):
    """Compute static masks. Model must be unpatched (dense forwards).

    Uses global top-k with per-layer mean normalization for TEAL and WINA.
    """
    if method == "teal":
        from evaluate import vanilla_teal_global_masks
        return vanilla_teal_global_masks(
            model, calib_ids, sparsity_target)
    elif method == "wina":
        from evaluate import wina_global_masks
        return wina_global_masks(
            model, calib_ids, sparsity_target)
    elif method == "rsparse":
        from evaluate import rsparse_global_masks, _collect_swiglu_magnitudes
        avg_mag = _collect_swiglu_magnitudes(model, calib_ids)
        return rsparse_global_masks(
            model, calib_ids, sparsity_target, avg_mag=avg_mag)
    elif method == "uniform":
        from evaluate import _collect_swiglu_magnitudes
        avg_mag = _collect_swiglu_magnitudes(model, calib_ids)
        n_layers = len(model.model.layers)
        inter_dim = model.model.layers[0].mlp.gate_proj.out_features
        masks = {}
        for li in range(n_layers):
            mag = avg_mag[li].float()
            nk = max(int(inter_dim * (1.0 - sparsity_target)), 1)
            thr = torch.topk(mag, nk).values[-1]
            masks[li] = (mag >= thr).to(torch.bfloat16)
        return masks
    raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# lm-eval integration
# ---------------------------------------------------------------------------

def run_lm_eval(model, tokenizer, tasks, device, batch_size=4):
    """Run lm-eval-harness tasks on the (possibly patched) model."""
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        print("  WARNING: lm-eval not installed, skipping downstream tasks")
        return {t: {"error": "lm-eval not installed"} for t in tasks}

    lm_obj = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)

    results = {}
    for task_name in tasks:
        num_fewshot = LM_EVAL_TASKS.get(task_name, 0)
        print(f"  {task_name} ({num_fewshot}-shot) ...", end=" ", flush=True)
        t0 = time.time()
        try:
            out = lm_eval.simple_evaluate(
                model=lm_obj,
                tasks=[task_name],
                num_fewshot=num_fewshot,
            )
            task_res = out["results"].get(task_name, {})
            elapsed = time.time() - t0
            metric_key = _primary_metric(task_name)
            value = task_res.get(metric_key)
            results[task_name] = {
                "metric": metric_key,
                "value": value,
                "full": {k: v for k, v in task_res.items()
                         if not k.endswith(",stderr")},
                "elapsed_s": round(elapsed, 1),
            }
            val_str = f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
            print(f"{val_str} ({elapsed:.0f}s)")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"ERROR ({elapsed:.0f}s): {e}")
            results[task_name] = {"error": str(e)}

    return results


def _primary_metric(task_name):
    return {
        "arc_challenge": "acc_norm,none",
        "winogrande": "acc,none",
        "mmlu": "acc,none",
        "gsm8k": "exact_match,strict-match",
    }.get(task_name, "acc,none")


# ---------------------------------------------------------------------------
# Results output
# ---------------------------------------------------------------------------

def save_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    summary = {}
    for cond, res in results.items():
        summary[cond] = {k: v for k, v in res.items() if not k.endswith("_full")}

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    with open(os.path.join(output_dir, "results_full.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    _write_markdown_table(summary, output_dir)


def _write_markdown_table(results, output_dir):
    if not results:
        return

    ppl_cols = []
    lm_cols = []
    for res in results.values():
        for k in res:
            if k.endswith("_ppl") and k not in ppl_cols:
                ppl_cols.append(k)
            elif k in LM_EVAL_TASKS and k not in lm_cols:
                lm_cols.append(k)
    metric_cols = ppl_cols + lm_cols

    lines = [
        "# Plan 007: Benchmark Results",
        f"Llama-3.1-8B, {SPARSITY_TARGET*100:.0f}% SwiGLU Sparsity",
        "",
    ]
    header = ["Condition"] + metric_cols
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")

    for cond, res in results.items():
        row = [cond]
        for col in metric_cols:
            val = res.get(col)
            if isinstance(val, float):
                fmt = f"{val:.2f}" if col.endswith("_ppl") else f"{val:.4f}"
                row.append(fmt)
            elif val is not None:
                row.append(str(val)[:20])
            else:
                row.append("-")
        lines.append("| " + " | ".join(row) + " |")

    with open(os.path.join(output_dir, "results_table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def _print_table(results, ppl_datasets, lm_eval_tasks):
    cols = [f"{d}_ppl" for d in ppl_datasets] + lm_eval_tasks
    if not cols:
        return

    hdr = f"{'Condition':<18}"
    for c in cols:
        hdr += f"  {c:>14}"
    print(hdr)
    print("-" * len(hdr))

    for cond_name, res in results.items():
        if "error" in res:
            print(f"{cond_name:<18}  ERROR: {res['error']}")
            continue
        row = f"{cond_name:<18}"
        for c in cols:
            val = res.get(c)
            if isinstance(val, float):
                row += f"  {val:>14.4f}"
            elif val is not None:
                row += f"  {str(val):>14}"
            else:
                row += f"  {'-':>14}"
        print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark evaluation for sparsity conditions (plan_007)")
    p.add_argument("--condition", type=str, default="all",
                   help="Condition name(s), comma-separated, or 'all'")
    p.add_argument("--tasks", type=str, default="all",
                   help="'ppl', 'lm_eval', task names (comma-sep), or 'all'")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=4,
                   help="Batch size for lm-eval tasks")
    p.add_argument("--max_ppl_samples", type=int, default=0,
                   help="Max sequences for PPL evaluation")
    p.add_argument("--skip_ppl", action="store_true")
    p.add_argument("--skip_lm_eval", action="store_true")
    p.add_argument("--list_conditions", action="store_true")
    p.add_argument("--sparsity_target", type=float, default=SPARSITY_TARGET)
    p.add_argument("--model_path", type=str, default=None,
                   help="Override model path (auto-detected from condition)")
    p.add_argument("--output_dir", type=str, default=RESULTS_DIR)
    return p.parse_args()


def main():
    args = parse_args()

    if args.list_conditions:
        for name, cfg in CONDITIONS.items():
            ckpt = cfg.get("checkpoint", cfg.get("predictor_checkpoint", "-"))
            print(f"  {name:<18} {cfg['mode']:<16} {ckpt}")
        return

    device = torch.device(f"cuda:{args.gpu}")
    os.makedirs(args.output_dir, exist_ok=True)

    # -- Resolve conditions --
    if args.condition == "all":
        cond_list = list(CONDITIONS.keys())
    else:
        cond_list = [c.strip() for c in args.condition.split(",")]
        for c in cond_list:
            if c not in CONDITIONS:
                sys.exit(f"Unknown condition '{c}'. Use --list-conditions.")

    # -- Resolve tasks --
    ppl_datasets, lm_eval_tasks = [], []
    if args.tasks == "all":
        if not args.skip_ppl:
            ppl_datasets = list(PPL_DATASETS)
        if not args.skip_lm_eval:
            lm_eval_tasks = list(LM_EVAL_TASKS.keys())
    elif args.tasks == "ppl":
        ppl_datasets = list(PPL_DATASETS)
    elif args.tasks == "lm_eval":
        lm_eval_tasks = list(LM_EVAL_TASKS.keys())
    else:
        for t in args.tasks.split(","):
            t = t.strip()
            if t in ("wikitext2", "c4"):
                ppl_datasets.append(t)
            elif t in LM_EVAL_TASKS:
                lm_eval_tasks.append(t)
            else:
                sys.exit(f"Unknown task '{t}'. PPL: {PPL_DATASETS}. "
                         f"lm-eval: {list(LM_EVAL_TASKS.keys())}")

    print(f"Conditions: {cond_list}")
    print(f"PPL: {ppl_datasets or '(skip)'}")
    print(f"lm-eval: {lm_eval_tasks or '(skip)'}")
    print(f"Device: {device}")
    print()

    # -- Resolve model path --
    model_path = args.model_path
    if model_path is None:
        cond_paths = set(CONDITIONS[c].get("model_path", MODEL_PATH) for c in cond_list)
        if len(cond_paths) > 1:
            sys.exit(f"Mixed model paths in conditions: {cond_paths}. "
                     f"Use --model_path or run conditions separately.")
        model_path = cond_paths.pop()

    # -- Load model --
    print(f"Loading model from {model_path} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation=_best_attn_impl(),
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"  Loaded in {time.time()-t0:.0f}s")

    # -- Load PPL datasets --
    ppl_data = {}
    for ds_name in ppl_datasets:
        print(f"Loading {ds_name} ...")
        try:
            ppl_data[ds_name] = get_eval_dataset(
                ds_name, tokenizer, SEQ_LEN, args.max_ppl_samples)
            print(f"  {len(ppl_data[ds_name])} sequences")
        except Exception as e:
            print(f"  ERROR loading {ds_name}: {e}")

    # -- Pre-compute static masks (needs unpatched model) --
    static_masks_cache = {}
    needs_calib = any(
        CONDITIONS[c]["mode"] == "static_mask" for c in cond_list)

    if needs_calib:
        if "wikitext2" in ppl_data:
            calib_src = ppl_data["wikitext2"]
        else:
            print("Loading WikiText-2 for calibration ...")
            try:
                calib_src = get_eval_dataset(
                    "wikitext2", tokenizer, SEQ_LEN, CALIB_SAMPLES)
            except Exception as e:
                print(f"  ERROR: {e}")
                calib_src = None

        if calib_src:
            calib_ids = torch.stack([
                calib_src[i]["input_ids"]
                for i in range(min(CALIB_SAMPLES, len(calib_src)))
            ]).to(device)

            for c in cond_list:
                cfg = CONDITIONS[c]
                if cfg["mode"] != "static_mask":
                    continue
                method = cfg["method"]
                if method in static_masks_cache:
                    continue
                print(f"Computing {method} masks ...")
                t0 = time.time()
                try:
                    static_masks_cache[method] = compute_static_masks(
                        model, calib_ids, method, args.sparsity_target)
                    masks = static_masks_cache[method]
                    n_total = sum(m.numel() for m in masks.values())
                    n_zero = sum((m == 0).sum().item() for m in masks.values())
                    print(f"  Done ({time.time()-t0:.0f}s, "
                          f"sparsity={n_zero/n_total:.3f})")
                except Exception as e:
                    print(f"  ERROR: {e}")

    # -- Main evaluation loop --
    all_results = {}

    for cond_name in cond_list:
        cfg = CONDITIONS[cond_name]
        print(f"\n{'='*60}")
        print(f"  {cond_name} ({cfg['mode']})")
        print(f"{'='*60}")

        unpatch_model(model)
        refs = {}

        # Verify checkpoint exists
        ckpt_paths = [cfg.get("checkpoint"), cfg.get("predictor_checkpoint"),
                      cfg.get("spon_checkpoint")]
        missing = [p for p in ckpt_paths if p and not os.path.exists(p)]
        if missing:
            print(f"  SKIP: checkpoint missing: {missing}")
            all_results[cond_name] = {"condition": cond_name,
                                       "error": f"missing: {missing}"}
            continue

        # Apply patches
        mode = cfg["mode"]
        try:
            if mode == "dense":
                pass
            elif mode == "predictor":
                refs = patch_predictor(model, cfg["checkpoint"], device)
            elif mode == "predictor_comp":
                refs = patch_predictor_comp(model, cfg["checkpoint"], device)
            elif mode == "spon":
                refs = patch_spon(
                    model, cfg["predictor_checkpoint"],
                    cfg["spon_checkpoint"], device)
            elif mode == "static_mask":
                masks = static_masks_cache.get(cfg["method"])
                if masks is None:
                    print("  SKIP: masks not computed")
                    all_results[cond_name] = {
                        "condition": cond_name, "error": "no masks"}
                    continue
                patch_static_masks(model, masks, device)
        except Exception as e:
            print(f"  ERROR patching: {e}")
            import traceback; traceback.print_exc()
            all_results[cond_name] = {"condition": cond_name,
                                       "error": str(e)}
            unpatch_model(model)
            continue

        result = {"condition": cond_name, "mode": mode}
        t_cond = time.time()

        # PPL evaluation
        for ds_name, dataset in ppl_data.items():
            print(f"  {ds_name} PPL ...", end=" ", flush=True)
            t0 = time.time()
            try:
                ppl = compute_ppl(model, dataset, device)
                print(f"{ppl:.4f} ({time.time()-t0:.0f}s)")
                result[f"{ds_name}_ppl"] = round(ppl, 4)
            except Exception as e:
                print(f"ERROR: {e}")
                result[f"{ds_name}_ppl"] = f"ERR:{e}"

        # lm-eval tasks
        if lm_eval_tasks:
            lm_res = run_lm_eval(
                model, tokenizer, lm_eval_tasks, device, args.batch_size)
            for task_name, task_res in lm_res.items():
                if "value" in task_res and task_res["value"] is not None:
                    result[task_name] = task_res["value"]
                elif "error" in task_res:
                    result[task_name] = f"ERR:{task_res['error'][:60]}"
                result[f"{task_name}_full"] = task_res

        result["elapsed_s"] = round(time.time() - t_cond, 1)
        all_results[cond_name] = result

        # Cleanup
        unpatch_model(model)
        del refs
        gc.collect()
        torch.cuda.empty_cache()

        # Incremental save
        save_results(all_results, args.output_dir)

    # -- Final summary --
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    _print_table(all_results, ppl_datasets, lm_eval_tasks)
    save_results(all_results, args.output_dir)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()

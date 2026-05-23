"""Patch evaluate.py and batch_eval.py to add vanilla TEAL baseline."""
import re

# ========== PATCH evaluate.py ==========
with open("/root/distill_sparse_swiglu/src/evaluate.py", "r") as f:
    code = f.read()

# 1. Add vanilla_teal_global_masks after teal_global_masks
vanilla_func = '''

@torch.no_grad()
def vanilla_teal_global_masks(model, input_ids, sparsity_target, calib_batch_size=4):
    """Activation-magnitude-based TEAL baseline (no predictor).

    Ranks neurons by mean |activation| of SwiGLU intermediates on calibration data,
    then applies global top-K pruning across all layers.
    """
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
                mag = intermediate.abs().mean(dim=(0, 1))
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

    all_layer_scores = []
    for layer_idx in sorted(accum_magnitudes):
        all_layer_scores.append((layer_idx, accum_magnitudes[layer_idx] / n_batches))

    all_concat = torch.cat([s for _, s in all_layer_scores])
    num_keep = int(len(all_concat) * (1.0 - sparsity_target))
    threshold = torch.topk(all_concat, num_keep).values[-1]

    masks = {}
    for layer_idx, scores in all_layer_scores:
        masks[layer_idx] = (scores >= threshold).to(scores.dtype)
    return masks

'''

# Insert before @torch.no_grad()\ndef evaluate_perplexity
insertion_point = "\n@torch.no_grad()\ndef evaluate_perplexity"
if "vanilla_teal_global_masks" not in code:
    code = code.replace(insertion_point, vanilla_func + insertion_point)
    print("  Added vanilla_teal_global_masks function")
else:
    print("  vanilla_teal_global_masks already exists, skipping")

# 2. Add [4/4] Vanilla TEAL to main()
vanilla_main_block = '''
    # [4/4] Vanilla TEAL Sparse (activation magnitude baseline)
    print("\\n[4/4] Vanilla TEAL Sparse (WikiText-2) ...")
    vanilla_masks = vanilla_teal_global_masks(model, calib_ids, args.sparsity_target)
    vteal_total_n = sum(m.numel() for m in vanilla_masks.values())
    vteal_total_z = sum((m == 0).sum().item() for m in vanilla_masks.values())
    vteal_sparsity = vteal_total_z / vteal_total_n
    print(f"  Vanilla TEAL model-level sparsity: {vteal_sparsity:.3f}")
    vanilla_teal_ppl = evaluate_perplexity(model, wt2, device, vanilla_masks)
    print(f"  Perplexity: {vanilla_teal_ppl:.2f}")

'''

summary_marker = '    print(f"\\n=== Summary'
if "Vanilla TEAL Sparse" not in code:
    code = code.replace(summary_marker, vanilla_main_block + summary_marker)
    print("  Added [4/4] vanilla TEAL eval to main()")
else:
    print("  [4/4] already exists, skipping")

# 3. Update summary print to include vanilla TEAL
old_summary = 'print(f"WikiText-2:  true_dense={true_dense_ppl:.2f}  predictor_hard_mask={predictor_ppl:.2f}  teal_sparse={teal_ppl:.2f}")'
new_summary = 'print(f"WikiText-2:  true_dense={true_dense_ppl:.2f}  predictor_hard_mask={predictor_ppl:.2f}  teal_sparse={teal_ppl:.2f}  vanilla_teal={vanilla_teal_ppl:.2f}")'
if "vanilla_teal_ppl" not in code.split("Summary")[1] if "Summary" in code else "":
    code = code.replace(old_summary, new_summary)
    print("  Updated summary line")

with open("/root/distill_sparse_swiglu/src/evaluate.py", "w") as f:
    f.write(code)
print("evaluate.py patched successfully")

# Verify
import hashlib
new_md5 = hashlib.md5(open("/root/distill_sparse_swiglu/src/evaluate.py", "rb").read()).hexdigest()
print(f"New evaluate.py md5: {new_md5}")

# ========== WRITE NEW batch_eval.py ==========
batch_eval_code = '''"""Batch evaluate all checkpoints: load model once, eval each checkpoint.
Includes vanilla TEAL (activation-magnitude baseline) computed once."""
import gc
import hashlib
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import (
    _best_attn_impl, _make_original_fwd,
    evaluate_perplexity, evaluate_true_dense,
    teal_global_masks, vanilla_teal_global_masks, collect_layer_inputs,
)
from data_utils import get_eval_dataset
from predictor import PredictorWrapper
from transformers import AutoModelForCausalLM, AutoTokenizer

CHECKPOINTS = [
    ("mvp_bce_s42", "checkpoints/mvp_bce_s42/predictor_bce.pt"),
    ("mvp_bce_s123", "checkpoints/mvp_bce_s123/predictor_bce.pt"),
    ("mvp_bce_s456", "checkpoints/mvp_bce_s456/predictor_bce.pt"),
    ("mvp_kl_norm_v2_s42", "checkpoints/mvp_kl_norm_v2_s42/predictor_kl_normalized.pt"),
    ("mvp_kl_norm_v2_s123", "checkpoints/mvp_kl_norm_v2_s123/predictor_kl_normalized.pt"),
    ("mvp_kl_norm_v2_s456", "checkpoints/mvp_kl_norm_v2_s456/predictor_kl_normalized.pt"),
]

MODEL_PATH = "/root/autodl-tmp/models/llama-3.1-8b"
SEQ_LEN = 2048
SPARSITY = 0.5
CALIB_SAMPLES = 32
BASE_DIR = "/root/distill_sparse_swiglu"


def main():
    device = torch.device("cuda:0")

    eval_py = os.path.join(BASE_DIR, "src/evaluate.py")
    with open(eval_py, "rb") as f:
        eval_md5 = hashlib.md5(f.read()).hexdigest()
    print(f"evaluate.py md5: {eval_md5}")

    print(f"Loading model from {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation=_best_attn_impl(),
    )
    model.eval()

    print("Loading WikiText-2 ...")
    wt2 = get_eval_dataset("wikitext2", tokenizer, SEQ_LEN)
    print(f"  {len(wt2)} sequences")

    calib_ids = torch.stack(
        [wt2[i]["input_ids"] for i in range(min(CALIB_SAMPLES, len(wt2)))]
    ).to(device)

    # ---- Computed once (no predictor dependency) ----
    print("\\n=== True Dense PPL (computed once) ===")
    true_dense_ppl = evaluate_true_dense(model, wt2, device)
    print(f"  WikiText-2 True Dense PPL: {true_dense_ppl:.4f}")

    print("\\n=== Vanilla TEAL (activation-magnitude baseline, computed once) ===")
    vanilla_masks = vanilla_teal_global_masks(model, calib_ids, SPARSITY)
    vteal_total_n = sum(m.numel() for m in vanilla_masks.values())
    vteal_total_z = sum((m == 0).sum().item() for m in vanilla_masks.values())
    vteal_sparsity = vteal_total_z / vteal_total_n
    print(f"  Vanilla TEAL model-level sparsity: {vteal_sparsity:.3f}")

    print("  Per-layer sparsity:")
    for idx in sorted(vanilla_masks):
        sp = 1.0 - vanilla_masks[idx].mean().item()
        print(f"    Layer {idx:2d}: {sp:.3f}")

    vanilla_teal_ppl = evaluate_perplexity(model, wt2, device, vanilla_masks)
    print(f"  Vanilla TEAL PPL: {vanilla_teal_ppl:.4f}")

    # ---- Per-checkpoint evaluation ----
    results = []

    for exp_id, ckpt_rel in CHECKPOINTS:
        ckpt_path = os.path.join(BASE_DIR, ckpt_rel)
        if not os.path.exists(ckpt_path):
            print(f"\\n--- {exp_id}: SKIPPED (checkpoint not found: {ckpt_path}) ---")
            results.append({"exp_id": exp_id, "status": "missing"})
            continue

        print(f"\\n--- {exp_id} ---")
        t0 = time.time()

        wrapper = PredictorWrapper(model, bottleneck_size=128)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        wrapper.predictors.load_state_dict(ckpt["predictors"])
        wrapper.predictors.to(device=device, dtype=torch.bfloat16)
        wrapper.predictors.eval()
        wrapper.gumbel_mask.hard = True

        # Predictor-based TEAL masks
        pred_global_masks = teal_global_masks(wrapper, calib_ids, SPARSITY)

        total_n = sum(m.numel() for m in pred_global_masks.values())
        total_z = sum((m == 0).sum().item() for m in pred_global_masks.values())
        model_sparsity = total_z / total_n

        # Predictor hard mask PPL
        pred_ppl = evaluate_perplexity(model, wt2, device)

        # Predictor TEAL sparse PPL
        teal_ppl = evaluate_perplexity(model, wt2, device, pred_global_masks)

        elapsed = time.time() - t0

        r = {
            "exp_id": exp_id,
            "true_dense_ppl": round(true_dense_ppl, 4),
            "predictor_hard_mask_ppl": round(pred_ppl, 4),
            "teal_sparse_ppl": round(teal_ppl, 4),
            "vanilla_teal_sparse_ppl": round(vanilla_teal_ppl, 4),
            "model_sparsity": round(model_sparsity, 4),
            "vanilla_teal_sparsity": round(vteal_sparsity, 4),
            "eval_seconds": round(elapsed, 1),
            "eval_md5": eval_md5,
            "calibration_samples": CALIB_SAMPLES,
        }
        results.append(r)
        print(f"  true_dense={true_dense_ppl:.2f}  pred_hard={pred_ppl:.2f}  pred_teal={teal_ppl:.2f}  vanilla_teal={vanilla_teal_ppl:.2f}  sparsity={model_sparsity:.3f}  ({elapsed:.0f}s)")

        del wrapper, ckpt, pred_global_masks
        gc.collect()
        torch.cuda.empty_cache()

    # Summary table
    print("\\n\\n=== RESULTS TABLE ===")
    print(f"{'exp_id':<25} {'True Dense':>12} {'Pred HardMask':>14} {'Pred TEAL':>10} {'Van. TEAL':>10} {'Sparsity':>9}")
    print("-" * 83)
    for r in results:
        if r.get("status") == "missing":
            print(f"{r['exp_id']:<25} {'MISSING':>12}")
            continue
        print(f"{r['exp_id']:<25} {r['true_dense_ppl']:>12.2f} {r['predictor_hard_mask_ppl']:>14.2f} {r['teal_sparse_ppl']:>10.2f} {r['vanilla_teal_sparse_ppl']:>10.2f} {r['model_sparsity']:>9.3f}")

    # Save JSON
    out_path = os.path.join(BASE_DIR, "artifacts", "eval_results_v3_vanilla_teal.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    full_results = {
        "eval_md5": eval_md5,
        "vanilla_teal_ppl": round(vanilla_teal_ppl, 4),
        "vanilla_teal_sparsity": round(vteal_sparsity, 4),
        "true_dense_ppl": round(true_dense_ppl, 4),
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
'''

with open("/root/distill_sparse_swiglu/src/batch_eval.py", "w") as f:
    f.write(batch_eval_code)
print("batch_eval.py written successfully")

print("\nDone. All patches applied.")

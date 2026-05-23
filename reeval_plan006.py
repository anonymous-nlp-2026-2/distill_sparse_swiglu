"""Plan 006 re-evaluation: all training-free baselines with fixed code."""
import json, os, sys, time
sys.path.insert(0, '/root/distill_sparse_swiglu/src')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_utils import get_eval_dataset
from evaluate import (
    _collect_swiglu_magnitudes,
    vanilla_teal_global_masks,
    teal_activation_magnitude_masks,
    wina_global_masks,
    rsparse_global_masks,
    evaluate_perplexity,
)
from baselines import teal_greedy_allocation_masks as baselines_teal_greedy
from baselines import wina_greedy_allocation_masks

DEVICE = "cuda:0"
MODEL_PATH = "/root/autodl-tmp/models/llama-3.1-8b"
SPARSITY = 0.5
CALIB_SAMPLES = 32
EVAL_SAMPLES = 200
SEQ_LEN = 2048
SVD_RANK = 256
OUT_DIR = "/root/distill_sparse_swiglu/results/plan_006_reeval"

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading model...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE, attn_implementation="sdpa"
)
model.eval()
print(f"Model loaded in {time.time()-t0:.1f}s")

print("Loading data...")
eval_ds = get_eval_dataset("wikitext2", tokenizer, seq_len=SEQ_LEN,
                           max_samples=EVAL_SAMPLES)
calib_ids = torch.stack([ex["input_ids"] for ex in eval_ds[:CALIB_SAMPLES]]).to(DEVICE)
print(f"Eval: {len(eval_ds)} samples, Calib: {calib_ids.shape}")

results = {}

# 1. Dense
print("\n=== Dense ===")
ppl = evaluate_perplexity(model, eval_ds, DEVICE, global_masks=None)
results["dense"] = round(ppl, 4)
print(f"Dense PPL: {ppl:.4f}")

# 2. Vanilla TEAL (global, normalized via fixed _global_topk_masks)
print("\n=== Vanilla TEAL (global, normalized) ===")
masks = vanilla_teal_global_masks(model, calib_ids, SPARSITY)
ppl = evaluate_perplexity(model, eval_ds, DEVICE, global_masks=masks)
results["vanilla_teal_global"] = round(ppl, 4)
keep_ratios = {li: masks[li].float().mean().item() for li in sorted(masks)}
print(f"PPL: {ppl:.4f}")
print(f"Keep ratios (first 5 layers): {[f'{keep_ratios[i]:.3f}' for i in range(min(5, len(keep_ratios)))]}")

# 3. TEAL greedy (evaluate.py version, with error normalization fix)
print("\n=== TEAL Greedy (evaluate.py, error-normalized) ===")
masks = teal_activation_magnitude_masks(model, calib_ids, SPARSITY)
ppl = evaluate_perplexity(model, eval_ds, DEVICE, global_masks=masks)
results["teal_greedy_eval"] = round(ppl, 4)
keep_ratios = {li: masks[li].float().mean().item() for li in sorted(masks)}
print(f"PPL: {ppl:.4f}")
print(f"Keep ratios (first 5): {[f'{keep_ratios[i]:.3f}' for i in range(min(5, len(keep_ratios)))]}")
print(f"Keep ratios (last 5): {[f'{keep_ratios[i]:.3f}' for i in range(max(0,len(keep_ratios)-5), len(keep_ratios))]}")

# 4. TEAL greedy (baselines.py version, with error normalization fix)
print("\n=== TEAL Greedy (baselines.py, error-normalized) ===")
masks = baselines_teal_greedy(model, calib_ids, SPARSITY)
ppl = evaluate_perplexity(model, eval_ds, DEVICE, global_masks=masks)
results["teal_greedy_baselines"] = round(ppl, 4)
keep_ratios = {li: masks[li].float().mean().item() for li in sorted(masks)}
print(f"PPL: {ppl:.4f}")
print(f"Keep ratios (first 5): {[f'{keep_ratios[i]:.3f}' for i in range(min(5, len(keep_ratios)))]}")

# 5. WINA global (normalized via fixed _global_topk_masks)
print("\n=== WINA Global (normalized) ===")
masks = wina_global_masks(model, calib_ids, SPARSITY)
ppl = evaluate_perplexity(model, eval_ds, DEVICE, global_masks=masks)
results["wina_global"] = round(ppl, 4)
print(f"PPL: {ppl:.4f}")

# 6. WINA greedy (from baselines.py)
print("\n=== WINA Greedy ===")
masks = wina_greedy_allocation_masks(model, calib_ids, SPARSITY)
ppl = evaluate_perplexity(model, eval_ds, DEVICE, global_masks=masks)
results["wina_greedy"] = round(ppl, 4)
print(f"PPL: {ppl:.4f}")

# 7. R-Sparse global (normalized via fixed _global_topk_masks)
print("\n=== R-Sparse Global (normalized) ===")
masks = rsparse_global_masks(model, calib_ids, SPARSITY, svd_rank=SVD_RANK)
ppl = evaluate_perplexity(model, eval_ds, DEVICE, global_masks=masks)
results["rsparse_global"] = round(ppl, 4)
print(f"PPL: {ppl:.4f}")

# 8. Uniform 50% (sanity check)
print("\n=== Uniform 50% ===")
inter_dim = model.model.layers[0].mlp.gate_proj.out_features
avg_mag = _collect_swiglu_magnitudes(model, calib_ids)
uniform_masks = {}
for li in avg_mag:
    nk = int(inter_dim * 0.5)
    thr = torch.topk(avg_mag[li], nk).values[-1]
    uniform_masks[li] = (avg_mag[li] >= thr).to(torch.bfloat16)
ppl = evaluate_perplexity(model, eval_ds, DEVICE, global_masks=uniform_masks)
results["uniform_50pct"] = round(ppl, 4)
print(f"PPL: {ppl:.4f}")

# Save results
print("\n" + "="*60)
print("FINAL RESULTS:")
for k, v in results.items():
    print(f"  {k:30s}: {v:.4f}")

out_path = os.path.join(OUT_DIR, "corrected_baselines.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")

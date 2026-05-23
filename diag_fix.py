"""Fix and re-evaluate training-free baselines with per-layer normalized global topk."""
import os, sys, json, time, torch
import torch.nn.functional as F
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/root/autodl-tmp/.hf_cache"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.insert(0, "/root/distill_sparse_swiglu/src")

from transformers import AutoModelForCausalLM, AutoTokenizer
from data_utils import get_eval_dataset
from evaluate import (
    _collect_swiglu_magnitudes, _global_topk_masks, _print_mask_stats,
    evaluate_perplexity, evaluate_true_dense,
    teal_activation_magnitude_masks,
    wina_global_masks, rsparse_global_masks, vanilla_teal_global_masks,
)

MODEL_PATH = "/root/autodl-tmp/models/llama-3.1-8b"
device = torch.device("cuda:0")

print("=== Loading model ===")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map={"": device},
    attn_implementation="sdpa"
)
model.eval()

print("Loading WikiText-2 ...")
wt2 = get_eval_dataset("wikitext2", tokenizer, 2048)
print(f"  {len(wt2)} sequences")

calib_ids = torch.stack([wt2[i]["input_ids"] for i in range(min(32, len(wt2)))]).to(device)
print(f"  Calibration: {calib_ids.shape}")

n_layers = model.config.num_hidden_layers
inter_dim = model.config.intermediate_size
results = {}

# ----------------------------------------------------------------
# Phase 0: Dense baseline
# ----------------------------------------------------------------
print("\n=== Dense baseline ===")
ppl_dense = evaluate_true_dense(model, wt2, device)
print(f"  Dense PPL: {ppl_dense:.4f}")
results["dense"] = ppl_dense

# ----------------------------------------------------------------
# Phase 1: Control - uniform per-layer 50% (magnitude-based)
# ----------------------------------------------------------------
print("\n=== Control: Uniform per-layer 50% (magnitude) ===")
avg_mag = _collect_swiglu_magnitudes(model, calib_ids)

uniform_masks = {}
for li in range(n_layers):
    s = avg_mag[li].float()
    nk = max(int(inter_dim * 0.5), 1)
    thr = torch.topk(s, nk).values[-1]
    uniform_masks[li] = (s >= thr).to(torch.bfloat16)

sp = _print_mask_stats("Uniform-50%", uniform_masks)
ppl_uniform = evaluate_perplexity(model, wt2, device, uniform_masks)
print(f"  Uniform 50% PPL: {ppl_uniform:.4f}")
results["uniform_50pct"] = ppl_uniform

# ----------------------------------------------------------------
# Phase 2: Fixed global topk with per-layer normalization
# ----------------------------------------------------------------
print("\n=== Fixed: Normalized global topk ===")

def _global_topk_masks_normalized(layer_scores, sparsity_target):
    """Global top-K with per-layer mean normalization to balance layer scales."""
    items = sorted(layer_scores.items())
    normalized = []
    for li, s in items:
        s_float = s.float()
        mu = s_float.mean()
        if mu > 1e-12:
            normalized.append((li, s_float / mu))
        else:
            normalized.append((li, s_float))
    all_s = torch.cat([s for _, s in normalized])
    num_keep = int(len(all_s) * (1.0 - sparsity_target))
    thr = torch.topk(all_s, num_keep).values[-1]
    return {li: (s >= thr).to(torch.bfloat16) for li, s in normalized}

# -- Vanilla TEAL (normalized) --
print("\n--- Vanilla TEAL (normalized) ---")
masks_vt = _global_topk_masks_normalized(avg_mag, 0.5)
sp = _print_mask_stats("Vanilla-TEAL-norm", masks_vt)
ppl_vt = evaluate_perplexity(model, wt2, device, masks_vt)
print(f"  Vanilla TEAL (norm) PPL: {ppl_vt:.4f}")
results["vanilla_teal_norm"] = ppl_vt

# -- WINA (normalized) --
print("\n--- WINA (normalized) ---")
wina_scores = {}
for li in avg_mag:
    w = model.model.layers[li].mlp.down_proj.weight
    cn = w.float().norm(dim=0)
    wina_scores[li] = avg_mag[li].float() * cn.to(avg_mag[li].device)

masks_wina = _global_topk_masks_normalized(wina_scores, 0.5)
sp = _print_mask_stats("WINA-norm", masks_wina)
ppl_wina = evaluate_perplexity(model, wt2, device, masks_wina)
print(f"  WINA (norm) PPL: {ppl_wina:.4f}")
results["wina_norm"] = ppl_wina

# -- R-Sparse (normalized) --
print("\n--- R-Sparse (normalized) ---")
rsparse_scores = {}
for li in avg_mag:
    w = model.model.layers[li].mlp.down_proj.weight
    _, S, Vh = torch.linalg.svd(w.float(), full_matrices=False)
    r = min(256, S.shape[0])
    ws = (S[:r].unsqueeze(1) * Vh[:r].abs()).sum(dim=0)
    rsparse_scores[li] = avg_mag[li].float() * ws.to(avg_mag[li].device)

masks_rs = _global_topk_masks_normalized(rsparse_scores, 0.5)
sp = _print_mask_stats("R-Sparse-norm", masks_rs)
ppl_rs = evaluate_perplexity(model, wt2, device, masks_rs)
print(f"  R-Sparse (norm) PPL: {ppl_rs:.4f}")
results["rsparse_norm"] = ppl_rs

# ----------------------------------------------------------------
# Phase 3: TEAL greedy (existing code - per-layer allocation)
# ----------------------------------------------------------------
print("\n=== TEAL greedy (existing per-layer allocation) ===")
masks_tg = teal_activation_magnitude_masks(model, calib_ids, 0.5)
sp = _print_mask_stats("TEAL-greedy", masks_tg)
ppl_tg = evaluate_perplexity(model, wt2, device, masks_tg)
print(f"  TEAL greedy PPL: {ppl_tg:.4f}")
results["teal_greedy"] = ppl_tg

# ----------------------------------------------------------------
# Phase 4: TEAL greedy with normalized errors
# ----------------------------------------------------------------
print("\n=== TEAL greedy (normalized errors) ===")

@torch.no_grad()
def teal_greedy_normalized(model, input_ids, sparsity_target=0.5,
                           step_size=0.05, calib_batch_size=4):
    """TEAL greedy with error normalization by dense output L2 norm."""
    n_layers = len(model.model.layers)
    inter_dim_local = model.model.layers[0].mlp.gate_proj.out_features
    dev = next(model.parameters()).device

    bufs = {i: [] for i in range(n_layers)}
    hooks = []
    for idx, layer in enumerate(model.model.layers):
        def _hook(idx_):
            def fn(module, inp, out):
                bufs[idx_].append(inp[0].detach().cpu())
            return fn
        hooks.append(layer.mlp.register_forward_hook(_hook(idx)))

    for s in range(0, input_ids.size(0), calib_batch_size):
        model(input_ids[s:s + calib_batch_size])

    for h in hooks:
        h.remove()

    mlp_inputs = {i: torch.cat(bufs[i], dim=0) for i in range(n_layers)}
    del bufs

    levels = [round(i * step_size, 4) for i in range(int(1.0 / step_size) + 1)]
    error_table = {}
    mean_mags = {}
    chunk = 4

    for li in range(n_layers):
        mlp = model.model.layers[li].mlp
        gp, up, dp, af = mlp.gate_proj, mlp.up_proj, mlp.down_proj, mlp.act_fn

        accum_mag = torch.zeros(inter_dim_local, dtype=torch.float32)
        inter_chunks, dense_chunks = [], []
        total_tokens = 0

        for s in range(0, mlp_inputs[li].size(0), chunk):
            x = mlp_inputs[li][s:s + chunk].to(dev)
            inter = af(gp(x)) * up(x)
            y_dense = dp(inter)
            accum_mag += inter.abs().float().sum(dim=(0, 1)).cpu()
            total_tokens += x.size(0) * x.size(1)
            inter_chunks.append(inter.cpu())
            dense_chunks.append(y_dense.cpu())
            del x, inter, y_dense

        mean_mag = accum_mag / total_tokens
        mean_mags[li] = mean_mag
        all_inter = torch.cat(inter_chunks, dim=0)
        all_dense = torch.cat(dense_chunks, dim=0)
        del inter_chunks, dense_chunks

        # Compute dense output norm for normalization
        dense_mse = all_dense.float().pow(2).mean().item()

        errors = []
        for sp_val in levels:
            if sp_val <= 0.0:
                errors.append(0.0)
                continue
            nk = max(int(inter_dim_local * (1.0 - sp_val)), 1)
            thr = torch.topk(mean_mag, nk).values[-1]
            mask_1d = mean_mag >= thr

            total_err = 0.0
            total_elem = 0
            for s in range(0, all_inter.size(0), chunk):
                ic = all_inter[s:s + chunk].to(dev)
                dc = all_dense[s:s + chunk].to(dev)
                m = mask_1d.unsqueeze(0).unsqueeze(0).to(device=dev, dtype=ic.dtype)
                y_sp = dp(ic * m)
                total_err += (dc - y_sp).pow(2).sum().item()
                total_elem += dc.numel()
                del ic, dc, y_sp
            raw_err = total_err / max(total_elem, 1)
            errors.append(raw_err / max(dense_mse, 1e-12))

        error_table[li] = errors
        del all_inter, all_dense

    del mlp_inputs
    torch.cuda.empty_cache()

    cur = {i: 0 for i in range(n_layers)}
    def avg_sp():
        return sum(levels[cur[i]] for i in range(n_layers)) / n_layers

    while avg_sp() < sparsity_target:
        best_l, best_m = None, float("inf")
        for li in range(n_layers):
            ci = cur[li]
            if ci + 1 >= len(levels):
                continue
            marginal = error_table[li][ci + 1] - error_table[li][ci]
            if marginal < best_m:
                best_m = marginal
                best_l = li
        if best_l is None:
            break
        cur[best_l] += 1

    masks = {}
    for li in range(n_layers):
        sp_val = levels[cur[li]]
        if sp_val <= 0:
            masks[li] = torch.ones(inter_dim_local, dtype=torch.bfloat16)
        else:
            nk = max(int(inter_dim_local * (1.0 - sp_val)), 1)
            thr = torch.topk(mean_mags[li], nk).values[-1]
            masks[li] = (mean_mags[li] >= thr).to(torch.bfloat16)
    return masks

masks_tgn = teal_greedy_normalized(model, calib_ids, 0.5)
sp = _print_mask_stats("TEAL-greedy-norm", masks_tgn)
ppl_tgn = evaluate_perplexity(model, wt2, device, masks_tgn)
print(f"  TEAL greedy (norm) PPL: {ppl_tgn:.4f}")
results["teal_greedy_norm"] = ppl_tgn

# ----------------------------------------------------------------
# Phase 5: Also reproduce original buggy results
# ----------------------------------------------------------------
print("\n=== Original (buggy) Vanilla TEAL ===")
masks_buggy = _global_topk_masks(avg_mag, 0.5)
_print_mask_stats("Vanilla-TEAL-buggy", masks_buggy)
ppl_buggy = evaluate_perplexity(model, wt2[:20], device, masks_buggy)
print(f"  Vanilla TEAL (buggy, 20 seq) PPL: {ppl_buggy:.4f}")
results["vanilla_teal_buggy"] = ppl_buggy

# ----------------------------------------------------------------
# Summary
# ----------------------------------------------------------------
print("\n" + "="*60)
print("=== SUMMARY ===")
print("="*60)
for k, v in results.items():
    print(f"  {k:30s}: {v:.4f}")

out_dir = "/root/distill_sparse_swiglu/results/plan_006_reeval"
os.makedirs(out_dir, exist_ok=True)
with open(f"{out_dir}/diag_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_dir}/diag_results.json")

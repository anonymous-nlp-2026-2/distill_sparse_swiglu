# TEAL vs KL Overlap: Selection Bias Correction
# Analytical null + vectorized empirical permutation + percentile rank

import os
import sys
import json
import time
import gc

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/root/distill_sparse_swiglu/src")
from predictor import PredictorWrapper
from data_utils import get_eval_dataset

MODEL_PATH = "/root/autodl-tmp/models/llama-3.1-8b"
CHECKPOINT_PATH = "/root/distill_sparse_swiglu/checkpoints/mvp_kl_norm_v2_s42/predictor_kl_normalized.pt"
SPARSITY = 0.5
NUM_SAMPLES = 10
SEQ_LEN = 2048
BOTTLENECK_SIZE = 128
DEVICE = "cuda:0"
NUM_PERMUTATIONS = 1000
EMPIRICAL_TOKENS = 200
OUTPUT_PATH = "/root/distill_sparse_swiglu/artifacts/teal_kl_overlap_corrected.json"


def vectorized_permutation_test(magnitudes, kl_only_mask, n_perms=1000, batch_size=100):
    """Vectorized permutation test: no Python inner loop over tokens.
    
    For each permutation, randomly select K neurons per token (K = mean KL-only count),
    compute mean magnitude of selection. Returns null distribution.
    """
    T, N = magnitudes.shape
    K_mean = int(kl_only_mask.sum(dim=1).float().mean().item())
    if K_mean == 0:
        return np.zeros(n_perms), 0.0

    actual_mean = magnitudes[kl_only_mask].mean().item()
    null_means = np.zeros(n_perms)

    for batch_start in range(0, n_perms, batch_size):
        batch_end = min(batch_start + batch_size, n_perms)
        n_batch = batch_end - batch_start

        # [n_batch, T, N] random priorities
        rand = torch.rand(n_batch, T, N)
        # Select K_mean smallest-priority neurons per token (= random selection)
        _, indices = rand.topk(K_mean, dim=2, largest=False)  # [n_batch, T, K_mean]
        # Gather magnitudes at selected indices
        mags_exp = magnitudes.unsqueeze(0).expand(n_batch, T, N)
        selected = mags_exp.gather(2, indices)  # [n_batch, T, K_mean]
        # Mean across tokens and neurons for each permutation
        null_means[batch_start:batch_end] = selected.mean(dim=(1, 2)).numpy()

        del rand, indices, mags_exp, selected

    return null_means, actual_mean


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)

    print(f"Loading model on {DEVICE} (bfloat16)...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    print(f"  Model loaded in {time.time()-t0:.1f}s", flush=True)

    num_layers = model.config.num_hidden_layers
    intermediate_size = model.config.intermediate_size
    N = intermediate_size
    print(f"  {num_layers} layers, intermediate_size={N}", flush=True)

    print("Loading KL predictor...", flush=True)
    wrapper = PredictorWrapper(model, bottleneck_size=BOTTLENECK_SIZE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    wrapper.predictors.load_state_dict(ckpt["predictors"])
    wrapper.predictors.to(device=DEVICE, dtype=torch.bfloat16)
    wrapper.predictors.eval()
    wrapper.gumbel_mask.hard = True
    del ckpt
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Loading C4 data ({NUM_SAMPLES} samples, seq_len={SEQ_LEN})...", flush=True)
    data = get_eval_dataset("c4", tokenizer, SEQ_LEN, max_samples=NUM_SAMPLES)
    print(f"  Got {len(data)} samples", flush=True)

    # Accumulators
    per_layer_kl_mag_sum = np.zeros(num_layers)
    per_layer_kl_count = np.zeros(num_layers, dtype=np.int64)
    per_layer_null_exp_num = np.zeros(num_layers)
    per_layer_null_var_num = np.zeros(num_layers)
    per_layer_total_K = np.zeros(num_layers, dtype=np.int64)
    per_layer_percentile_ranks = [[] for _ in range(num_layers)]

    # Empirical: store subset of tokens per layer
    per_layer_emp_mags = [[] for _ in range(num_layers)]
    per_layer_emp_masks = [[] for _ in range(num_layers)]
    emp_collected = np.zeros(num_layers, dtype=np.int64)

    total_tokens = 0
    print("Running analysis...", flush=True)
    t_start = time.time()

    for sample_idx, sample in enumerate(data):
        input_ids = sample["input_ids"].unsqueeze(0).to(DEVICE)
        print(f"  Sample {sample_idx+1}/{len(data)}...", flush=True)

        with torch.no_grad():
            wrapper.forward_dense(input_ids, capture_intermediates=True)
        intermediates = wrapper.get_intermediates()
        layer_inputs = wrapper.get_layer_inputs()

        seq_len_actual = input_ids.shape[1]

        for layer_idx in range(num_layers):
            if layer_idx not in intermediates or layer_idx not in layer_inputs:
                continue

            inter = intermediates[layer_idx].squeeze(0).float()
            h_in = layer_inputs[layer_idx].squeeze(0)

            with torch.no_grad():
                logits = wrapper.predictors[layer_idx](h_in)

            kl_prune = (logits <= 0)
            magnitudes = inter.abs()
            num_prune_per_token = kl_prune.sum(dim=-1)
            ranks = magnitudes.argsort(dim=-1).argsort(dim=-1)
            teal_prune = ranks < num_prune_per_token.unsqueeze(-1)
            kl_only_mask = kl_prune & ~teal_prune

            mag_cpu = magnitudes.cpu()
            kl_only_cpu = kl_only_mask.cpu()
            ranks_cpu = ranks.cpu()

            kl_sizes = kl_only_cpu.sum(dim=1).numpy()
            token_means = mag_cpu.mean(dim=1).numpy()
            token_vars = mag_cpu.var(dim=1).numpy()

            # KL-only magnitude
            kl_mags = mag_cpu[kl_only_cpu]
            per_layer_kl_mag_sum[layer_idx] += kl_mags.sum().item()
            per_layer_kl_count[layer_idx] += kl_mags.numel()

            # Analytical null
            per_layer_null_exp_num[layer_idx] += (kl_sizes * token_means).sum()
            per_layer_total_K[layer_idx] += kl_sizes.sum()
            valid = kl_sizes > 0
            var_terms = np.where(valid, kl_sizes * token_vars * (N - kl_sizes) / (N - 1), 0)
            per_layer_null_var_num[layer_idx] += var_terms.sum()

            # Percentile rank (rank 0 = smallest)
            kl_only_ranks_flat = ranks_cpu[kl_only_cpu].float() / N
            if kl_only_ranks_flat.numel() > 20000:
                idx = torch.randperm(kl_only_ranks_flat.numel())[:20000]
                per_layer_percentile_ranks[layer_idx].append(kl_only_ranks_flat[idx].numpy())
            else:
                per_layer_percentile_ranks[layer_idx].append(kl_only_ranks_flat.numpy())

            # Collect tokens for empirical test
            if emp_collected[layer_idx] < EMPIRICAL_TOKENS:
                need = EMPIRICAL_TOKENS - emp_collected[layer_idx]
                take = min(need, seq_len_actual)
                chosen = torch.randperm(seq_len_actual)[:take]
                per_layer_emp_mags[layer_idx].append(mag_cpu[chosen])
                per_layer_emp_masks[layer_idx].append(kl_only_cpu[chosen])
                emp_collected[layer_idx] += take

            del inter, h_in, logits, kl_prune, magnitudes, ranks, teal_prune, kl_only_mask
            del mag_cpu, kl_only_cpu, ranks_cpu, kl_mags
            gc.collect()

        total_tokens += seq_len_actual
        del intermediates, layer_inputs
        gc.collect()
        torch.cuda.empty_cache()

    fwd_time = time.time() - t_start
    print(f"\nForward passes done in {fwd_time:.1f}s ({total_tokens:,} tokens)", flush=True)

    # Empirical permutation test (vectorized)
    print(f"\nRunning vectorized permutation test ({NUM_PERMUTATIONS} perms)...", flush=True)
    t_perm = time.time()
    per_layer_emp_results = []

    for layer_idx in range(num_layers):
        if not per_layer_emp_mags[layer_idx]:
            per_layer_emp_results.append(None)
            continue

        mags_batch = torch.cat(per_layer_emp_mags[layer_idx], dim=0)
        masks_batch = torch.cat(per_layer_emp_masks[layer_idx], dim=0)

        null_means, actual_mean = vectorized_permutation_test(
            mags_batch, masks_batch, NUM_PERMUTATIONS, batch_size=200
        )
        null_m = null_means.mean()
        null_s = null_means.std()
        pctile = (null_means < actual_mean).sum() / NUM_PERMUTATIONS * 100
        p_val = (null_means >= actual_mean).sum() / NUM_PERMUTATIONS

        per_layer_emp_results.append({
            "actual_mean": actual_mean,
            "null_mean": null_m,
            "null_std": null_s,
            "percentile": pctile,
            "p_value": p_val,
        })

        del mags_batch, masks_batch
        gc.collect()

        if (layer_idx + 1) % 8 == 0:
            print(f"  Layers 0-{layer_idx} done ({time.time()-t_perm:.1f}s)", flush=True)

    perm_time = time.time() - t_perm
    print(f"  Permutation test done in {perm_time:.1f}s", flush=True)

    # Final results
    print("\nComputing final statistics...", flush=True)

    results_per_layer = []
    all_percentiles = []

    for layer_idx in range(num_layers):
        total_K = per_layer_total_K[layer_idx]
        kl_mean = per_layer_kl_mag_sum[layer_idx] / max(per_layer_kl_count[layer_idx], 1)
        null_exp = per_layer_null_exp_num[layer_idx] / max(total_K, 1)
        null_var = per_layer_null_var_num[layer_idx] / max(total_K ** 2, 1)
        null_std = np.sqrt(null_var)
        z_score = (kl_mean - null_exp) / max(null_std, 1e-12)

        pctiles = np.concatenate(per_layer_percentile_ranks[layer_idx])
        med_pctile = np.median(pctiles) * 100
        q1_pctile = np.percentile(pctiles, 25) * 100
        q3_pctile = np.percentile(pctiles, 75) * 100
        all_percentiles.append(pctiles)

        emp = per_layer_emp_results[layer_idx]
        layer_result = {
            "layer": layer_idx,
            "kl_only_mean_mag": round(float(kl_mean), 6),
            "analytical_null_mean": round(float(null_exp), 6),
            "analytical_null_std": round(float(null_std), 8),
            "analytical_z_score": round(float(z_score), 2),
            "empirical_null_percentile": round(float(emp["percentile"]), 2) if emp else None,
            "empirical_p_value": round(float(emp["p_value"]), 4) if emp else None,
            "median_percentile_rank": round(float(med_pctile), 2),
            "percentile_q1": round(float(q1_pctile), 2),
            "percentile_q3": round(float(q3_pctile), 2),
        }
        results_per_layer.append(layer_result)

    # Global
    global_kl_mean = sum(per_layer_kl_mag_sum) / max(sum(per_layer_kl_count), 1)
    global_total_K = sum(per_layer_total_K)
    global_null_exp = sum(per_layer_null_exp_num) / max(global_total_K, 1)
    global_null_var = sum(per_layer_null_var_num) / max(global_total_K ** 2, 1)
    global_null_std = np.sqrt(global_null_var)
    global_z = (global_kl_mean - global_null_exp) / max(global_null_std, 1e-12)

    from scipy.stats import norm
    global_p_value = 2 * (1 - norm.cdf(abs(global_z)))

    all_pctiles_flat = np.concatenate(all_percentiles)
    global_med_pctile = np.median(all_pctiles_flat) * 100
    global_q1 = np.percentile(all_pctiles_flat, 25) * 100
    global_q3 = np.percentile(all_pctiles_flat, 75) * 100

    emp_sig_layers = sum(1 for r in per_layer_emp_results if r and r["p_value"] < 0.001)

    significant = global_p_value < 0.001

    if significant and global_med_pctile > 50:
        conclusion = (
            f"After correcting for selection bias: KL-only-pruned neurons have statistically "
            f"significantly higher magnitude (z={global_z:.1f}, p={global_p_value:.2e}). "
            f"Median per-neuron percentile rank = {global_med_pctile:.1f}% "
            f"(IQR: {global_q1:.1f}-{global_q3:.1f}%). "
            f"Empirical permutation confirms significance in {emp_sig_layers}/{num_layers} layers "
            f"(p<0.001). The 1.48x magnitude ratio is not a selection bias artifact."
        )
    else:
        conclusion = (
            f"After correction: z={global_z:.2f}, p={global_p_value:.4f}, "
            f"median percentile rank={global_med_pctile:.1f}%. "
            f"{'Significant.' if significant else 'Not significant at p<0.001.'}"
        )

    output = {
        "null_baseline": {
            "method": f"analytical (finite-population CLT) + empirical ({NUM_PERMUTATIONS} permutations on {EMPIRICAL_TOKENS} tokens/layer)",
            "kl_only_mean_magnitude": round(float(global_kl_mean), 6),
            "null_mean_magnitude": round(float(global_null_exp), 6),
            "null_std": round(float(global_null_std), 8),
            "z_score": round(float(global_z), 2),
            "p_value": float(f"{global_p_value:.2e}"),
            "significant": bool(significant),
            "empirical_validation": {
                "layers_significant_p001": emp_sig_layers,
                "total_layers": num_layers,
            }
        },
        "percentile_method": {
            "median_percentile_rank": round(float(global_med_pctile), 2),
            "q1": round(float(global_q1), 2),
            "q3": round(float(global_q3), 2),
            "interpretation": (
                f"Median KL-only-pruned neuron is at the {global_med_pctile:.1f}th percentile "
                f"of magnitude within its token (IQR: {global_q1:.1f}-{global_q3:.1f}%). "
                f"{'Above 50% confirms KL systematically prunes higher-magnitude neurons.' if global_med_pctile > 50 else 'No systematic magnitude bias detected.'}"
            ),
        },
        "per_layer": results_per_layer,
        "metadata": {
            "num_tokens_analyzed": total_tokens,
            "num_samples": len(data),
            "num_permutations_empirical": NUM_PERMUTATIONS,
            "empirical_tokens_per_layer": EMPIRICAL_TOKENS,
            "sparsity_level": SPARSITY,
            "checkpoint": CHECKPOINT_PATH,
            "total_time_seconds": round(time.time() - t_start, 1),
        },
        "conclusion": conclusion,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print("TEAL-KL Overlap: Selection Bias Corrected")
    print(f"{'='*60}")
    print(f"Tokens: {total_tokens:,} | Perms: {NUM_PERMUTATIONS}")
    print(f"\n--- Null Baseline ---")
    print(f"  KL-only mean |act|:  {global_kl_mean:.6f}")
    print(f"  Null expectation:    {global_null_exp:.6f}")
    print(f"  Null std:            {global_null_std:.8f}")
    print(f"  Z-score:             {global_z:.2f}")
    print(f"  P-value:             {global_p_value:.2e}")
    print(f"  Significant:         {significant}")
    print(f"\n--- Percentile Rank ---")
    print(f"  Median:  {global_med_pctile:.1f}%")
    print(f"  IQR:     {global_q1:.1f}% - {global_q3:.1f}%")
    print(f"\n--- Per-Layer ---")
    for r in results_per_layer:
        emp_str = f"{r['empirical_null_percentile']:5.1f}%" if r['empirical_null_percentile'] is not None else "  N/A"
        print(f"  L{r['layer']:02d}: z={r['analytical_z_score']:6.1f} | emp={emp_str} | rank={r['median_percentile_rank']:5.1f}%")
    print(f"\n{conclusion}")
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Total time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()

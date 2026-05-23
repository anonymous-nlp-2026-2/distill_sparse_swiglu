"""Supplementary SPON evaluation: dense+SPON degradation and TEAL+SPON.

Uses predictor-based TEAL (global allocation from predictor scores).
"""
import argparse
import gc
import importlib.util
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import get_eval_dataset
from predictor import PredictorWrapper
from spon import SPONBiasVectors
from evaluate import teal_global_masks, evaluate_perplexity, evaluate_true_dense


def _best_attn_impl():
    if importlib.util.find_spec("flash_attn"):
        return "flash_attention_2"
    return "sdpa"


def load_model(model_path, device):
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
    return model, tokenizer


def load_spon_biases(path, num_layers, intermediate_size, device):
    spon_biases = SPONBiasVectors(num_layers, intermediate_size)
    ckpt = torch.load(path, map_location=device, weights_only=True)
    spon_biases.load_state_dict(ckpt["spon_biases"])
    spon_biases.to(device=device, dtype=torch.bfloat16)
    spon_biases.eval()
    bias_norms = [b.data.norm().item() for b in spon_biases.biases]
    print(f"SPON bias stats: mean_norm={sum(bias_norms)/len(bias_norms):.6f}, "
          f"max_norm={max(bias_norms):.6f}")
    return spon_biases


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", default="/root/autodl-tmp/models/llama-3.1-8b")
    p.add_argument("--predictor_checkpoint", required=True)
    p.add_argument("--spon_checkpoint", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--max_eval_samples", type=int, default=200)
    p.add_argument("--sparsity_target", type=float, default=0.5)
    p.add_argument("--calibration_samples", type=int, default=32)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}")

    # === Phase 1: Load model, get true dense PPL, then fold SPON ===
    print("Loading model ...")
    model, tokenizer = load_model(args.model_name_or_path, device)
    config = model.config

    print("Loading WikiText-2 ...")
    wt2 = get_eval_dataset("wikitext2", tokenizer, args.seq_len, args.max_eval_samples)
    print(f"  {len(wt2)} sequences, seq_len={args.seq_len}")

    print("\n[1/5] True Dense PPL ...")
    true_dense_ppl = evaluate_true_dense(model, wt2, device)
    print(f"  {true_dense_ppl:.4f}")

    # Fold SPON biases into model
    print("\nLoading and folding SPON biases ...")
    spon_biases = load_spon_biases(
        args.spon_checkpoint, config.num_hidden_layers,
        config.intermediate_size, device
    )
    spon_biases.fold_into_model(model)

    print("[2/5] Dense with SPON biases folded ...")
    dense_spon_ppl = evaluate_true_dense(model, wt2, device)
    print(f"  {dense_spon_ppl:.4f}")
    dense_degradation = dense_spon_ppl - true_dense_ppl
    print(f"  degradation: +{dense_degradation:.4f}")

    # === Phase 2: TEAL masks (predictor-based) ===
    # Need clean model + predictor for TEAL mask computation
    # But we also need the SPON-folded model for TEAL+SPON eval.
    # Strategy: compute TEAL masks on clean model with predictor,
    # then evaluate on clean model (TEAL no SPON) and SPON-folded model (TEAL+SPON).

    # Keep SPON-folded model for later
    model_spon = model

    print("\nPreparing calibration data ...")
    calib_data = get_eval_dataset("wikitext2", tokenizer, args.seq_len, args.calibration_samples)
    calib_ids = torch.stack([ex["input_ids"] for ex in calib_data[:args.calibration_samples]]).to(device)
    print(f"  calibration: {calib_ids.shape[0]} sequences")

    # Load clean model for predictor wrapper
    print("\nLoading clean model for TEAL mask computation ...")
    del model  # alias to model_spon, don't gc
    model_clean, _ = load_model(args.model_name_or_path, device)

    # Create predictor wrapper and load predictor checkpoint
    print("Loading predictor ...")
    wrapper = PredictorWrapper(model_clean, bottleneck_size=128)
    pred_ckpt = torch.load(args.predictor_checkpoint, map_location=device, weights_only=True)
    wrapper.predictors.load_state_dict(pred_ckpt["predictors"])
    wrapper.predictors.to(device=device, dtype=torch.bfloat16)
    wrapper.predictors.eval()
    wrapper.gumbel_mask.hard = True

    print("[3/5] Computing predictor-based TEAL global masks ...")
    teal_masks = teal_global_masks(wrapper, calib_ids, args.sparsity_target)

    total_n = sum(m.numel() for m in teal_masks.values())
    total_z = sum((m == 0).sum().item() for m in teal_masks.values())
    teal_sparsity = total_z / total_n
    print(f"  TEAL sparsity: {teal_sparsity:.4f}")
    print("  Per-layer sparsity:")
    for idx in sorted(teal_masks):
        sp = 1.0 - teal_masks[idx].mean().item()
        print(f"    Layer {idx:2d}: {sp:.3f}")

    # Evaluate TEAL without SPON on clean model (unwrap predictor first)
    print("\n[4/5] TEAL sparse (no SPON) ...")
    teal_no_spon_ppl = evaluate_perplexity(model_clean, wt2, device, teal_masks)
    print(f"  {teal_no_spon_ppl:.4f}")

    # Free clean model, use SPON-folded model for TEAL+SPON
    del wrapper, model_clean
    gc.collect()
    torch.cuda.empty_cache()

    print("[5/5] TEAL sparse + SPON biases ...")
    teal_with_spon_ppl = evaluate_perplexity(model_spon, wt2, device, teal_masks)
    print(f"  {teal_with_spon_ppl:.4f}")

    # === Summary ===
    spon_recovery_teal = (
        100 * (teal_no_spon_ppl - teal_with_spon_ppl) / (teal_no_spon_ppl - true_dense_ppl)
        if teal_no_spon_ppl > true_dense_ppl else 0
    )

    print(f"\n{'='*50}")
    print(f"SPON 2048 Supplementary Evaluation")
    print(f"{'='*50}")
    print(f"true_dense_ppl:            {true_dense_ppl:.4f}")
    print(f"dense_with_spon_ppl:       {dense_spon_ppl:.4f}")
    print(f"dense_ppl_degradation:     +{dense_degradation:.4f}")
    print(f"teal_sparse_no_spon_ppl:   {teal_no_spon_ppl:.4f}")
    print(f"teal_sparse_with_spon_ppl: {teal_with_spon_ppl:.4f}")
    print(f"spon_recovery_teal:        {spon_recovery_teal:.1f}%")
    print(f"teal_sparsity:             {teal_sparsity:.4f}")


if __name__ == "__main__":
    main()

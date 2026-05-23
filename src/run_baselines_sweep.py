"""Sweep TEAL + WINA baselines at 30/50/70% sparsity. Single model load."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_utils import get_eval_dataset
from evaluate import (
    _best_attn_impl, _collect_swiglu_magnitudes, _global_topk_masks,
    vanilla_teal_global_masks, evaluate_perplexity, evaluate_true_dense,
    _print_mask_stats,
)

# Also import WINA
try:
    from evaluate import wina_global_masks
except ImportError:
    wina_global_masks = None

MODEL = "/root/autodl-tmp/models/llama-3.1-8b"
SEQ_LEN = 2048
CALIB_SAMPLES = 32
SPARSITY_LEVELS = [0.3, 0.5, 0.7]

def main():
    device = torch.device("cuda:0")
    print(f"Loading model from {MODEL} ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16,
        attn_implementation=_best_attn_impl(),
        device_map={"": device},
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    print(f"Model loaded in {time.time()-t0:.1f}s")

    # Load eval + calib data
    wt2 = get_eval_dataset("wikitext2", tokenizer, SEQ_LEN)
    print(f"WikiText-2 eval: {wt2.size(0)} sequences x {wt2.size(1)} tokens")
    wt2 = wt2.to(device)

    calib_ids = wt2[:CALIB_SAMPLES]
    print(f"Calibration: {calib_ids.size(0)} sequences")

    # Dense PPL (once)
    print("\n[Dense] True Dense PPL ...")
    dense_ppl = evaluate_true_dense(model, wt2, device)
    print(f"  Dense PPL: {dense_ppl:.4f}")

    # Collect activation magnitudes (once, reused for all sparsity levels)
    print("\nCollecting calibration activation magnitudes ...")
    avg_mag = _collect_swiglu_magnitudes(model, calib_ids)
    print("  Done.")

    results = {"dense_ppl": dense_ppl}

    for sp in SPARSITY_LEVELS:
        sp_key = f"{int(sp*100)}pct"
        print(f"\n{'='*60}")
        print(f"Sparsity target: {sp*100:.0f}%")
        print(f"{'='*60}")

        # Vanilla TEAL
        print(f"\n[Vanilla TEAL @ {sp*100:.0f}%]")
        masks = vanilla_teal_global_masks(model, calib_ids, sp, avg_mag=avg_mag)
        model_sp = _print_mask_stats(f"TEAL-{sp_key}", masks)
        ppl = evaluate_perplexity(model, wt2, device, masks)
        print(f"  PPL: {ppl:.4f}")
        results[f"teal_{sp_key}"] = {"ppl": ppl, "model_sparsity": model_sp}

        # WINA
        if wina_global_masks is not None:
            print(f"\n[WINA @ {sp*100:.0f}%]")
            masks_w = wina_global_masks(model, calib_ids, sp, avg_mag=avg_mag)
            model_sp_w = _print_mask_stats(f"WINA-{sp_key}", masks_w)
            ppl_w = evaluate_perplexity(model, wt2, device, masks_w)
            print(f"  PPL: {ppl_w:.4f}")
            results[f"wina_{sp_key}"] = {"ppl": ppl_w, "model_sparsity": model_sp_w}

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Dense PPL: {dense_ppl:.4f}")
    for sp in SPARSITY_LEVELS:
        sp_key = f"{int(sp*100)}pct"
        t = results.get(f"teal_{sp_key}", {})
        w = results.get(f"wina_{sp_key}", {})
        print(f"\n  {sp*100:.0f}% sparsity:")
        if t:
            print(f"    Vanilla TEAL PPL: {t['ppl']:.4f} (model sparsity: {t['model_sparsity']:.3f})")
        if w:
            print(f"    WINA PPL:         {w['ppl']:.4f} (model sparsity: {w['model_sparsity']:.3f})")

    # Save JSON
    out_path = "/root/distill_sparse_swiglu/results/baseline_sweep_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()

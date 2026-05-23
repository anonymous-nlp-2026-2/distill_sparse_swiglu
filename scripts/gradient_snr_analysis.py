# Gradient SNR (Signal-to-Noise Ratio) analysis: KL vs BCE training configurations.
# Measures gradient quality by computing SNR = mean(grad)^2 / var(grad) per layer per step.
# Input: frozen LLaMA-3.1-8B + C4 calibration data (local)
# Output: /root/distill_sparse_swiglu/artifacts/gradient_snr_results.json

import argparse
import json
import os
import sys
import time
from collections import defaultdict

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from data_utils import get_calibration_loader
from losses import BCESparsityLoss, KLDistillLoss
from predictor import PredictorWrapper


def compute_layer_grad_snr(wrapper):
    """Compute per-predictor-layer gradient SNR after a backward pass."""
    snr_per_layer = {}
    for i, predictor in enumerate(wrapper.predictors):
        grads = []
        for p in predictor.parameters():
            if p.grad is not None:
                grads.append(p.grad.detach().float().flatten())
        if not grads:
            continue
        g = torch.cat(grads)
        mean_g = g.mean()
        var_g = g.var()
        snr = (mean_g ** 2) / (var_g + 1e-12)
        snr_per_layer[f"layer_{i}"] = {
            "snr": snr.item(),
            "grad_mean": mean_g.item(),
            "grad_std": (var_g ** 0.5).item(),
            "grad_norm": g.norm().item(),
        }
    return snr_per_layer


def run_kl_steps(wrapper, dataloader, device, num_steps, tau_start=1.0, tau_end=0.1, sparsity_target=0.5):
    """Run KL training steps and record per-step gradient SNR."""
    kl_fn = KLDistillLoss(temperature=1.0)
    params = list(wrapper.predictors.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)

    wrapper.predictors.train()
    wrapper.model.train()
    wrapper.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    data_iter = iter(dataloader)
    results_per_step = []

    for step in range(1, num_steps + 1):
        tau = tau_start + (tau_end - tau_start) * (step / num_steps)
        wrapper.gumbel_mask.tau = tau

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        optimizer.zero_grad()

        dense_logits = wrapper.forward_dense(input_ids).logits.detach()
        sparse_logits = wrapper.forward_sparse(input_ids).logits

        kl = kl_fn(dense_logits, sparse_logits)

        masks = wrapper.get_masks()
        actual_sparsity = 1.0 - torch.stack([m.mean() for m in masks.values()]).mean()
        sparsity_violation = sparsity_target - actual_sparsity
        constraint_loss = torch.clamp(sparsity_violation, min=0.0)

        loss = kl + constraint_loss
        loss.backward()

        snr_data = compute_layer_grad_snr(wrapper)
        snr_data["_step_loss"] = loss.item()
        snr_data["_step_kl"] = kl.item()
        snr_data["_step_sparsity"] = actual_sparsity.item()
        results_per_step.append(snr_data)

        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        if step % 10 == 0 or step == 1:
            avg_snr = sum(v["snr"] for k, v in snr_data.items() if k.startswith("layer_")) / 32
            print(f"  [KL step {step}/{num_steps}] loss={loss.item():.4f} kl={kl.item():.4f} "
                  f"sparsity={actual_sparsity.item():.3f} avg_snr={avg_snr:.6f}")

    wrapper.model.gradient_checkpointing_disable()
    wrapper.model.eval()
    return results_per_step


def run_bce_steps(wrapper, dataloader, device, num_steps, sparsity_target=0.5):
    """Run BCE training steps and record per-step gradient SNR."""
    bce_fn = BCESparsityLoss(sparsity_target=sparsity_target)
    params = list(wrapper.predictors.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)

    wrapper.predictors.train()
    data_iter = iter(dataloader)
    results_per_step = []

    for step in range(1, num_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        optimizer.zero_grad()

        wrapper.forward_dense(input_ids, capture_intermediates=True)
        intermediates = wrapper.get_intermediates()
        layer_inputs = wrapper.get_layer_inputs()

        total_bce = torch.tensor(0.0, device=device)
        num_layers = len(wrapper.predictors)

        for layer_idx in range(num_layers):
            if layer_idx not in intermediates:
                continue
            dense_act = intermediates[layer_idx]
            inp = layer_inputs[layer_idx]
            pred_logits = wrapper.predictors[layer_idx](inp)
            total_bce = total_bce + bce_fn(pred_logits, dense_act)

        loss = total_bce / max(num_layers, 1)
        loss.backward()

        snr_data = compute_layer_grad_snr(wrapper)
        snr_data["_step_loss"] = loss.item()
        results_per_step.append(snr_data)

        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        if step % 10 == 0 or step == 1:
            avg_snr = sum(v["snr"] for k, v in snr_data.items() if k.startswith("layer_")) / 32
            print(f"  [BCE step {step}/{num_steps}] loss={loss.item():.4f} avg_snr={avg_snr:.6f}")

    return results_per_step


def aggregate_results(step_results):
    """Aggregate per-step SNR into per-layer statistics."""
    layer_snrs = defaultdict(list)
    layer_norms = defaultdict(list)

    for step_data in step_results:
        for key, val in step_data.items():
            if key.startswith("layer_"):
                layer_snrs[key].append(val["snr"])
                layer_norms[key].append(val["grad_norm"])

    aggregated = {}
    all_means = []
    for layer_name in sorted(layer_snrs.keys(), key=lambda x: int(x.split("_")[1])):
        snrs = layer_snrs[layer_name]
        norms = layer_norms[layer_name]
        mean_snr = sum(snrs) / len(snrs)
        std_snr = (sum((s - mean_snr) ** 2 for s in snrs) / len(snrs)) ** 0.5
        aggregated[layer_name] = {
            "snr_mean": mean_snr,
            "snr_std": std_snr,
            "snr_per_step": snrs,
            "grad_norm_mean": sum(norms) / len(norms),
        }
        all_means.append(mean_snr)

    overall_avg = sum(all_means) / len(all_means) if all_means else 0.0
    return aggregated, overall_avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/models/llama-3.1-8b")
    parser.add_argument("--dry_run", action="store_true", help="Run only 2 steps for validation")
    args = parser.parse_args()

    if args.dry_run:
        args.num_steps = 2

    device = f"cuda:{args.gpu}"

    print(f"=== Gradient SNR Analysis: KL vs BCE ===")
    print(f"Steps: {args.num_steps}, Batch: {args.batch_size}, Seq: {args.seq_len}, Seed: {args.seed}")
    print(f"Device: {device}")

    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="flash_attention_2",
        local_files_only=True,
    )
    model.eval()

    print("Loading calibration data (C4 local)...")
    dataloader = get_calibration_loader(tokenizer, args.batch_size, args.seq_len, args.seed, "c4")

    # --- KL run ---
    print("\n" + "=" * 60)
    print("Phase 1: KL distillation gradient SNR")
    print("=" * 60)

    torch.manual_seed(args.seed)
    wrapper_kl = PredictorWrapper(model, bottleneck_size=128)
    wrapper_kl.predictors.to(device=device, dtype=torch.bfloat16)

    t0 = time.time()
    kl_steps = run_kl_steps(wrapper_kl, dataloader, device, args.num_steps)
    kl_time = time.time() - t0
    print(f"KL phase complete in {kl_time:.1f}s")

    # Save KL predictor state for reproducibility check
    kl_predictor_state = {k: v.clone() for k, v in wrapper_kl.predictors.state_dict().items()}
    del wrapper_kl
    torch.cuda.empty_cache()

    # --- BCE run ---
    print("\n" + "=" * 60)
    print("Phase 2: BCE activation-matching gradient SNR")
    print("=" * 60)

    torch.manual_seed(args.seed)
    wrapper_bce = PredictorWrapper(model, bottleneck_size=128)
    wrapper_bce.predictors.to(device=device, dtype=torch.bfloat16)

    # Reset dataloader to ensure same data sequence
    dataloader = get_calibration_loader(tokenizer, args.batch_size, args.seq_len, args.seed, "c4")

    t0 = time.time()
    bce_steps = run_bce_steps(wrapper_bce, dataloader, device, args.num_steps)
    bce_time = time.time() - t0
    print(f"BCE phase complete in {bce_time:.1f}s")

    del wrapper_bce
    torch.cuda.empty_cache()

    # --- Aggregate ---
    print("\n" + "=" * 60)
    print("Aggregating results...")
    print("=" * 60)

    kl_agg, kl_avg = aggregate_results(kl_steps)
    bce_agg, bce_avg = aggregate_results(bce_steps)

    ratio = kl_avg / (bce_avg + 1e-12)

    results = {
        "kl": kl_agg,
        "bce": bce_agg,
        "summary": {
            "kl_avg_snr": kl_avg,
            "bce_avg_snr": bce_avg,
            "ratio_kl_over_bce": ratio,
            "num_steps": args.num_steps,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "seed": args.seed,
            "kl_time_s": kl_time,
            "bce_time_s": bce_time,
        }
    }

    output_path = "/root/distill_sparse_swiglu/artifacts/gradient_snr_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  KL  avg gradient SNR: {kl_avg:.8f}")
    print(f"  BCE avg gradient SNR: {bce_avg:.8f}")
    print(f"  Ratio (KL/BCE):       {ratio:.4f}x")
    print()

    # Per-layer comparison
    print("  Per-layer SNR (selected):")
    print(f"  {'Layer':<10} {'KL SNR':<14} {'BCE SNR':<14} {'Ratio':<10}")
    print(f"  {'-'*48}")
    for i in list(range(5)) + list(range(27, 32)):
        k = f"layer_{i}"
        if k in kl_agg and k in bce_agg:
            kl_s = kl_agg[k]["snr_mean"]
            bce_s = bce_agg[k]["snr_mean"]
            r = kl_s / (bce_s + 1e-12)
            print(f"  {k:<10} {kl_s:<14.8f} {bce_s:<14.8f} {r:<10.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()

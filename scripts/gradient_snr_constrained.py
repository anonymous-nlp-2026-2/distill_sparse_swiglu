# Gradient SNR analysis v2: KL vs BCE with identical Lagrangian sparsity constraint.
# v1 used simplified constraint (KL: simple clamp, BCE: none), Reviewer flagged unfairness.
# v2 uses production Lagrangian (adaptive lambda, x2/x0.5 every 25 steps, cap 1000) on BOTH.
# Output: /root/distill_sparse_swiglu/artifacts/gradient_snr_constrained_results.json

import argparse
import json
import math
import os
import sys
import gc
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


def cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_layer_grad_snr(wrapper):
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


def run_kl_steps(wrapper, dataloader, device, num_steps, args):
    kl_fn = KLDistillLoss(temperature=args.kl_temperature)
    params = list(wrapper.predictors.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    scheduler = cosine_schedule_with_warmup(optimizer, args.warmup_steps, num_steps)

    wrapper.predictors.train()
    wrapper.model.train()
    wrapper.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    data_iter = iter(dataloader)
    results_per_step = []

    lambda_sparse = 1.0
    target_sparsity = args.sparsity_target

    for step in range(1, num_steps + 1):
        tau = args.tau_start + (args.tau_end - args.tau_start) * (step / num_steps)
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
        sparsity_violation = target_sparsity - actual_sparsity
        constraint_loss = lambda_sparse * torch.clamp(sparsity_violation, min=0.0)

        loss = kl + constraint_loss
        loss.backward()

        snr_data = compute_layer_grad_snr(wrapper)
        snr_data["_step_loss"] = loss.item()
        snr_data["_step_kl"] = kl.item()
        snr_data["_step_sparsity"] = actual_sparsity.item()
        snr_data["_step_lambda"] = lambda_sparse
        results_per_step.append(snr_data)

        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()

        # Lagrangian update every 25 steps (production config)
        if step % 25 == 0:
            avg_sp = actual_sparsity.item()
            old_lambda = lambda_sparse
            if avg_sp < target_sparsity - 0.02:
                lambda_sparse *= 2.0
            elif avg_sp > target_sparsity + 0.02:
                lambda_sparse *= 0.5
            lambda_sparse = max(0.1, min(lambda_sparse, args.lambda_max))
            if old_lambda != lambda_sparse:
                print(f"  [KL Lambda] step={step} sp={avg_sp:.3f} lambda: {old_lambda:.2f} -> {lambda_sparse:.2f}")

        if step % 10 == 0 or step == 1:
            avg_snr = sum(v["snr"] for k, v in snr_data.items() if k.startswith("layer_")) / 32
            print(f"  [KL step {step}/{num_steps}] loss={loss.item():.4f} kl={kl.item():.4f} "
                  f"constraint={constraint_loss.item():.4f} sparsity={actual_sparsity.item():.3f} "
                  f"lambda={lambda_sparse:.2f} tau={tau:.3f} avg_snr={avg_snr:.6f}")

    wrapper.model.gradient_checkpointing_disable()
    wrapper.model.eval()
    return results_per_step


def run_bce_steps(wrapper, dataloader, device, num_steps, args):
    bce_fn = BCESparsityLoss(sparsity_target=args.sparsity_target)
    params = list(wrapper.predictors.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    scheduler = cosine_schedule_with_warmup(optimizer, args.warmup_steps, num_steps)

    wrapper.predictors.train()
    data_iter = iter(dataloader)

    results_per_step = []
    lambda_sparse = 1.0
    target_sparsity = args.sparsity_target

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
        all_masks = {}
        num_layers = len(wrapper.predictors)

        for layer_idx in range(num_layers):
            if layer_idx not in intermediates:
                continue
            dense_act = intermediates[layer_idx]
            inp = layer_inputs[layer_idx]
            pred_logits = wrapper.predictors[layer_idx](inp)
            total_bce = total_bce + bce_fn(pred_logits, dense_act)
            hard_mask = (pred_logits > 0).float()
            soft_mask = torch.sigmoid(pred_logits)
            all_masks[layer_idx] = hard_mask - soft_mask.detach() + soft_mask

        total_bce = total_bce / max(num_layers, 1)

        # Same Lagrangian as KL
        actual_sparsity = 1.0 - torch.stack([m.mean() for m in all_masks.values()]).mean()
        sparsity_violation = target_sparsity - actual_sparsity
        constraint_loss = lambda_sparse * torch.clamp(sparsity_violation, min=0.0)

        loss = total_bce + constraint_loss
        loss.backward()

        snr_data = compute_layer_grad_snr(wrapper)
        snr_data["_step_loss"] = loss.item()
        snr_data["_step_bce"] = total_bce.item()
        snr_data["_step_sparsity"] = actual_sparsity.item()
        snr_data["_step_lambda"] = lambda_sparse
        results_per_step.append(snr_data)

        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()

        # Same Lagrangian update schedule as KL
        if step % 25 == 0:
            avg_sp = actual_sparsity.item()
            old_lambda = lambda_sparse
            if avg_sp < target_sparsity - 0.02:
                lambda_sparse *= 2.0
            elif avg_sp > target_sparsity + 0.02:
                lambda_sparse *= 0.5
            lambda_sparse = max(0.1, min(lambda_sparse, args.lambda_max))
            if old_lambda != lambda_sparse:
                print(f"  [BCE Lambda] step={step} sp={avg_sp:.3f} lambda: {old_lambda:.2f} -> {lambda_sparse:.2f}")

        if step % 10 == 0 or step == 1:
            avg_snr = sum(v["snr"] for k, v in snr_data.items() if k.startswith("layer_")) / 32
            print(f"  [BCE step {step}/{num_steps}] loss={loss.item():.4f} bce={total_bce.item():.4f} "
                  f"constraint={constraint_loss.item():.4f} sparsity={actual_sparsity.item():.3f} "
                  f"lambda={lambda_sparse:.2f} avg_snr={avg_snr:.6f}")

    return results_per_step


def aggregate_results(steps_data):
    layer_snrs = defaultdict(list)
    sparsity_curve = []

    for step_data in steps_data:
        sparsity_curve.append(step_data.get("_step_sparsity", 0.0))
        for k, v in step_data.items():
            if k.startswith("layer_"):
                layer_snrs[k].append(v["snr"])

    agg = {}
    all_means = []
    for k, vals in sorted(layer_snrs.items()):
        t = torch.tensor(vals)
        mean_val = t.mean().item()
        agg[k] = {"snr_mean": mean_val, "snr_std": t.std().item()}
        all_means.append(mean_val)

    avg_snr = sum(all_means) / len(all_means) if all_means else 0.0
    return agg, avg_snr, sparsity_curve


def main():
    parser = argparse.ArgumentParser(description="Gradient SNR v2: KL vs BCE with Lagrangian constraint")
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/models/llama-3.1-8b")
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--sparsity_target", type=float, default=0.5)
    parser.add_argument("--tau_start", type=float, default=1.0)
    parser.add_argument("--tau_end", type=float, default=0.1)
    parser.add_argument("--kl_temperature", type=float, default=1.0)
    parser.add_argument("--lambda_max", type=float, default=1000.0)
    args = parser.parse_args()

    if args.dry_run:
        args.num_steps = 2

    device = f"cuda:{args.gpu}"

    print(f"=== Gradient SNR v2: KL vs BCE (Lagrangian constrained) ===")
    print(f"Steps: {args.num_steps}, Batch: {args.batch_size}, Seq: {args.seq_len}, Seed: {args.seed}")
    print(f"Device: {device}, LR: {args.lr}, Warmup: {args.warmup_steps}")
    print(f"Sparsity target: {args.sparsity_target}, Lambda max: {args.lambda_max}")
    print(f"Tau: {args.tau_start} -> {args.tau_end}")
    print()

    print("Loading tokenizer and model...")
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
    print("Phase 1: KL distillation + Lagrangian")
    print("=" * 60)

    torch.manual_seed(args.seed)
    wrapper_kl = PredictorWrapper(model, bottleneck_size=128)
    wrapper_kl.predictors.to(device=device, dtype=torch.bfloat16)

    t0 = time.time()
    kl_steps = run_kl_steps(wrapper_kl, dataloader, device, args.num_steps, args)
    kl_time = time.time() - t0
    print(f"KL phase complete in {kl_time:.1f}s")

    del wrapper_kl
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # --- BCE run ---
    print("\n" + "=" * 60)
    print("Phase 2: BCE activation-matching + Lagrangian")
    print("=" * 60)

    torch.manual_seed(args.seed)
    wrapper_bce = PredictorWrapper(model, bottleneck_size=128)
    wrapper_bce.predictors.to(device=device, dtype=torch.bfloat16)

    dataloader = get_calibration_loader(tokenizer, args.batch_size, args.seq_len, args.seed, "c4")

    t0 = time.time()
    bce_steps = run_bce_steps(wrapper_bce, dataloader, device, args.num_steps, args)
    bce_time = time.time() - t0
    print(f"BCE phase complete in {bce_time:.1f}s")

    del wrapper_bce
    gc.collect()
    torch.cuda.empty_cache()

    # --- Aggregate ---
    print("\n" + "=" * 60)
    print("Aggregating results...")
    print("=" * 60)

    kl_agg, kl_avg, kl_sparsity_curve = aggregate_results(kl_steps)
    bce_agg, bce_avg, bce_sparsity_curve = aggregate_results(bce_steps)

    ratio = kl_avg / (bce_avg + 1e-12)

    # Sparsity curves sampled every 10 steps
    kl_sp_sampled = [kl_sparsity_curve[i] for i in range(0, len(kl_sparsity_curve), 10)]
    bce_sp_sampled = [bce_sparsity_curve[i] for i in range(0, len(bce_sparsity_curve), 10)]

    results = {
        "kl": {
            "per_layer_snr": {k: v["snr_mean"] for k, v in kl_agg.items()},
            "avg_snr": kl_avg,
            "sparsity_curve": kl_sp_sampled,
            "final_sparsity": kl_sparsity_curve[-1] if kl_sparsity_curve else 0.0,
        },
        "bce": {
            "per_layer_snr": {k: v["snr_mean"] for k, v in bce_agg.items()},
            "avg_snr": bce_avg,
            "sparsity_curve": bce_sp_sampled,
            "final_sparsity": bce_sparsity_curve[-1] if bce_sparsity_curve else 0.0,
        },
        "summary": {
            "kl_avg_snr": kl_avg,
            "bce_avg_snr": bce_avg,
            "ratio": ratio,
            "kl_final_sparsity": kl_sparsity_curve[-1] if kl_sparsity_curve else 0.0,
            "bce_final_sparsity": bce_sparsity_curve[-1] if bce_sparsity_curve else 0.0,
            "num_steps": args.num_steps,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "seed": args.seed,
            "lr": args.lr,
            "warmup_steps": args.warmup_steps,
            "sparsity_target": args.sparsity_target,
            "lambda_max": args.lambda_max,
            "kl_time_s": kl_time,
            "bce_time_s": bce_time,
        },
        "v1_comparison": {
            "v1_kl_avg_snr": 5.77e-05,
            "v1_bce_avg_snr": 1.00e-05,
            "v1_ratio": 5.76,
            "v1_note": "v1 had no Lagrangian (KL sparsity ~0.36, BCE sparsity uncontrolled)",
        },
    }

    output_path = "/root/distill_sparse_swiglu/artifacts/gradient_snr_constrained_results.json"
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
    print(f"  KL  final sparsity:   {kl_sparsity_curve[-1]:.3f}")
    print(f"  BCE final sparsity:   {bce_sparsity_curve[-1]:.3f}")
    print()

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

    print()
    print("  Sparsity convergence (every 10 steps):")
    print(f"  {'Step':<8} {'KL sp':<10} {'BCE sp':<10}")
    for i, (ks, bs) in enumerate(zip(kl_sp_sampled, bce_sp_sampled)):
        print(f"  {i*10+1:<8} {ks:<10.3f} {bs:<10.3f}")

    print()
    print("  vs v1 (no Lagrangian):")
    print(f"    v1 ratio: 5.76x, v2 ratio: {ratio:.4f}x")
    print()
    print("Done.")


if __name__ == "__main__":
    main()

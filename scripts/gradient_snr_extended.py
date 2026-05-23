# Extended Gradient SNR analysis: KL vs BCE, 300 steps, SNR snapshots with 95% CI.
# Responds to Reviewer R6-W3: extend SNR analysis beyond 200 steps, add CI + BCE loss curve.
# Output: /root/distill_sparse_swiglu/artifacts/snr_full_1000steps_results.json

import argparse
import json
import math
import os
import sys
import gc
import time
from collections import defaultdict

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
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
        snr_per_layer[f"layer_{i}"] = snr.item()
    return snr_per_layer


def compute_snr_snapshot(snr_per_layer):
    """Compute mean + 95% CI across layers."""
    vals = list(snr_per_layer.values())
    if not vals:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0, "n": 0}
    t = torch.tensor(vals)
    mean = t.mean().item()
    std = t.std().item()
    n = len(vals)
    ci_half = 1.96 * std / math.sqrt(n)
    return {
        "mean": mean,
        "ci_lower": mean - ci_half,
        "ci_upper": mean + ci_half,
        "std": std,
        "n": n,
        "per_layer": snr_per_layer,
    }


SNAPSHOT_STEPS = [0, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000]


def run_kl_extended(wrapper, dataloader, device, num_steps, args):
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
    snapshots = {}
    loss_curve = []
    lambda_sparse = 1.0
    target_sparsity = args.sparsity_target

    def get_batch():
        nonlocal data_iter
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        return batch["input_ids"].to(device)

    def do_forward_backward(input_ids):
        nonlocal lambda_sparse
        tau = args.tau_start + (args.tau_end - args.tau_start) * (step / num_steps) if step > 0 else args.tau_start
        wrapper.gumbel_mask.tau = tau

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
        return loss.item(), kl.item(), actual_sparsity.item()

    # Step 0 snapshot: forward+backward without optimizer step
    step = 0
    input_ids = get_batch()
    loss_val, kl_val, sp_val = do_forward_backward(input_ids)
    snr_data = compute_layer_grad_snr(wrapper)
    snapshots[0] = compute_snr_snapshot(snr_data)
    snapshots[0]["loss"] = loss_val
    snapshots[0]["sparsity"] = sp_val
    print(f"  [KL snapshot step=0] SNR mean={snapshots[0]['mean']:.8f} CI=[{snapshots[0]['ci_lower']:.8f}, {snapshots[0]['ci_upper']:.8f}]")

    for step in range(1, num_steps + 1):
        tau = args.tau_start + (args.tau_end - args.tau_start) * (step / num_steps)
        wrapper.gumbel_mask.tau = tau

        input_ids = get_batch()
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

        if step in SNAPSHOT_STEPS:
            snr_data = compute_layer_grad_snr(wrapper)
            snap = compute_snr_snapshot(snr_data)
            snap["loss"] = loss.item()
            snap["sparsity"] = actual_sparsity.item()
            snapshots[step] = snap
            print(f"  [KL snapshot step={step}] SNR mean={snap['mean']:.8f} CI=[{snap['ci_lower']:.8f}, {snap['ci_upper']:.8f}]")

        if step % 10 == 0:
            loss_curve.append({"step": step, "loss": loss.item(), "kl": kl.item(), "sparsity": actual_sparsity.item()})

        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()

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

        if step % 50 == 0:
            print(f"  [KL step {step}/{num_steps}] loss={loss.item():.4f} kl={kl.item():.4f} sp={actual_sparsity.item():.3f} lambda={lambda_sparse:.2f}")

    wrapper.model.gradient_checkpointing_disable()
    wrapper.model.eval()
    return snapshots, loss_curve


def run_bce_extended(wrapper, dataloader, device, num_steps, args):
    bce_fn = BCESparsityLoss(sparsity_target=args.sparsity_target)
    params = list(wrapper.predictors.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    scheduler = cosine_schedule_with_warmup(optimizer, args.warmup_steps, num_steps)

    wrapper.predictors.train()
    data_iter = iter(dataloader)
    snapshots = {}
    loss_curve = []
    lambda_sparse = 1.0
    target_sparsity = args.sparsity_target
    num_layers = len(wrapper.predictors)

    def get_batch():
        nonlocal data_iter
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        return batch["input_ids"].to(device)

    def do_bce_forward_backward(input_ids):
        nonlocal lambda_sparse
        optimizer.zero_grad()
        wrapper.forward_dense(input_ids, capture_intermediates=True)
        intermediates = wrapper.get_intermediates()
        layer_inputs = wrapper.get_layer_inputs()

        total_bce = torch.tensor(0.0, device=device)
        all_masks = {}

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
        actual_sparsity = 1.0 - torch.stack([m.mean() for m in all_masks.values()]).mean()
        sparsity_violation = target_sparsity - actual_sparsity
        constraint_loss = lambda_sparse * torch.clamp(sparsity_violation, min=0.0)

        loss = total_bce + constraint_loss
        loss.backward()
        return loss.item(), total_bce.item(), actual_sparsity.item()

    # Step 0 snapshot
    step = 0
    input_ids = get_batch()
    loss_val, bce_val, sp_val = do_bce_forward_backward(input_ids)
    snr_data = compute_layer_grad_snr(wrapper)
    snapshots[0] = compute_snr_snapshot(snr_data)
    snapshots[0]["loss"] = loss_val
    snapshots[0]["bce"] = bce_val
    snapshots[0]["sparsity"] = sp_val
    print(f"  [BCE snapshot step=0] SNR mean={snapshots[0]['mean']:.8f} CI=[{snapshots[0]['ci_lower']:.8f}, {snapshots[0]['ci_upper']:.8f}]")

    for step in range(1, num_steps + 1):
        input_ids = get_batch()
        optimizer.zero_grad()

        wrapper.forward_dense(input_ids, capture_intermediates=True)
        intermediates = wrapper.get_intermediates()
        layer_inputs = wrapper.get_layer_inputs()

        total_bce = torch.tensor(0.0, device=device)
        all_masks = {}

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
        actual_sparsity = 1.0 - torch.stack([m.mean() for m in all_masks.values()]).mean()
        sparsity_violation = target_sparsity - actual_sparsity
        constraint_loss = lambda_sparse * torch.clamp(sparsity_violation, min=0.0)

        loss = total_bce + constraint_loss
        loss.backward()

        if step in SNAPSHOT_STEPS:
            snr_data = compute_layer_grad_snr(wrapper)
            snap = compute_snr_snapshot(snr_data)
            snap["loss"] = loss.item()
            snap["bce"] = total_bce.item()
            snap["sparsity"] = actual_sparsity.item()
            snapshots[step] = snap
            print(f"  [BCE snapshot step={step}] SNR mean={snap['mean']:.8f} CI=[{snap['ci_lower']:.8f}, {snap['ci_upper']:.8f}]")

        if step % 10 == 0:
            loss_curve.append({"step": step, "loss": loss.item(), "bce": total_bce.item(), "sparsity": actual_sparsity.item()})

        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()

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

        if step % 50 == 0:
            print(f"  [BCE step {step}/{num_steps}] loss={loss.item():.4f} bce={total_bce.item():.4f} sp={actual_sparsity.item():.3f} lambda={lambda_sparse:.2f}")

    return snapshots, loss_curve


def main():
    parser = argparse.ArgumentParser(description="Extended SNR analysis: 300 steps, snapshots with 95% CI")
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/models/llama-3.1-8b")
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--sparsity_target", type=float, default=0.3)
    parser.add_argument("--tau_start", type=float, default=1.0)
    parser.add_argument("--tau_end", type=float, default=0.1)
    parser.add_argument("--kl_temperature", type=float, default=1.0)
    parser.add_argument("--lambda_max", type=float, default=5000.0)
    args = parser.parse_args()

    if args.dry_run:
        args.num_steps = 3
        global SNAPSHOT_STEPS
        SNAPSHOT_STEPS = [0, 1, 2, 3]

    device = f"cuda:{args.gpu}"

    print(f"=== Extended Gradient SNR: KL vs BCE (R6-W3 response) ===")
    print(f"Steps: {args.num_steps}, Batch: {args.batch_size}, Seq: {args.seq_len}, Seed: {args.seed}")
    print(f"Device: {device}, LR: {args.lr}, Warmup: {args.warmup_steps}")
    print(f"Sparsity target: {args.sparsity_target}, Lambda max: {args.lambda_max}")
    print(f"Snapshot steps: {SNAPSHOT_STEPS}")
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

    # --- KL phase ---
    print("\n" + "=" * 60)
    print("Phase 1: KL distillation (300 steps)")
    print("=" * 60)

    torch.manual_seed(args.seed)
    wrapper_kl = PredictorWrapper(model, bottleneck_size=128)
    wrapper_kl.predictors.to(device=device, dtype=torch.bfloat16)

    t0 = time.time()
    kl_snapshots, kl_loss_curve = run_kl_extended(wrapper_kl, dataloader, device, args.num_steps, args)
    kl_time = time.time() - t0
    print(f"KL phase complete in {kl_time:.1f}s")

    del wrapper_kl
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # --- BCE phase ---
    print("\n" + "=" * 60)
    print("Phase 2: BCE activation-matching (300 steps)")
    print("=" * 60)

    torch.manual_seed(args.seed)
    wrapper_bce = PredictorWrapper(model, bottleneck_size=128)
    wrapper_bce.predictors.to(device=device, dtype=torch.bfloat16)

    dataloader = get_calibration_loader(tokenizer, args.batch_size, args.seq_len, args.seed, "c4")

    t0 = time.time()
    bce_snapshots, bce_loss_curve = run_bce_extended(wrapper_bce, dataloader, device, args.num_steps, args)
    bce_time = time.time() - t0
    print(f"BCE phase complete in {bce_time:.1f}s")

    del wrapper_bce
    gc.collect()
    torch.cuda.empty_cache()

    # --- Assemble results ---
    print("\n" + "=" * 60)
    print("Assembling results...")
    print("=" * 60)

    # Convert int keys to str for JSON
    kl_snap_out = {}
    for k, v in kl_snapshots.items():
        kl_snap_out[str(k)] = v
    bce_snap_out = {}
    for k, v in bce_snapshots.items():
        bce_snap_out[str(k)] = v

    results = {
        "experiment": "snr_extended_r6w3",
        "config": {
            "num_steps": args.num_steps,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "seed": args.seed,
            "lr": args.lr,
            "warmup_steps": args.warmup_steps,
            "sparsity_target": args.sparsity_target,
            "lambda_max": args.lambda_max,
            "kl_temperature": args.kl_temperature,
            "snapshot_steps": SNAPSHOT_STEPS,
        },
        "kl": {
            "snapshots": kl_snap_out,
            "loss_curve": kl_loss_curve,
            "time_s": kl_time,
        },
        "bce": {
            "snapshots": bce_snap_out,
            "loss_curve": bce_loss_curve,
            "time_s": bce_time,
        },
        "summary": {
            "kl_final_snr_mean": kl_snapshots.get(args.num_steps, {}).get("mean", 0.0),
            "kl_final_snr_ci": [
                kl_snapshots.get(args.num_steps, {}).get("ci_lower", 0.0),
                kl_snapshots.get(args.num_steps, {}).get("ci_upper", 0.0),
            ],
            "bce_final_snr_mean": bce_snapshots.get(args.num_steps, {}).get("mean", 0.0),
            "bce_final_snr_ci": [
                bce_snapshots.get(args.num_steps, {}).get("ci_lower", 0.0),
                bce_snapshots.get(args.num_steps, {}).get("ci_upper", 0.0),
            ],
            "ratio_at_final": (
                kl_snapshots.get(args.num_steps, {}).get("mean", 0.0) /
                (bce_snapshots.get(args.num_steps, {}).get("mean", 1e-12) + 1e-12)
            ),
            "kl_time_s": kl_time,
            "bce_time_s": bce_time,
        },
    }

    output_path = "/root/distill_sparse_swiglu/artifacts/snr_full_1000steps_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY: SNR over training")
    print("=" * 60)
    print(f"  {'Step':<6} {'KL SNR (mean)':<16} {'KL 95% CI':<28} {'BCE SNR (mean)':<16} {'BCE 95% CI':<28}")
    print(f"  {'-'*94}")
    for s in SNAPSHOT_STEPS:
        kl_s = kl_snapshots.get(s, {})
        bce_s = bce_snapshots.get(s, {})
        kl_m = kl_s.get("mean", 0.0)
        kl_lo = kl_s.get("ci_lower", 0.0)
        kl_hi = kl_s.get("ci_upper", 0.0)
        bce_m = bce_s.get("mean", 0.0)
        bce_lo = bce_s.get("ci_lower", 0.0)
        bce_hi = bce_s.get("ci_upper", 0.0)
        print(f"  {s:<6} {kl_m:<16.8f} [{kl_lo:.8f}, {kl_hi:.8f}]  {bce_m:<16.8f} [{bce_lo:.8f}, {bce_hi:.8f}]")

    print(f"\n  Final ratio (KL/BCE at step {args.num_steps}): {results['summary']['ratio_at_final']:.4f}x")
    print(f"\n  BCE loss curve (sampled every 50 steps):")
    for entry in bce_loss_curve:
        if entry["step"] % 50 == 0:
            print(f"    step={entry['step']:>3} bce={entry['bce']:.4f} sp={entry['sparsity']:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()

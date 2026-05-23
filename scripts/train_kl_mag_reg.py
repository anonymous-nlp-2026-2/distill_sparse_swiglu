# Magnitude-preservation regularization for KL sparsity predictor.
# Adds penalty for pruning high-magnitude neurons: loss = KL + lagrangian + alpha * mean((1-mask) * importance)

import argparse
import importlib.util
import math
import os
import sys
import time
import json

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from data_utils import get_calibration_loader, get_eval_dataset
from losses import KLDistillLoss
from predictor import PredictorWrapper


def _best_attn_impl():
    if importlib.util.find_spec("flash_attn"):
        return "flash_attention_2"
    return "sdpa"


def cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def compute_neuron_importance(wrapper, dataloader, device, num_batches=50):
    """Compute per-layer mean |activation| for each intermediate neuron."""
    num_layers = len(wrapper.predictors)
    accum = [None] * num_layers
    counts = [0] * num_layers

    wrapper.sparse_mode = False
    wrapper.capture_intermediates = True

    data_iter = iter(dataloader)
    for batch_idx in range(num_batches):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        wrapper._layer_intermediates.clear()
        wrapper._layer_inputs.clear()
        wrapper.model(input_ids=input_ids)

        for layer_idx in range(num_layers):
            if layer_idx not in wrapper._layer_intermediates:
                continue
            intermediate = wrapper._layer_intermediates[layer_idx]
            # mean over batch and seq dims -> [intermediate_size]
            mag = intermediate.abs().float().mean(dim=(0, 1))
            if accum[layer_idx] is None:
                accum[layer_idx] = mag
            else:
                accum[layer_idx] += mag
            counts[layer_idx] += 1

        if (batch_idx + 1) % 10 == 0:
            print(f"  Importance computation: {batch_idx+1}/{num_batches} batches")

    wrapper.sparse_mode = True
    wrapper.capture_intermediates = False

    importance = {}
    for i in range(num_layers):
        if accum[i] is not None and counts[i] > 0:
            imp = accum[i] / counts[i]
            # Normalize per layer to [0, 1]
            imp = imp / (imp.max() + 1e-8)
            importance[i] = imp.to(device=device, dtype=torch.bfloat16)

    return importance


@torch.no_grad()
def evaluate_ppl_teal_global(wrapper, model, tokenizer, device, sparsity_target=0.5):
    """Evaluate PPL using teal_global_masks protocol (same as paper results)."""
    from evaluate import teal_global_masks, evaluate_perplexity, get_eval_dataset as _unused
    from data_utils import get_eval_dataset

    examples = get_eval_dataset("wikitext2", tokenizer, 2048, max_samples=200)
    calib_examples = get_eval_dataset("wikitext2", tokenizer, 2048, max_samples=32)
    calib_ids = torch.stack([ex["input_ids"] for ex in calib_examples]).to(device)

    wrapper.predictors.eval()
    global_masks = teal_global_masks(wrapper, calib_ids, sparsity_target)
    ppl = evaluate_perplexity(model, examples, device, global_masks=global_masks)
    wrapper.predictors.train()
    return ppl


def train_kl_mag_reg(wrapper, dataloader, args, device, neuron_importance):
    """KL training with magnitude-preservation regularization."""
    kl_fn = KLDistillLoss(temperature=args.kl_temperature)

    params = list(wrapper.predictors.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    scheduler = cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.num_steps)

    wrapper.predictors.train()
    wrapper.model.train()
    wrapper.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    print("Gradient checkpointing enabled")

    data_iter = iter(dataloader)
    ga = args.gradient_accumulation_steps

    lambda_sparse = 1.0
    target_sparsity = args.sparsity_target
    margin = 0.02
    print(f"KL + MagReg: alpha={args.alpha}, lambda={lambda_sparse}, target={target_sparsity}, "
          f"margin={margin}, cap={args.lambda_max}")

    t0 = time.time()
    for opt_step in range(1, args.num_steps + 1):
        tau = args.tau_start + (args.tau_end - args.tau_start) * (opt_step / args.num_steps)
        wrapper.gumbel_mask.tau = tau

        accum_loss = 0.0
        accum_kl = 0.0
        accum_constraint = 0.0
        accum_mag_reg = 0.0

        for _ in range(ga):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            dense_logits = wrapper.forward_dense(input_ids).logits.detach()
            sparse_logits = wrapper.forward_sparse(input_ids).logits

            kl = kl_fn(dense_logits, sparse_logits)

            masks = wrapper.get_masks()
            actual_sparsity = 1.0 - torch.stack([m.mean() for m in masks.values()]).mean()

            # Soft Lagrangian constraint
            sparsity_error = torch.abs(actual_sparsity - target_sparsity)
            constraint_loss = lambda_sparse * torch.clamp(sparsity_error - margin, min=0.0) ** 2

            # Magnitude-preservation regularization
            # Penalize pruning (1-mask) of high-importance neurons
            mag_reg = torch.tensor(0.0, device=device)
            for layer_idx, mask in masks.items():
                if layer_idx in neuron_importance:
                    imp = neuron_importance[layer_idx]
                    prune_prob = 1.0 - mask  # [batch, seq, intermediate]
                    # mean over batch, seq; weighted by importance
                    layer_penalty = (prune_prob * imp.unsqueeze(0).unsqueeze(0)).mean()
                    mag_reg = mag_reg + layer_penalty
            mag_reg = mag_reg / len(masks)

            loss = (kl + constraint_loss + args.alpha * mag_reg) / ga
            loss.backward()

            accum_loss += loss.item()
            accum_kl += kl.item() / ga
            accum_constraint += constraint_loss.item() / ga
            accum_mag_reg += mag_reg.item() / ga

        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        masks = wrapper.get_masks()
        avg_sparsity = (
            1.0 - torch.stack([m.mean() for m in masks.values()]).mean().item()
            if masks else 0.0
        )

        # Lagrangian update (every step, aggressive)
        old_lambda = lambda_sparse
        if avg_sparsity < target_sparsity - margin:
            lambda_sparse *= 2.0
        elif avg_sparsity > target_sparsity + margin:
            lambda_sparse *= 0.5
        lambda_sparse = max(0.01, min(lambda_sparse, args.lambda_max))

        if opt_step % 10 == 0 or opt_step == 1:
            elapsed = time.time() - t0
            print(
                f"[Step {opt_step}/{args.num_steps}] loss={accum_loss:.6f} "
                f"kl={accum_kl:.4f} constraint={accum_constraint:.6f} "
                f"mag_reg={accum_mag_reg:.4f} sparsity={avg_sparsity:.3f} "
                f"lambda={lambda_sparse:.2f} tau={tau:.3f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} elapsed={elapsed:.0f}s"
            )

    elapsed = time.time() - t0
    print(f"Training done in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    return avg_sparsity


def main():
    p = argparse.ArgumentParser(description="KL + Magnitude-Preservation Regularization")
    p.add_argument("--model_name_or_path", type=str, default="/root/autodl-tmp/models/llama-3.1-8b")
    p.add_argument("--num_steps", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sparsity_target", type=float, default=0.5)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--alpha", type=float, required=True, help="Magnitude reg weight")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--tau_start", type=float, default=1.0)
    p.add_argument("--tau_end", type=float, default=0.1)
    p.add_argument("--kl_temperature", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--lambda_max", type=float, default=1000.0)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--bottleneck_dim", type=int, default=128)
    p.add_argument("--importance_batches", type=int, default=50)
    p.add_argument("--eval_after", action="store_true", default=True)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    device = f"cuda:{args.gpu}"
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=== KL + MagReg Training (alpha={args.alpha}) ===")
    print(f"Device: {device}, Steps: {args.num_steps}, Seed: {args.seed}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation=_best_attn_impl(),
    )
    model.eval()

    wrapper = PredictorWrapper(model, bottleneck_size=args.bottleneck_dim)
    wrapper.predictors.to(device=device, dtype=torch.bfloat16)

    dataloader = get_calibration_loader(tokenizer, args.batch_size, args.seq_len, args.seed, "wikitext-103")

    trainable = sum(p.numel() for p in wrapper.predictors.parameters())
    print(f"Trainable params: {trainable:,}")

    # Phase 1: Compute neuron importance
    print("\n--- Phase 1: Computing neuron importance ---")
    neuron_importance = compute_neuron_importance(
        wrapper, dataloader, device, num_batches=args.importance_batches
    )
    print(f"Computed importance for {len(neuron_importance)} layers")
    for i in [0, 15, 31]:
        if i in neuron_importance:
            imp = neuron_importance[i]
            print(f"  Layer {i}: mean={imp.mean():.4f}, max={imp.max():.4f}, min={imp.min():.4f}")

    if args.dry_run:
        print("Dry run: exiting after importance computation + 2 steps")
        args.num_steps = 2

    # Phase 2: Train with magnitude regularization
    print(f"\n--- Phase 2: Training ({args.num_steps} steps) ---")
    # Re-create dataloader to reset iterator
    dataloader = get_calibration_loader(tokenizer, args.batch_size, args.seq_len, args.seed, "wikitext-103")
    final_sparsity = train_kl_mag_reg(wrapper, dataloader, args, device, neuron_importance)

    # Save checkpoint
    ckpt_path = os.path.join(args.output_dir, "predictor_kl_mag_reg.pt")
    torch.save({
        "predictors": wrapper.predictors.state_dict(),
        "args": vars(args),
        "neuron_importance": {k: v.cpu() for k, v in neuron_importance.items()},
    }, ckpt_path)
    print(f"Saved checkpoint -> {ckpt_path}")

    # Phase 3: Evaluate PPL
    if args.eval_after and not args.dry_run:
        print("\n--- Phase 3: Evaluating PPL (teal_global protocol) ---")
        ppl = evaluate_ppl_teal_global(wrapper, model, tokenizer, device, args.sparsity_target)
        print(f"WikiText-2 PPL (teal_global, sparsity={args.sparsity_target}): {ppl:.4f}")

        results = {
            "alpha": args.alpha,
            "ppl": ppl,
            "final_sparsity": final_sparsity,
            "steps": args.num_steps,
            "seed": args.seed,
        }
        results_path = os.path.join(args.output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved -> {results_path}")


if __name__ == "__main__":
    main()

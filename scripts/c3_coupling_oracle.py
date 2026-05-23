"""C3 Coupling vs Oracle: Train BCE predictor with KL predictor's soft targets.

Tests whether KL's advantage comes from gradient coupling (hypothesis B)
or from better oracle targets (hypothesis A).

If BCE+KL-oracle ≈ KL → Oracle quality is the main factor (hypothesis A).
If BCE+KL-oracle << KL → Gradient coupling is key (hypothesis B).
"""
import argparse
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from data_utils import get_calibration_loader, get_random_loader
from predictor import SparsityPredictor


def cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def parse_args():
    p = argparse.ArgumentParser(description="C3: BCE with KL-oracle soft targets")
    p.add_argument("--model_name_or_path", type=str,
                   default="/root/autodl-tmp/models/llama-3.1-8b")
    p.add_argument("--kl_checkpoint", type=str,
                   default="/root/distill_sparse_swiglu/checkpoints/mvp_kl_norm_v2_s42/predictor_kl_normalized.pt")
    p.add_argument("--num_steps", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sparsity_target", type=float, default=0.5)
    p.add_argument("--sparsity_reg_weight", type=float, default=0.1)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output_dir", type=str,
                   default="/root/distill_sparse_swiglu/checkpoints/c3_coupling_oracle_s42")
    p.add_argument("--wandb_project", type=str, default="distill_sparse_swiglu")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--bottleneck_dim", type=int, default=128)
    p.add_argument("--dataset", type=str, default="wikitext-103",
                   choices=["wikitext-103", "c4"])
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--use_random_data", action="store_true",
                   help="Use random tokens instead of real data (for dry-run)")
    p.add_argument("--dry_run_steps", type=int, default=0,
                   help="If >0, override num_steps for quick validation")
    return p.parse_args()


def main():
    args = parse_args()
    if args.dry_run_steps > 0:
        args.num_steps = args.dry_run_steps

    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")

    run_name = args.wandb_run_name or f"c3_bce_kl_oracle_s{args.seed}"
    wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    print(f"Loading model from {args.model_name_or_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    ).to(device).eval()

    config = model.config
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    print(f"Model: {num_layers} layers, hidden={hidden_size}, intermediate={intermediate_size}")

    # Load frozen KL oracle predictor
    print(f"Loading KL oracle from {args.kl_checkpoint} ...")
    kl_ckpt = torch.load(args.kl_checkpoint, map_location=device, weights_only=True)
    oracle_predictors = torch.nn.ModuleList([
        SparsityPredictor(hidden_size, intermediate_size, args.bottleneck_dim)
        for _ in range(num_layers)
    ]).to(device)
    oracle_predictors.load_state_dict(kl_ckpt["predictors"])
    oracle_predictors.half().eval()
    for p in oracle_predictors.parameters():
        p.requires_grad = False
    print("KL oracle loaded and frozen")

    # Trainable student predictor (fresh init)
    student_predictors = torch.nn.ModuleList([
        SparsityPredictor(hidden_size, intermediate_size, args.bottleneck_dim)
        for _ in range(num_layers)
    ]).to(device)
    student_predictors.train()

    # Hook to capture MLP inputs
    layer_inputs = {}

    def make_hook(layer_idx):
        def hook_fn(module, args_tuple, output):
            layer_inputs[layer_idx] = args_tuple[0].detach()
        return hook_fn

    hooks = []
    for li, layer in enumerate(model.model.layers):
        h = layer.mlp.register_forward_hook(make_hook(li))
        hooks.append(h)

    # Data
    if args.use_random_data:
        dataloader = get_random_loader(config.vocab_size, args.batch_size, args.seq_len, args.seed)
    else:
        dataloader = get_calibration_loader(
            tokenizer, args.batch_size, args.seq_len, args.seed, args.dataset)

    # Optimizer
    params = list(student_predictors.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    scheduler = cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.num_steps)

    ga = args.gradient_accumulation_steps
    data_iter = iter(dataloader)
    t0 = time.time()

    print(f"\nTraining: {args.num_steps} steps, bs={args.batch_size}, ga={ga}, "
          f"lr={args.lr}, warmup={args.warmup_steps}")
    print(f"Sparsity target={args.sparsity_target}, reg_weight={args.sparsity_reg_weight}")
    print("=" * 60)

    for opt_step in range(1, args.num_steps + 1):
        accum_loss = 0.0
        accum_bce = 0.0
        accum_reg = 0.0
        last_sparsity = 0.0

        for _ in range(ga):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            layer_inputs.clear()

            with torch.no_grad():
                model(input_ids=input_ids)

            total_bce = torch.tensor(0.0, device=device)
            all_masks = {}

            for li in range(num_layers):
                if li not in layer_inputs:
                    continue
                inp = layer_inputs[li]  # (batch, seq, hidden_size), detached

                # Oracle: KL predictor's soft probability (frozen, no grad)
                with torch.no_grad():
                    oracle_logits = oracle_predictors[li](inp.half())
                    soft_target = torch.sigmoid(oracle_logits).float()

                # Student: trainable predictor
                student_logits = student_predictors[li](inp.float())

                # BCE loss with soft target
                bce = F.binary_cross_entropy_with_logits(student_logits, soft_target)
                total_bce = total_bce + bce

                # Track mask for sparsity regularization
                with torch.no_grad():
                    hard_mask = (student_logits > 0).float()
                    all_masks[li] = hard_mask

            total_bce = total_bce / max(num_layers, 1)

            # Sparsity regularizer (same as BCE baseline)
            if all_masks:
                density = torch.stack([(m > 0.5).float().mean() for m in all_masks.values()]).mean()
                reg = args.sparsity_reg_weight * torch.abs(density - (1.0 - args.sparsity_target))
            else:
                reg = torch.tensor(0.0, device=device)

            loss = (total_bce + reg) / ga
            loss.backward()

            accum_loss += loss.item()
            accum_bce += total_bce.item() / ga
            accum_reg += reg.item() / ga

            with torch.no_grad():
                last_sparsity = (
                    1.0 - torch.stack([(m > 0.5).float().mean() for m in all_masks.values()]).mean().item()
                    if all_masks else 0.0
                )

        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        elapsed = time.time() - t0
        log_dict = {
            "loss": accum_loss, "bce_loss": accum_bce, "reg_loss": accum_reg,
            "sparsity": last_sparsity, "lr": scheduler.get_last_lr()[0],
            "wall_time": elapsed, "step_time": elapsed / opt_step,
        }
        wandb.log(log_dict, step=opt_step)

        if opt_step % args.log_every == 0 or opt_step == 1:
            print(
                f"[Step {opt_step}/{args.num_steps}] loss={accum_loss:.4f} "
                f"bce={accum_bce:.4f} reg={accum_reg:.4f} "
                f"sparsity={last_sparsity:.3f} lr={scheduler.get_last_lr()[0]:.2e} "
                f"elapsed={elapsed:.1f}s ({elapsed/opt_step:.2f}s/step)"
            )

    # Cleanup hooks
    for h in hooks:
        h.remove()

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "predictor_bce_kl_oracle.pt")
    torch.save({
        "predictors": student_predictors.state_dict(),
        "args": vars(args),
        "kl_checkpoint_used": args.kl_checkpoint,
    }, ckpt_path)
    total_time = time.time() - t0
    print(f"\nTraining complete: {args.num_steps} steps in {total_time:.1f}s "
          f"({total_time/args.num_steps:.2f}s/step)")
    print(f"Saved checkpoint -> {ckpt_path}")

    pred_params = sum(p.numel() for p in student_predictors.parameters())
    print(f"Predictor params: {pred_params:,} ({pred_params * 4 / 1024**2:.1f} MB in fp32)")

    wandb.finish()


if __name__ == "__main__":
    main()

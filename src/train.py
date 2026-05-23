# Training script for SwiGLU sparsity predictors (KL distillation, BCE, or BCE+compensation).
# Supports --relaxation gumbel (default, Gumbel-sigmoid with tau annealing) or ste (Straight-Through Estimator).
# Input: frozen Llama 3.1 8B + calibration data (wikitext-103 or c4)
# Output: predictor checkpoint in output_dir
#
# Usage:
#   python train.py --loss_type kl --num_steps 1000 --gpu 0
#   python train.py --loss_type kl --batch_size 2 --gradient_accumulation_steps 2 --num_steps 1000 --gpu 0
#   python train.py --loss_type bce --num_steps 1000 --gpu 0
#   python train.py --loss_type bce_comp --comp_mode staged --predictor_checkpoint checkpoints/mvp_bce_s42/predictor_bce.pt --num_steps 500 --gpu 0

import argparse
import importlib.util
import math
import os
import sys
import time

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import get_calibration_loader, get_random_loader
from losses import BCESparsityLoss, CompensationLoss, JSDDistillLoss, KLDistillLoss, ReverseKLDistillLoss, SparsityRegularizer
from predictor import CompensationNetwork, PredictorWrapper, STEMask


def parse_args():
    p = argparse.ArgumentParser(description="Train SwiGLU sparsity predictor")
    p.add_argument("--model_name_or_path", type=str,
                    default="/root/autodl-tmp/models/llama-3.1-8b")
    p.add_argument("--loss_type", type=str, choices=["kl", "kl_normalized", "kl_ste", "reverse_kl", "jsd", "bce", "bce_comp"], required=True)
    p.add_argument("--num_steps", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sparsity_target", type=float, default=0.5)
    p.add_argument("--sparsity_reg_weight", type=float, default=0.1)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output_dir", type=str,
                    default="/root/distill_sparse_swiglu/checkpoints")
    p.add_argument("--wandb_project", type=str, default="distill_sparse_swiglu")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--tau_start", type=float, default=1.0)
    p.add_argument("--tau_end", type=float, default=0.1)
    p.add_argument("--kl_temperature", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--use_random_data", action="store_true",
                    help="Use random tokens instead of real data (for dry-run)")
    p.add_argument("--dataset", type=str, default="wikitext-103",
                    choices=["wikitext-103", "c4"],
                    help="Calibration dataset (default: wikitext-103)")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--gradient_checkpointing", action="store_true",
                    help="Enable gradient checkpointing (auto-enabled for KL mode)")
    # Compensation-specific args
    p.add_argument("--predictor_checkpoint", type=str, default=None,
                    help="Pretrained predictor checkpoint (required for bce_comp staged mode)")
    p.add_argument("--comp_mode", type=str, choices=["staged", "joint"], default="staged",
                    help="staged: freeze predictor, train comp only. joint: train both.")
    p.add_argument("--comp_bottleneck", type=int, default=256,
                    help="Bottleneck size for compensation heads")
    p.add_argument("--bottleneck_dim", type=int, default=128,
                    help="Predictor bottleneck dimension (d_b)")
    p.add_argument("--comp_loss_weight", type=float, default=1.0,
                    help="Weight for compensation MSE loss (joint mode)")
    p.add_argument("--relaxation", type=str, choices=["gumbel", "ste"], default="gumbel",
                    help="Relaxation method: gumbel (Gumbel-sigmoid + tau annealing) or ste (Straight-Through Estimator)")
    p.add_argument("--lambda_max", type=float, default=1000.0,
                    help="Max value for Lagrangian multiplier lambda")
    p.add_argument("--save_every", type=int, default=0,
                    help="Save predictor checkpoint every N steps (0=disabled)")
    return p.parse_args()



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


def train_kl(wrapper, dataloader, args, device):
    kl_fn = KLDistillLoss(temperature=args.kl_temperature)

    params = list(wrapper.predictors.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    scheduler = cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.num_steps)

    wrapper.predictors.train()

    if args.gradient_checkpointing:
        wrapper.model.train()
        wrapper.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("Gradient checkpointing enabled")

    data_iter = iter(dataloader)
    ga = args.gradient_accumulation_steps

    lambda_sparse = 1.0
    target_sparsity = args.sparsity_target
    print(f"Lagrangian mode: lambda_sparse={lambda_sparse:.2f}, target_sparsity={target_sparsity}, update_every=25, tol=0.02, cap={args.lambda_max}")

    for opt_step in range(1, args.num_steps + 1):
        if args.relaxation != "ste":
            tau = args.tau_start + (args.tau_end - args.tau_start) * (opt_step / args.num_steps)
            wrapper.gumbel_mask.tau = tau
        else:
            tau = 0.0

        accum_loss = 0.0
        accum_kl = 0.0
        accum_constraint = 0.0

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

            if opt_step == 1:
                with torch.no_grad():
                    sl, vs = dense_logits.shape[-2], dense_logits.shape[-1]
                    print(f"  [Diag] KL normalized: {kl.item():.6f}, raw batchmean would be: {kl.item() * sl * vs:.4f}, seq_len={sl}, vocab_size={vs}")

            masks = wrapper.get_masks()
            actual_sparsity = 1.0 - torch.stack([m.mean() for m in masks.values()]).mean()
            sparsity_violation = target_sparsity - actual_sparsity
            constraint_loss = lambda_sparse * torch.clamp(sparsity_violation, min=0.0)

            loss = (kl + constraint_loss) / ga
            loss.backward()

            accum_loss += loss.item()
            accum_kl += kl.item() / ga
            accum_constraint += constraint_loss.item() / ga

        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        masks = wrapper.get_masks()
        avg_sparsity = (
            1.0 - torch.stack([m.mean() for m in masks.values()]).mean().item()
            if masks else 0.0
        )

        if opt_step % 25 == 0 and opt_step > 0:
            old_lambda = lambda_sparse
            if avg_sparsity < target_sparsity - 0.02:
                lambda_sparse *= 2.0
            elif avg_sparsity > target_sparsity + 0.02:
                lambda_sparse *= 0.5
            lambda_sparse = max(0.1, min(lambda_sparse, args.lambda_max))
            if old_lambda != lambda_sparse:
                print(f"  [Lambda update] step={opt_step} sparsity={avg_sparsity:.3f} lambda: {old_lambda:.2f} -> {lambda_sparse:.2f}")

        log_dict = {
            "loss": accum_loss, "kl_loss": accum_kl, "constraint_loss": accum_constraint,
            "sparsity": avg_sparsity, "tau": tau, "lr": scheduler.get_last_lr()[0],
            "lambda_sparse": lambda_sparse,
        }
        wandb.log(log_dict, step=opt_step)

        if opt_step % 10 == 0 or opt_step == 1:
            print(
                f"[Step {opt_step}/{args.num_steps}] loss={accum_loss:.4f} "
                f"kl={accum_kl:.4f} constraint={accum_constraint:.4f} "
                f"sparsity={avg_sparsity:.3f} lambda={lambda_sparse:.2f} "
                f"tau={tau:.3f} lr={scheduler.get_last_lr()[0]:.2e}"
            )



def train_kl_normalized(wrapper, dataloader, args, device, loss_fn=None):
    """KL with per-token-per-dim normalization + aggressive Lagrangian.

    KL already per-token (batchmean / seq_len from KLDistillLoss). No further division.
    Soft constraint: clamp(|sparsity - target| - margin, 0)^2 for smooth gradients.
    Lambda: every-step update, x2.0 growth, x0.5 decay, cap={args.lambda_max}.
    """
    kl_fn = loss_fn if loss_fn is not None else KLDistillLoss(temperature=args.kl_temperature)

    params = list(wrapper.predictors.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    scheduler = cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.num_steps)

    wrapper.predictors.train()

    if args.gradient_checkpointing:
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
    print(f"KL-normalized Lagrangian: lambda={lambda_sparse}, target={target_sparsity}, "
          f"margin={margin}, update=every_step, growth=2.0x, decay=0.5x, cap={args.lambda_max}")

    for opt_step in range(1, args.num_steps + 1):
        if args.relaxation != "ste":
            tau = args.tau_start + (args.tau_end - args.tau_start) * (opt_step / args.num_steps)
            wrapper.gumbel_mask.tau = tau
        else:
            tau = 0.0

        accum_loss = 0.0
        accum_kl_raw = 0.0
        accum_kl_norm = 0.0
        accum_constraint = 0.0

        for _ in range(ga):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            dense_logits = wrapper.forward_dense(input_ids).logits.detach()
            sparse_logits = wrapper.forward_sparse(input_ids).logits

            # kl_fn returns per-token KL (batchmean / seq_len), summed over vocab
            kl_raw = kl_fn(dense_logits, sparse_logits)
            kl_norm = kl_raw

            masks = wrapper.get_masks()
            actual_sparsity = 1.0 - torch.stack([m.mean() for m in masks.values()]).mean()

            # Soft constraint: clamp(|sparsity - target| - margin, 0)^2
            sparsity_error = torch.abs(actual_sparsity - target_sparsity)
            constraint_loss = lambda_sparse * torch.clamp(sparsity_error - margin, min=0.0) ** 2

            loss = (kl_norm + constraint_loss) / ga
            loss.backward()

            accum_loss += loss.item()
            accum_kl_raw += kl_raw.item() / ga
            accum_kl_norm += kl_norm.item() / ga
            accum_constraint += constraint_loss.item() / ga

        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        masks = wrapper.get_masks()
        avg_sparsity = (
            1.0 - torch.stack([m.mean() for m in masks.values()]).mean().item()
            if masks else 0.0
        )

        # Aggressive Lagrangian: every-step update, x2.0/x0.5, margin=0.02, cap=lambda_max
        old_lambda = lambda_sparse
        if avg_sparsity < target_sparsity - margin:
            lambda_sparse *= 2.0
        elif avg_sparsity > target_sparsity + margin:
            lambda_sparse *= 0.5
        lambda_sparse = max(0.01, min(lambda_sparse, args.lambda_max))

        log_dict = {
            "loss": accum_loss, "kl_raw": accum_kl_raw, "kl_normalized": accum_kl_norm,
            "constraint_loss": accum_constraint, "sparsity": avg_sparsity,
            "tau": tau, "lr": scheduler.get_last_lr()[0], "lambda_sparse": lambda_sparse,
        }
        wandb.log(log_dict, step=opt_step)

        if opt_step % 10 == 0 or opt_step == 1:
            print(
                f"[Step {opt_step}/{args.num_steps}] loss={accum_loss:.6f} "
                f"kl_raw={accum_kl_raw:.4f} kl_norm={accum_kl_norm:.6f} "
                f"constraint={accum_constraint:.6f} sparsity={avg_sparsity:.3f} "
                f"lambda={lambda_sparse:.2f} tau={tau:.3f} lr={scheduler.get_last_lr()[0]:.2e}"
            )
            if old_lambda != lambda_sparse:
                print(f"  [Lambda] {old_lambda:.2f} -> {lambda_sparse:.2f}")

        if args.save_every > 0 and (opt_step % args.save_every == 0 or opt_step == args.num_steps):
            ckpt_path = os.path.join(args.output_dir, f"predictor_{args.loss_type}_step_{opt_step}.pt")
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save({"predictors": wrapper.predictors.state_dict(), "args": vars(args)}, ckpt_path)
            print(f"Saved checkpoint -> {ckpt_path}")


def train_kl_ste(wrapper, dataloader, args, device):
    """KL distillation with Straight-Through Estimator relaxation.

    Same loss/constraint as train_kl_normalized, but uses hard threshold + STE
    gradient instead of Gumbel-sigmoid. No temperature annealing.
    """
    from predictor import STEMask
    wrapper.gumbel_mask = STEMask()

    kl_fn = KLDistillLoss(temperature=args.kl_temperature)

    params = list(wrapper.predictors.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    scheduler = cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.num_steps)

    wrapper.predictors.train()

    if args.gradient_checkpointing:
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
    print(f"KL-STE Lagrangian: lambda={lambda_sparse}, target={target_sparsity}, "
          f"margin={margin}, relaxation=STE")

    for opt_step in range(1, args.num_steps + 1):
        accum_loss = 0.0
        accum_kl_raw = 0.0
        accum_kl_norm = 0.0
        accum_constraint = 0.0

        for _ in range(ga):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            dense_logits = wrapper.forward_dense(input_ids).logits.detach()
            sparse_logits = wrapper.forward_sparse(input_ids).logits

            kl_raw = kl_fn(dense_logits, sparse_logits)
            kl_norm = kl_raw

            masks = wrapper.get_masks()
            actual_sparsity = 1.0 - torch.stack([m.mean() for m in masks.values()]).mean()

            sparsity_error = torch.abs(actual_sparsity - target_sparsity)
            constraint_loss = lambda_sparse * torch.clamp(sparsity_error - margin, min=0.0) ** 2

            loss = (kl_norm + constraint_loss) / ga
            loss.backward()

            accum_loss += loss.item()
            accum_kl_raw += kl_raw.item() / ga
            accum_kl_norm += kl_norm.item() / ga
            accum_constraint += constraint_loss.item() / ga

        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        masks = wrapper.get_masks()
        avg_sparsity = (
            1.0 - torch.stack([m.mean() for m in masks.values()]).mean().item()
            if masks else 0.0
        )

        old_lambda = lambda_sparse
        if avg_sparsity < target_sparsity - margin:
            lambda_sparse *= 2.0
        elif avg_sparsity > target_sparsity + margin:
            lambda_sparse *= 0.5
        lambda_sparse = max(0.01, min(lambda_sparse, args.lambda_max))

        log_dict = {
            "loss": accum_loss, "kl_raw": accum_kl_raw, "kl_normalized": accum_kl_norm,
            "constraint_loss": accum_constraint, "sparsity": avg_sparsity,
            "lr": scheduler.get_last_lr()[0], "lambda_sparse": lambda_sparse,
        }
        wandb.log(log_dict, step=opt_step)

        if opt_step % 10 == 0 or opt_step == 1:
            print(
                f"[Step {opt_step}/{args.num_steps}] loss={accum_loss:.6f} "
                f"kl_raw={accum_kl_raw:.4f} kl_norm={accum_kl_norm:.6f} "
                f"constraint={accum_constraint:.6f} sparsity={avg_sparsity:.3f} "
                f"lambda={lambda_sparse:.2f} lr={scheduler.get_last_lr()[0]:.2e}"
            )
            if old_lambda != lambda_sparse:
                print(f"  [Lambda] {old_lambda:.2f} -> {lambda_sparse:.2f}")


def train_bce(wrapper, dataloader, args, device):
    bce_fn = BCESparsityLoss(sparsity_target=args.sparsity_target)
    reg_fn = SparsityRegularizer(args.sparsity_target, args.sparsity_reg_weight)

    params = list(wrapper.predictors.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    scheduler = cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.num_steps)

    wrapper.predictors.train()
    data_iter = iter(dataloader)
    ga = args.gradient_accumulation_steps

    t0 = time.time()

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
                soft_mask = torch.sigmoid(pred_logits)
                hard_mask = (pred_logits > 0).float()
                all_masks[layer_idx] = hard_mask - soft_mask.detach() + soft_mask

            total_bce = total_bce / max(num_layers, 1)
            reg = reg_fn(all_masks)
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

        if opt_step % args.log_every == 0:
            print(
                f"[Step {opt_step}/{args.num_steps}] loss={accum_loss:.4f} "
                f"bce={accum_bce:.4f} reg={accum_reg:.4f} "
                f"sparsity={last_sparsity:.3f} lr={scheduler.get_last_lr()[0]:.2e} "
                f"elapsed={elapsed:.1f}s ({elapsed/opt_step:.2f}s/step)"
            )

        if args.save_every > 0 and (opt_step % args.save_every == 0 or opt_step == args.num_steps):
            ckpt_path = os.path.join(args.output_dir, f"predictor_{args.loss_type}_step_{opt_step}.pt")
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save({"predictors": wrapper.predictors.state_dict(), "args": vars(args)}, ckpt_path)
            print(f"Saved checkpoint -> {ckpt_path}")

    total_time = time.time() - t0
    print(f"\nBCE training complete: {args.num_steps} steps in {total_time:.1f}s ({total_time/args.num_steps:.2f}s/step)")


def train_bce_comp(wrapper, dataloader, args, device):
    """Train compensation network to correct sparse MLP output errors.

    Staged: predictor frozen, only train compensation heads.
    Joint: train predictor (BCE) + compensation (MSE) together.
    """
    config = wrapper.model.config
    comp_net = CompensationNetwork(
        num_layers=config.num_hidden_layers,
        hidden_size=config.hidden_size,
        bottleneck_size=args.comp_bottleneck,
    ).to(device=device, dtype=torch.bfloat16)
    wrapper.comp_network = comp_net

    comp_loss_fn = CompensationLoss()

    if args.comp_mode == "staged":
        for p in wrapper.predictors.parameters():
            p.requires_grad = False
        wrapper.predictors.eval()
        params = list(comp_net.parameters())
        print(f"Staged mode: predictor frozen, training {sum(p.numel() for p in params):,} compensation params")
    else:
        bce_fn = BCESparsityLoss(sparsity_target=args.sparsity_target)
        reg_fn = SparsityRegularizer(args.sparsity_target, args.sparsity_reg_weight)
        wrapper.predictors.train()
        comp_net.train()
        params = list(wrapper.predictors.parameters()) + list(comp_net.parameters())
        print(f"Joint mode: training {sum(p.numel() for p in params):,} params (predictor + compensation)")

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    scheduler = cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.num_steps)

    data_iter = iter(dataloader)
    ga = args.gradient_accumulation_steps
    num_layers = len(wrapper.predictors)

    for opt_step in range(1, args.num_steps + 1):
        accum_loss = 0.0
        accum_comp = 0.0
        accum_bce = 0.0
        last_sparsity = 0.0

        for _ in range(ga):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)

            wrapper.forward_dense(input_ids, capture_intermediates=True)
            intermediates = wrapper.get_intermediates()
            layer_inputs = wrapper.get_layer_inputs()

            total_comp_loss = torch.tensor(0.0, device=device)
            total_bce_loss = torch.tensor(0.0, device=device)
            all_masks = {}
            active_layers = 0

            for layer_idx in range(num_layers):
                if layer_idx not in intermediates:
                    continue
                active_layers += 1

                dense_act = intermediates[layer_idx]
                inp = layer_inputs[layer_idx]

                if args.comp_mode == "staged":
                    with torch.no_grad():
                        pred_logits = wrapper.predictors[layer_idx](inp)
                        mask = (pred_logits > 0).to(dense_act.dtype)
                else:
                    pred_logits = wrapper.predictors[layer_idx](inp)
                    total_bce_loss = total_bce_loss + bce_fn(pred_logits, dense_act)
                    mask = (pred_logits > 0).to(dense_act.dtype)

                all_masks[layer_idx] = mask.detach()

                down_proj = wrapper.model.model.layers[layer_idx].mlp.down_proj
                with torch.no_grad():
                    dense_out = down_proj(dense_act)
                    sparse_out = down_proj(dense_act * mask)

                comp = comp_net.forward_layer(layer_idx, inp)
                total_comp_loss = total_comp_loss + comp_loss_fn(comp, sparse_out, dense_out)

            total_comp_loss = total_comp_loss / max(active_layers, 1)

            if args.comp_mode == "joint":
                total_bce_loss = total_bce_loss / max(active_layers, 1)
                reg = reg_fn(all_masks)
                loss = (total_bce_loss + args.comp_loss_weight * total_comp_loss + reg) / ga
                accum_bce += total_bce_loss.item() / ga
            else:
                loss = total_comp_loss / ga

            loss.backward()
            accum_loss += loss.item()
            accum_comp += total_comp_loss.item() / ga

            last_sparsity = (
                1.0 - torch.stack([m.mean() for m in all_masks.values()]).mean().item()
                if all_masks else 0.0
            )

        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        log_dict = {
            "loss": accum_loss, "comp_loss": accum_comp,
            "sparsity": last_sparsity, "lr": scheduler.get_last_lr()[0],
        }
        if args.comp_mode == "joint":
            log_dict["bce_loss"] = accum_bce
        wandb.log(log_dict, step=opt_step)

        if opt_step % args.log_every == 0:
            msg = (f"[Step {opt_step}/{args.num_steps}] loss={accum_loss:.4f} "
                   f"comp={accum_comp:.4f} sparsity={last_sparsity:.3f} "
                   f"lr={scheduler.get_last_lr()[0]:.2e}")
            if args.comp_mode == "joint":
                msg += f" bce={accum_bce:.4f}"
            print(msg)

    return comp_net


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    os.makedirs(args.output_dir, exist_ok=True)

    run_name = args.wandb_run_name or f"{args.loss_type}_s{args.sparsity_target}_lr{args.lr}"
    wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    print(f"Loading model from {args.model_name_or_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation=_best_attn_impl(),
    )
    model.eval()

    print("Initializing predictor wrapper ...")
    wrapper = PredictorWrapper(model, bottleneck_size=args.bottleneck_dim)
    wrapper.predictors.to(device=device, dtype=torch.bfloat16)

    if args.relaxation == "ste":
        wrapper.gumbel_mask = STEMask()
        print("Using STE (Straight-Through Estimator) relaxation")
    else:
        print("Using Gumbel-sigmoid relaxation")

    print("Setting up data loader ...")
    if args.use_random_data:
        vocab_size = tokenizer.vocab_size
        dataloader = get_random_loader(vocab_size, args.batch_size, args.seq_len, args.seed)
        print(f"Using random data (vocab_size={vocab_size})")
    else:
        dataloader = get_calibration_loader(tokenizer, args.batch_size, args.seq_len, args.seed, args.dataset)

    trainable = sum(p.numel() for p in wrapper.predictors.parameters())
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

    if args.loss_type == "bce_comp":
        if args.predictor_checkpoint:
            print(f"Loading predictor from {args.predictor_checkpoint} ...")
            ckpt = torch.load(args.predictor_checkpoint, map_location=device, weights_only=True)
            wrapper.predictors.load_state_dict(ckpt["predictors"])
            print("Predictor loaded")
        elif args.comp_mode == "staged":
            raise ValueError("Staged mode requires --predictor_checkpoint")

        comp_net = train_bce_comp(wrapper, dataloader, args, device)

        ckpt_path = os.path.join(args.output_dir, "predictor_bce_comp.pt")
        torch.save({
            "predictors": wrapper.predictors.state_dict(),
            "comp_network": comp_net.state_dict(),
            "args": vars(args),
        }, ckpt_path)
        print(f"Saved checkpoint -> {ckpt_path}")
        comp_params = sum(p.numel() for p in comp_net.parameters())
        print(f"Compensation params: {comp_params:,} ({comp_params * 2 / 1024**2:.1f} MB in bf16)")

    elif args.loss_type == "kl":
        if not args.gradient_checkpointing:
            args.gradient_checkpointing = True
            print("Auto-enabling gradient checkpointing for KL mode")
        train_kl(wrapper, dataloader, args, device)
        ckpt_path = os.path.join(args.output_dir, f"predictor_{args.loss_type}.pt")
        torch.save({"predictors": wrapper.predictors.state_dict(), "args": vars(args)}, ckpt_path)
        print(f"Saved checkpoint -> {ckpt_path}")

    elif args.loss_type == "kl_ste":
        if not args.gradient_checkpointing:
            args.gradient_checkpointing = True
            print("Auto-enabling gradient checkpointing for KL-STE mode")
        train_kl_ste(wrapper, dataloader, args, device)
        ckpt_path = os.path.join(args.output_dir, f"predictor_{args.loss_type}.pt")
        torch.save({"predictors": wrapper.predictors.state_dict(), "args": vars(args)}, ckpt_path)
        print(f"Saved checkpoint -> {ckpt_path}")

    elif args.loss_type == "kl_normalized":
        if not args.gradient_checkpointing:
            args.gradient_checkpointing = True
            print("Auto-enabling gradient checkpointing for KL-normalized mode")
        train_kl_normalized(wrapper, dataloader, args, device)
        ckpt_path = os.path.join(args.output_dir, f"predictor_{args.loss_type}.pt")
        torch.save({"predictors": wrapper.predictors.state_dict(), "args": vars(args)}, ckpt_path)
        print(f"Saved checkpoint -> {ckpt_path}")

    elif args.loss_type == "reverse_kl":
        if not args.gradient_checkpointing:
            args.gradient_checkpointing = True
            print("Auto-enabling gradient checkpointing for reverse-KL mode")
        loss_fn = ReverseKLDistillLoss(temperature=args.kl_temperature)
        train_kl_normalized(wrapper, dataloader, args, device, loss_fn=loss_fn)
        ckpt_path = os.path.join(args.output_dir, f"predictor_{args.loss_type}.pt")
        torch.save({"predictors": wrapper.predictors.state_dict(), "args": vars(args)}, ckpt_path)
        print(f"Saved checkpoint -> {ckpt_path}")

    elif args.loss_type == "jsd":
        if not args.gradient_checkpointing:
            args.gradient_checkpointing = True
            print("Auto-enabling gradient checkpointing for JSD mode")
        loss_fn = JSDDistillLoss(temperature=args.kl_temperature)
        train_kl_normalized(wrapper, dataloader, args, device, loss_fn=loss_fn)
        ckpt_path = os.path.join(args.output_dir, f"predictor_{args.loss_type}.pt")
        torch.save({"predictors": wrapper.predictors.state_dict(), "args": vars(args)}, ckpt_path)
        print(f"Saved checkpoint -> {ckpt_path}")

    else:
        train_bce(wrapper, dataloader, args, device)
        ckpt_path = os.path.join(args.output_dir, f"predictor_{args.loss_type}.pt")
        torch.save({"predictors": wrapper.predictors.state_dict(), "args": vars(args)}, ckpt_path)
        print(f"Saved checkpoint -> {ckpt_path}")

    wandb.finish()


if __name__ == "__main__":
    main()

# Per-layer MSE training for SwiGLU sparsity predictors.
# "Intermediate" baseline between per-layer BCE (activation space) and end-to-end KL (logit space).
# Loss: sum_l ||down_proj_l(act_l * mask_l) - down_proj_l(act_l)||^2 / num_layers
# Gradient flows through Gumbel-sigmoid mask -> predictor only; model stays frozen.
#
# Input:  frozen Llama model + calibration data (wikitext-103 or c4)
# Output: predictor_mse.pt (same format as KL/BCE checkpoints for benchmark_eval.py)
#
# Usage:
#   python train_perlayer_mse.py --num_steps 1000 --sparsity_target 0.5 --gpu 0

import argparse
import importlib.util
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import get_calibration_loader, get_random_loader
from losses import PerLayerMSELoss
from predictor import PredictorWrapper


def parse_args():
    p = argparse.ArgumentParser(
        description="Per-layer MSE training for SwiGLU sparsity predictors"
    )
    p.add_argument("--model_name_or_path", type=str,
                    default="/root/autodl-tmp/models/llama-3.1-8b")
    p.add_argument("--num_steps", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sparsity_target", type=float, default=0.5)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output_dir", type=str,
                    default="/root/distill_sparse_swiglu/checkpoints")
    p.add_argument("--wandb_project", type=str, default="distill_sparse_swiglu")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--tau_start", type=float, default=1.0)
    p.add_argument("--tau_end", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--use_random_data", action="store_true",
                    help="Use random tokens instead of real data (for dry-run)")
    p.add_argument("--dataset", type=str, default="wikitext-103",
                    choices=["wikitext-103", "c4"])
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--bottleneck_dim", type=int, default=128,
                    help="Predictor bottleneck dimension (d_b)")
    p.add_argument("--lambda_max", type=float, default=1000.0,
                    help="Max value for adaptive Lagrangian multiplier")
    p.add_argument("--save_every", type=int, default=0,
                    help="Save checkpoint every N steps (0=disabled)")
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


def train(wrapper, dataloader, args, device):
    """Per-layer MSE training with adaptive Lagrangian sparsity constraint.

    Each step: dense forward (capture intermediates) -> per-layer MSE loop -> backward.
    Lagrangian: lambda * clamp(|sparsity - target| - margin, 0)^2, lambda adapts x2/x0.5.
    """
    mse_fn = PerLayerMSELoss()
    num_layers = wrapper.model.config.num_hidden_layers

    params = list(wrapper.predictors.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    scheduler = cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.num_steps)

    wrapper.predictors.train()
    data_iter = iter(dataloader)
    ga = args.gradient_accumulation_steps

    lambda_sparse = 1.0
    target_sparsity = args.sparsity_target
    margin = 0.02
    print(f"Per-layer MSE Lagrangian: lambda={lambda_sparse}, target={target_sparsity}, "
          f"margin={margin}, growth=2.0x, decay=0.5x, cap={args.lambda_max}")

    for opt_step in range(1, args.num_steps + 1):
        t0 = time.time()
        tau = args.tau_start + (args.tau_end - args.tau_start) * (opt_step / args.num_steps)
        wrapper.gumbel_mask.tau = tau

        accum_loss = 0.0
        accum_mse = 0.0
        accum_constraint = 0.0
        last_sparsity = 0.0

        for _ in range(ga):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)

            # Dense forward: captures per-layer intermediates and inputs (all detached)
            wrapper.forward_dense(input_ids, capture_intermediates=True)
            intermediates = wrapper.get_intermediates()
            layer_inputs = wrapper.get_layer_inputs()

            total_mse = torch.tensor(0.0, device=device)
            all_masks = {}

            for layer_idx in range(num_layers):
                dense_act = intermediates[layer_idx]    # [B, S, intermediate_size], detached
                inp = layer_inputs[layer_idx]            # [B, S, hidden_size], detached

                logits = wrapper.predictors[layer_idx](inp)
                mask = wrapper.gumbel_mask(logits)
                all_masks[layer_idx] = mask.detach()

                down_proj = wrapper.model.model.layers[layer_idx].mlp.down_proj
                with torch.no_grad():
                    dense_out = down_proj(dense_act)
                # grad flows: mse -> sparse_out -> down_proj (frozen linear) -> mask -> predictor
                sparse_out = down_proj(dense_act * mask)

                total_mse = total_mse + mse_fn(sparse_out, dense_out)
                del dense_out, sparse_out

            total_mse = total_mse / num_layers

            actual_sparsity = 1.0 - torch.stack([m.mean() for m in all_masks.values()]).mean()
            sparsity_error = torch.abs(actual_sparsity - target_sparsity)
            constraint_loss = lambda_sparse * torch.clamp(sparsity_error - margin, min=0.0) ** 2

            loss = (total_mse + constraint_loss) / ga
            loss.backward()

            accum_loss += loss.item()
            accum_mse += total_mse.item() / ga
            accum_constraint += constraint_loss.item() / ga
            last_sparsity = actual_sparsity.item()

        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Adaptive Lagrangian: push lambda up when too dense, down when too sparse
        old_lambda = lambda_sparse
        if last_sparsity < target_sparsity - margin:
            lambda_sparse *= 2.0
        elif last_sparsity > target_sparsity + margin:
            lambda_sparse *= 0.5
        lambda_sparse = max(0.01, min(lambda_sparse, args.lambda_max))

        dt = time.time() - t0
        log_dict = {
            "loss": accum_loss, "mse_loss": accum_mse,
            "constraint_loss": accum_constraint, "sparsity": last_sparsity,
            "tau": tau, "lr": scheduler.get_last_lr()[0], "lambda_sparse": lambda_sparse,
        }
        wandb.log(log_dict, step=opt_step)

        if opt_step % args.log_every == 0 or opt_step == 1:
            print(
                f"[Step {opt_step}/{args.num_steps}] loss={accum_loss:.6f} "
                f"mse={accum_mse:.6f} constraint={accum_constraint:.6f} "
                f"sparsity={last_sparsity:.3f} lambda={lambda_sparse:.2f} "
                f"tau={tau:.3f} lr={scheduler.get_last_lr()[0]:.2e} dt={dt:.1f}s"
            )
            if old_lambda != lambda_sparse:
                print(f"  [Lambda] {old_lambda:.2f} -> {lambda_sparse:.2f}")

        if args.save_every > 0 and (opt_step % args.save_every == 0 or opt_step == args.num_steps):
            ckpt_path = os.path.join(args.output_dir, f"predictor_mse_step_{opt_step}.pt")
            torch.save({"predictors": wrapper.predictors.state_dict(), "args": vars(args)}, ckpt_path)
            print(f"Saved checkpoint -> {ckpt_path}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    os.makedirs(args.output_dir, exist_ok=True)

    run_name = args.wandb_run_name or f"mse_s{args.sparsity_target}_lr{args.lr}"
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

    print("Setting up data loader ...")
    if args.use_random_data:
        vocab_size = tokenizer.vocab_size
        dataloader = get_random_loader(vocab_size, args.batch_size, args.seq_len, args.seed)
        print(f"Using random data (vocab_size={vocab_size})")
    else:
        dataloader = get_calibration_loader(
            tokenizer, args.batch_size, args.seq_len, args.seed, args.dataset
        )

    pred_params = sum(p.numel() for p in wrapper.predictors.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Predictor params: {pred_params:,} ({pred_params * 2 / 1024**2:.1f} MB bf16)")
    print(f"Model params: {total_params:,}")

    train(wrapper, dataloader, args, device)

    ckpt_path = os.path.join(args.output_dir, "predictor_mse.pt")
    torch.save({"predictors": wrapper.predictors.state_dict(), "args": vars(args)}, ckpt_path)
    print(f"Saved checkpoint -> {ckpt_path}")

    wandb.finish()


if __name__ == "__main__":
    main()

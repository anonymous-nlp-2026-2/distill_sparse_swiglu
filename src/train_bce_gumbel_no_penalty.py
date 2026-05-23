"""BCE + Gumbel-sigmoid WITHOUT Lagrangian penalty controller.

Control experiment: keeps Gumbel relaxation + tau annealing but removes
the adaptive lambda multiplier that enforces sparsity. Isolates the
effect of Gumbel relaxation from Lagrangian sparsity control.
"""
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
from data_utils import get_calibration_loader
from losses import BCESparsityLoss
from predictor import PredictorWrapper


def parse_args():
    p = argparse.ArgumentParser(description="BCE+Gumbel NO penalty control experiment")
    p.add_argument("--model_name_or_path", type=str,
                    default="/root/autodl-tmp/models/llama-3.1-8b")
    p.add_argument("--num_steps", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sparsity_target", type=float, default=0.5)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--wandb_project", type=str, default="distill_sparse_swiglu")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--tau_start", type=float, default=1.0)
    p.add_argument("--tau_end", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--dataset", type=str, default="wikitext-103",
                    choices=["wikitext-103", "c4"])
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--gradient_checkpointing", action="store_true")
    return p.parse_args()


def cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _best_attn_impl():
    if importlib.util.find_spec("flash_attn"):
        return "flash_attention_2"
    return "sdpa"


def train_bce_gumbel_no_penalty(wrapper, dataloader, args, device):
    """BCE loss + Gumbel-sigmoid relaxation, NO Lagrangian sparsity constraint.

    Compared to train_bce_gumbel: same tau annealing, same optimizer/scheduler,
    same BCE loss. Only difference: no lambda_sparse multiplier, no constraint_loss.
    Loss = BCE only. Sparsity is whatever Gumbel-sigmoid naturally produces.
    """
    bce_fn = BCESparsityLoss(sparsity_target=args.sparsity_target)

    params = list(wrapper.predictors.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    scheduler = cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.num_steps)

    wrapper.predictors.train()
    if args.gradient_checkpointing:
        wrapper.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("Gradient checkpointing enabled")

    data_iter = iter(dataloader)
    ga = args.gradient_accumulation_steps

    print(f"BCE+Gumbel NO PENALTY: target={args.sparsity_target} (BCE oracle only, no constraint)")
    print(f"Tau annealing: {args.tau_start} -> {args.tau_end} over {args.num_steps} steps")

    t0 = time.time()

    for opt_step in range(1, args.num_steps + 1):
        tau = args.tau_start + (args.tau_end - args.tau_start) * (opt_step / args.num_steps)
        wrapper.gumbel_mask.tau = tau

        accum_loss = 0.0
        accum_bce = 0.0

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

                mask = wrapper.gumbel_mask(pred_logits)
                all_masks[layer_idx] = mask

            total_bce = total_bce / max(num_layers, 1)

            loss = total_bce / ga
            loss.backward()

            accum_loss += loss.item()
            accum_bce += total_bce.item() / ga

        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        with torch.no_grad():
            if all_masks:
                avg_sparsity = 1.0 - torch.stack([m.mean() for m in all_masks.values()]).mean().item()
            else:
                avg_sparsity = 0.0

        log_dict = {
            "loss": accum_loss, "bce_loss": accum_bce,
            "sparsity": avg_sparsity, "tau": tau, "lr": scheduler.get_last_lr()[0],
        }
        wandb.log(log_dict, step=opt_step)

        if opt_step % args.log_every == 0 or opt_step == 1:
            elapsed = time.time() - t0
            print(
                f"[Step {opt_step}/{args.num_steps}] loss={accum_loss:.4f} "
                f"bce={accum_bce:.4f} "
                f"sparsity={avg_sparsity:.3f} "
                f"tau={tau:.3f} lr={scheduler.get_last_lr()[0]:.2e} "
                f"elapsed={elapsed:.1f}s"
            )

    total_time = time.time() - t0
    print(f"\nBCE+Gumbel (no penalty) training complete: {args.num_steps} steps in {total_time:.1f}s "
          f"({total_time/args.num_steps:.2f}s/step)")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    os.makedirs(args.output_dir, exist_ok=True)

    run_name = args.wandb_run_name or f"bce_gumbel_no_penalty_lr{args.lr}_s{args.seed}"
    wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    print(f"Loading model from {args.model_name_or_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation=_best_attn_impl(),
    )
    model.eval()

    print("Initializing predictor wrapper ...")
    wrapper = PredictorWrapper(model, bottleneck_size=128)
    wrapper.predictors.to(device=device, dtype=torch.bfloat16)
    print("Using Gumbel-sigmoid relaxation (NO Lagrangian penalty)")

    print("Setting up data loader ...")
    dataloader = get_calibration_loader(tokenizer, args.batch_size, args.seq_len, args.seed, args.dataset)

    trainable = sum(p.numel() for p in wrapper.predictors.parameters())
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

    train_bce_gumbel_no_penalty(wrapper, dataloader, args, device)

    ckpt_path = os.path.join(args.output_dir, "predictor_bce_gumbel_no_penalty.pt")
    torch.save({"predictors": wrapper.predictors.state_dict(), "args": vars(args)}, ckpt_path)
    print(f"Saved checkpoint -> {ckpt_path}")

    wandb.finish()


if __name__ == "__main__":
    main()

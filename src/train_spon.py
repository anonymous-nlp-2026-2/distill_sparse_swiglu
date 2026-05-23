"""Train SPON bias vectors for sparse SwiGLU error compensation.

Usage:
    python train_spon.py --predictor_checkpoint checkpoints/predictor_kl.pt --num_steps 1000 --gpu 0
    python train_spon.py --predictor_checkpoint checkpoints/predictor_kl.pt --use_random_data --gpu 0
"""

import argparse
import importlib.util
import math
import os
import sys

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import get_calibration_loader, get_random_loader
from losses import KLDistillLoss
from predictor import SparsityPredictor
from spon import SPONBiasVectors, SPONPatcher


def parse_args():
    p = argparse.ArgumentParser(description="Train SPON bias vectors")
    p.add_argument("--model_name_or_path", type=str,
                    default="/root/autodl-tmp/models/llama-3.1-8b")
    p.add_argument("--predictor_checkpoint", type=str, required=True,
                    help="Path to trained predictor .pt checkpoint")
    p.add_argument("--num_steps", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sparsity_target", type=float, default=0.5)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output_dir", type=str,
                    default="/root/distill_sparse_swiglu/checkpoints")
    p.add_argument("--wandb_project", type=str, default="distill_sparse_swiglu")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--kl_temperature", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--use_random_data", action="store_true",
                    help="Use random tokens instead of real data (for dry-run)")
    p.add_argument("--dataset", type=str, default="wikitext-103",
                    choices=["wikitext-103", "c4"])
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--gradient_checkpointing", action="store_true")
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


def load_predictors(checkpoint_path, num_layers, hidden_size, intermediate_size,
                    bottleneck_size=128, device="cpu"):
    predictors = torch.nn.ModuleList([
        SparsityPredictor(hidden_size, intermediate_size, bottleneck_size)
        for _ in range(num_layers)
    ])
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    predictors.load_state_dict(ckpt["predictors"])
    predictors.eval()
    for p in predictors.parameters():
        p.requires_grad = False
    return predictors


def train_spon(model, patcher, dataloader, args, device):
    kl_fn = KLDistillLoss(temperature=args.kl_temperature)

    params = list(patcher.spon_biases.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    scheduler = cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.num_steps)

    if args.gradient_checkpointing:
        model.train()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("Gradient checkpointing enabled")

    data_iter = iter(dataloader)
    ga = args.gradient_accumulation_steps

    for opt_step in range(1, args.num_steps + 1):
        optimizer.zero_grad()
        accum_loss = 0.0

        for _ in range(ga):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)

            patcher.mode = "dense"
            with torch.no_grad():
                dense_logits = model(input_ids=input_ids).logits.detach()

            patcher.mode = "sparse_spon"
            sparse_logits = model(input_ids=input_ids).logits

            loss = kl_fn(dense_logits, sparse_logits) / ga
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()

        bias_norms = [b.data.norm().item() for b in patcher.spon_biases.biases]
        mean_norm = sum(bias_norms) / len(bias_norms)

        wandb.log({
            "spon/loss": accum_loss,
            "spon/lr": scheduler.get_last_lr()[0],
            "spon/mean_bias_norm": mean_norm,
        }, step=opt_step)

        if opt_step % args.log_every == 0:
            print(
                f"[Step {opt_step}/{args.num_steps}] loss={accum_loss:.4f} "
                f"mean_bias_norm={mean_norm:.6f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    os.makedirs(args.output_dir, exist_ok=True)

    run_name = args.wandb_run_name or f"spon_lr{args.lr}_s{args.sparsity_target}"
    wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    print(f"Loading model from {args.model_name_or_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation=_best_attn_impl(),
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    config = model.config
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size

    print(f"Loading predictor from {args.predictor_checkpoint} ...")
    predictors = load_predictors(
        args.predictor_checkpoint, num_layers, hidden_size,
        intermediate_size, device=device,
    )
    predictors.to(device=device, dtype=torch.bfloat16)

    print("Initializing SPON bias vectors ...")
    spon_biases = SPONBiasVectors(num_layers, intermediate_size)
    spon_biases.to(device=device, dtype=torch.bfloat16)

    trainable = spon_biases.param_count()
    total = sum(p.numel() for p in model.parameters())
    print(f"SPON params: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")

    patcher = SPONPatcher(model, predictors, spon_biases)
    patcher.patch()

    print("Setting up data loader ...")
    if args.use_random_data:
        dataloader = get_random_loader(
            tokenizer.vocab_size, args.batch_size, args.seq_len, args.seed)
        print(f"Using random data (vocab_size={tokenizer.vocab_size})")
    else:
        dataloader = get_calibration_loader(
            tokenizer, args.batch_size, args.seq_len, args.seed, args.dataset)

    train_spon(model, patcher, dataloader, args, device)

    patcher.unpatch()

    ckpt_path = os.path.join(args.output_dir, "spon_biases.pt")
    torch.save({
        "spon_biases": spon_biases.state_dict(),
        "args": vars(args),
    }, ckpt_path)
    print(f"Saved SPON checkpoint -> {ckpt_path}")

    wandb.finish()


if __name__ == "__main__":
    main()

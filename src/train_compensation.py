"""Train CompensationNetwork for sparse SwiGLU error correction (sequential).

Loads a frozen predictor checkpoint, trains per-layer MLP compensation heads
with end-to-end KL distillation: KL(dense_logits, sparse_with_comp_logits).

Usage:
    python train_compensation.py --predictor_checkpoint checkpoints/mvp_bce_s42/predictor_bce.pt --num_steps 1000 --gpu 0
    python train_compensation.py --predictor_checkpoint checkpoints/mvp_bce_s42/predictor_bce.pt --use_random_data --num_steps 2 --gpu 0
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
from predictor import PredictorWrapper, CompensationNetwork


def parse_args():
    p = argparse.ArgumentParser(description="Train compensation network")
    p.add_argument("--model_name_or_path", type=str,
                    default="/root/autodl-tmp/models/llama-3.1-8b")
    p.add_argument("--predictor_checkpoint", type=str, required=True,
                    help="Path to trained predictor .pt checkpoint")
    p.add_argument("--num_steps", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-4)
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
    p.add_argument("--comp_bottleneck_size", type=int, default=256)
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


def train_compensation(wrapper, dataloader, args, device):
    kl_fn = KLDistillLoss(temperature=args.kl_temperature)

    params = list(wrapper.comp_network.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    scheduler = cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.num_steps)

    wrapper.comp_network.train()
    wrapper.gumbel_mask.hard = True

    wrapper.model.train()
    wrapper.model.gradient_checkpointing_enable(
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

            dense_logits = wrapper.forward_dense(input_ids).logits.detach()
            sparse_comp_logits = wrapper.forward_sparse(input_ids).logits

            loss = kl_fn(dense_logits, sparse_comp_logits) / ga
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()

        comp_norms = [
            h.net[0].weight.data.norm().item()
            for h in wrapper.comp_network.heads
        ]
        mean_norm = sum(comp_norms) / len(comp_norms)

        wandb.log({
            "comp/loss": accum_loss,
            "comp/lr": scheduler.get_last_lr()[0],
            "comp/mean_head_norm": mean_norm,
        }, step=opt_step)

        if opt_step % args.log_every == 0:
            print(
                f"[Step {opt_step}/{args.num_steps}] loss={accum_loss:.4f} "
                f"mean_head_norm={mean_norm:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    os.makedirs(args.output_dir, exist_ok=True)

    run_name = args.wandb_run_name or f"comp_lr{args.lr}"
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

    print("Initializing predictor wrapper ...")
    wrapper = PredictorWrapper(model, bottleneck_size=128)

    print(f"Loading predictor from {args.predictor_checkpoint} ...")
    ckpt = torch.load(args.predictor_checkpoint, map_location=device, weights_only=True)
    wrapper.predictors.load_state_dict(ckpt["predictors"])
    wrapper.predictors.to(device=device, dtype=torch.bfloat16)
    wrapper.predictors.eval()
    for p in wrapper.predictors.parameters():
        p.requires_grad = False

    config = model.config
    comp_network = CompensationNetwork(
        config.num_hidden_layers, config.hidden_size,
        bottleneck_size=args.comp_bottleneck_size,
    )
    comp_network.to(device=device, dtype=torch.bfloat16)
    wrapper.comp_network = comp_network
    wrapper.use_compensation = True

    trainable = sum(p.numel() for p in comp_network.parameters())
    total = sum(p.numel() for p in model.parameters())
    print(f"Compensation params: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

    print("Setting up data loader ...")
    if args.use_random_data:
        dataloader = get_random_loader(
            tokenizer.vocab_size, args.batch_size, args.seq_len, args.seed)
        print(f"Using random data (vocab_size={tokenizer.vocab_size})")
    else:
        dataloader = get_calibration_loader(
            tokenizer, args.batch_size, args.seq_len, args.seed, args.dataset)

    train_compensation(wrapper, dataloader, args, device)

    ckpt_path = os.path.join(args.output_dir, "compensation.pt")
    torch.save({
        "comp_network": comp_network.state_dict(),
        "args": vars(args),
    }, ckpt_path)
    print(f"Saved compensation checkpoint -> {ckpt_path}")

    wandb.finish()


if __name__ == "__main__":
    main()

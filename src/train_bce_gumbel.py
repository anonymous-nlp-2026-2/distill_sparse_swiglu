"""BCE + Gumbel-sigmoid + Lagrangian iso-framework ablation.

Gives BCE the exact same training framework as KL (Gumbel relaxation + tau annealing
+ Lagrangian sparsity constraint) to isolate the effect of loss function choice.
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
    p = argparse.ArgumentParser(description="BCE+Gumbel+Lagrangian iso-framework ablation")
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
    p.add_argument("--lambda_max", type=float, default=1000.0)
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


def train_bce_gumbel(wrapper, dataloader, args, device):
    """BCE loss + Gumbel-sigmoid relaxation + Lagrangian sparsity constraint.

    Iso-framework with train_kl: same tau annealing, same Lagrangian updates,
    same optimizer/scheduler. Only the loss differs:
    - KL: KL(P_dense || P_sparse) on output logit distributions
    - BCE: BinaryCrossEntropy(predictor_logits, magnitude_oracle) per neuron
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

    lambda_sparse = 1.0
    target_sparsity = args.sparsity_target
    print(f"BCE+Gumbel+Lagrangian: lambda={lambda_sparse:.2f}, target={target_sparsity}, "
          f"update_every=25, tol=0.02, cap={args.lambda_max}")
    print(f"Tau annealing: {args.tau_start} -> {args.tau_end} over {args.num_steps} steps")

    t0 = time.time()

    for opt_step in range(1, args.num_steps + 1):
        tau = args.tau_start + (args.tau_end - args.tau_start) * (opt_step / args.num_steps)
        wrapper.gumbel_mask.tau = tau

        accum_loss = 0.0
        accum_bce = 0.0
        accum_constraint = 0.0

        for _ in range(ga):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)

            # Dense forward to capture intermediates (oracle targets)
            wrapper.forward_dense(input_ids, capture_intermediates=True)
            intermediates = wrapper.get_intermediates()
            layer_inputs = wrapper.get_layer_inputs()

            # Compute BCE loss + Gumbel-sigmoid masks for Lagrangian
            total_bce = torch.tensor(0.0, device=device)
            all_masks = {}
            num_layers = len(wrapper.predictors)

            for layer_idx in range(num_layers):
                if layer_idx not in intermediates:
                    continue
                dense_act = intermediates[layer_idx]
                inp = layer_inputs[layer_idx]

                # Predictor logits (requires grad)
                pred_logits = wrapper.predictors[layer_idx](inp)

                # BCE loss: predictor logits vs magnitude oracle
                total_bce = total_bce + bce_fn(pred_logits, dense_act)

                # Gumbel-sigmoid mask (same as KL forward_sparse)
                mask = wrapper.gumbel_mask(pred_logits)
                all_masks[layer_idx] = mask

            total_bce = total_bce / max(num_layers, 1)

            # Lagrangian sparsity constraint (same as KL)
            if all_masks:
                actual_sparsity = 1.0 - torch.stack([m.mean() for m in all_masks.values()]).mean()
            else:
                actual_sparsity = torch.tensor(0.0, device=device)
            sparsity_violation = target_sparsity - actual_sparsity
            constraint_loss = lambda_sparse * torch.clamp(sparsity_violation, min=0.0)

            loss = (total_bce + constraint_loss) / ga
            loss.backward()

            accum_loss += loss.item()
            accum_bce += total_bce.item() / ga
            accum_constraint += constraint_loss.item() / ga

        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Measure sparsity for Lagrangian update (detached)
        with torch.no_grad():
            if all_masks:
                avg_sparsity = 1.0 - torch.stack([m.mean() for m in all_masks.values()]).mean().item()
            else:
                avg_sparsity = 0.0

        # Lagrangian multiplier update (same schedule as KL)
        if opt_step % 25 == 0 and opt_step > 0:
            old_lambda = lambda_sparse
            if avg_sparsity < target_sparsity - 0.02:
                lambda_sparse *= 2.0
            elif avg_sparsity > target_sparsity + 0.02:
                lambda_sparse *= 0.5
            lambda_sparse = max(0.1, min(lambda_sparse, args.lambda_max))
            if old_lambda != lambda_sparse:
                print(f"  [Lambda update] step={opt_step} sparsity={avg_sparsity:.3f} "
                      f"lambda: {old_lambda:.2f} -> {lambda_sparse:.2f}")

        log_dict = {
            "loss": accum_loss, "bce_loss": accum_bce, "constraint_loss": accum_constraint,
            "sparsity": avg_sparsity, "tau": tau, "lr": scheduler.get_last_lr()[0],
            "lambda_sparse": lambda_sparse,
        }
        wandb.log(log_dict, step=opt_step)

        if opt_step % args.log_every == 0 or opt_step == 1:
            elapsed = time.time() - t0
            print(
                f"[Step {opt_step}/{args.num_steps}] loss={accum_loss:.4f} "
                f"bce={accum_bce:.4f} constraint={accum_constraint:.4f} "
                f"sparsity={avg_sparsity:.3f} lambda={lambda_sparse:.2f} "
                f"tau={tau:.3f} lr={scheduler.get_last_lr()[0]:.2e} "
                f"elapsed={elapsed:.1f}s"
            )

    total_time = time.time() - t0
    print(f"\nBCE+Gumbel+Lagrangian training complete: {args.num_steps} steps in {total_time:.1f}s "
          f"({total_time/args.num_steps:.2f}s/step)")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    os.makedirs(args.output_dir, exist_ok=True)

    run_name = args.wandb_run_name or f"bce_gumbel_lr{args.lr}_s{args.seed}"
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
    print("Using Gumbel-sigmoid relaxation (iso-framework with KL)")

    print("Setting up data loader ...")
    dataloader = get_calibration_loader(tokenizer, args.batch_size, args.seq_len, args.seed, args.dataset)

    trainable = sum(p.numel() for p in wrapper.predictors.parameters())
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

    train_bce_gumbel(wrapper, dataloader, args, device)

    ckpt_path = os.path.join(args.output_dir, "predictor_bce_gumbel.pt")
    torch.save({"predictors": wrapper.predictors.state_dict(), "args": vars(args)}, ckpt_path)
    print(f"Saved checkpoint -> {ckpt_path}")

    wandb.finish()


if __name__ == "__main__":
    main()

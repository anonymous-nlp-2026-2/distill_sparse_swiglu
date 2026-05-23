"""Evaluate SPON bias compensation on WikiText-2.

Reports: true_dense_ppl, sparse_ppl (no SPON), spon_sparse_ppl (with SPON),
         teal_sparse_ppl (static global masks + SPON folded).
"""
import argparse
import importlib.util
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import get_eval_dataset
from predictor import SparsityPredictor
from spon import SPONBiasVectors, SPONPatcher


def _best_attn_impl():
    if importlib.util.find_spec("flash_attn"):
        return "flash_attention_2"
    return "sdpa"


def evaluate_ppl(model, dataset, device):
    """Standard dense perplexity."""
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for sample in dataset:
            ids = sample["input_ids"].unsqueeze(0).to(device)
            out = model(input_ids=ids, labels=ids)
            total_loss += out.loss.item() * (ids.size(1) - 1)
            total_tokens += ids.size(1) - 1
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()


def evaluate_ppl_patched(model, patcher, mode, dataset, device):
    """Perplexity with SPONPatcher in given mode."""
    patcher.mode = mode
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for sample in dataset:
            ids = sample["input_ids"].unsqueeze(0).to(device)
            logits = model(input_ids=ids).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = ids[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += shift_labels.numel()
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", default="/root/autodl-tmp/models/llama-3.1-8b")
    p.add_argument("--predictor_checkpoint", required=True)
    p.add_argument("--spon_checkpoint", required=True)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--max_eval_samples", type=int, default=200)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
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
    for param in model.parameters():
        param.requires_grad = False

    config = model.config
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size

    # Load predictor
    print(f"Loading predictor from {args.predictor_checkpoint} ...")
    predictors = torch.nn.ModuleList([
        SparsityPredictor(hidden_size, intermediate_size, 128)
        for _ in range(num_layers)
    ])
    ckpt = torch.load(args.predictor_checkpoint, map_location=device, weights_only=True)
    predictors.load_state_dict(ckpt["predictors"])
    predictors.to(device=device, dtype=torch.bfloat16)
    predictors.eval()
    for param in predictors.parameters():
        param.requires_grad = False

    # Load SPON biases
    print(f"Loading SPON biases from {args.spon_checkpoint} ...")
    spon_biases = SPONBiasVectors(num_layers, intermediate_size)
    spon_ckpt = torch.load(args.spon_checkpoint, map_location=device, weights_only=True)
    spon_biases.load_state_dict(spon_ckpt["spon_biases"])
    spon_biases.to(device=device, dtype=torch.bfloat16)
    spon_biases.eval()

    bias_norms = [b.data.norm().item() for b in spon_biases.biases]
    mean_bias_norm = sum(bias_norms) / len(bias_norms)
    max_bias_norm = max(bias_norms)
    print(f"SPON bias stats: mean_norm={mean_bias_norm:.6f}, max_norm={max_bias_norm:.6f}")

    # Patch model
    patcher = SPONPatcher(model, predictors, spon_biases)
    patcher.patch()

    # Load WikiText-2
    print("Loading WikiText-2 ...")
    wt2 = get_eval_dataset("wikitext2", tokenizer, args.seq_len, args.max_eval_samples)
    print(f"  {len(wt2)} sequences, seq_len={args.seq_len}")

    # 1) True dense
    print("\n[1/3] True Dense PPL ...")
    dense_ppl = evaluate_ppl_patched(model, patcher, "dense", wt2, device)
    print(f"  Dense PPL: {dense_ppl:.4f}")

    # 2) Sparse (predictor mask only, no SPON)
    print("[2/3] Sparse PPL (predictor mask, no SPON) ...")
    sparse_ppl = evaluate_ppl_patched(model, patcher, "sparse", wt2, device)
    print(f"  Sparse PPL: {sparse_ppl:.4f}")

    # 3) Sparse + SPON
    print("[3/3] Sparse + SPON PPL ...")
    spon_ppl = evaluate_ppl_patched(model, patcher, "sparse_spon", wt2, device)
    print(f"  SPON Sparse PPL: {spon_ppl:.4f}")

    patcher.unpatch()

    # Summary
    ppl_increase_sparse = 100 * (sparse_ppl - dense_ppl) / dense_ppl
    ppl_increase_spon = 100 * (spon_ppl - dense_ppl) / dense_ppl
    spon_recovery = 100 * (sparse_ppl - spon_ppl) / (sparse_ppl - dense_ppl) if sparse_ppl > dense_ppl else 0

    print(f"\n=== SPON Evaluation Summary ===")
    print(f"  True Dense PPL:     {dense_ppl:.4f}")
    print(f"  Sparse PPL:         {sparse_ppl:.4f} (+{ppl_increase_sparse:.1f}%)")
    print(f"  SPON Sparse PPL:    {spon_ppl:.4f} (+{ppl_increase_spon:.1f}%)")
    print(f"  SPON recovery:      {spon_recovery:.1f}% of sparse-vs-dense gap")
    print(f"  Mean bias norm:     {mean_bias_norm:.6f}")
    print(f"  Max bias norm:      {max_bias_norm:.6f}")


if __name__ == "__main__":
    main()

"""F1/Precision/Recall of KL and BCE predictors vs oracle magnitude-based mask."""
import importlib.util
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from data_utils import get_calibration_loader
from predictor import SparsityPredictor


def _best_attn_impl():
    if False:  # force sdpa for hook compatibility
        return "flash_attention_2"
    return "sdpa"


@torch.no_grad()
def compute_f1_vs_oracle(
    model_path="/root/autodl-tmp/models/llama-3.1-8b",
    kl_ckpt="/root/distill_sparse_swiglu/checkpoints/mvp_kl_norm_v2_s42/predictor_kl_normalized.pt",
    bce_ckpt="/root/distill_sparse_swiglu/checkpoints/bce_isocompute_v3_s42/predictor_bce.pt",
    num_samples=50,
    seq_len=2048,
    sparsity=0.5,
    device="cuda:0",
):
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, attn_implementation=_best_attn_impl()
    ).to(device).eval()

    config = model.config
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size

    # Load predictors
    predictors = {}
    for name, ckpt_path in [("kl", kl_ckpt), ("bce", bce_ckpt)]:
        if not os.path.isfile(ckpt_path):
            print(f"WARNING: {name} checkpoint not found at {ckpt_path}, skipping")
            continue
        raw = torch.load(ckpt_path, map_location=device, weights_only=True)
        state = raw["predictors"] if isinstance(raw, dict) and "predictors" in raw else raw
        preds = torch.nn.ModuleList([
            SparsityPredictor(hidden_size, intermediate_size, bottleneck_size=128)
            for _ in range(num_layers)
        ]).to(device)
        preds.load_state_dict(state)
        preds = preds.half()
        preds.eval()
        predictors[name] = preds
        print(f"Loaded {name} predictor from {ckpt_path}")

    if not predictors:
        raise RuntimeError("No predictor checkpoints found")

    # Data
    print("Loading C4 calibration data...")
    loader = get_calibration_loader(tokenizer, batch_size=1, seq_len=seq_len, seed=42, dataset_name="c4")

    # Accumulators: per-layer and global
    stats = {name: {"tp": torch.zeros(num_layers, device=device),
                    "fp": torch.zeros(num_layers, device=device),
                    "fn": torch.zeros(num_layers, device=device)}
             for name in predictors}

    # Hook to capture layer inputs and intermediates
    layer_data = {}

    def make_hook(layer_idx):
        def hook_fn(module, args, output):
            x = args[0]
            gate = module.act_fn(module.gate_proj(x))
            up = module.up_proj(x)
            intermediate = gate * up
            layer_data[layer_idx] = {"input": x, "intermediate": intermediate}
        return hook_fn

    hooks = []
    for li, layer in enumerate(model.model.layers):
        h = layer.mlp.register_forward_hook(make_hook(li))
        hooks.append(h)

    n_processed = 0
    for batch in loader:
        if n_processed >= num_samples:
            break
        input_ids = batch["input_ids"].unsqueeze(0).to(device)
        layer_data.clear()

        model(input_ids=input_ids)

        for li in range(num_layers):
            if li not in layer_data:
                continue
            intermediate = layer_data[li]["intermediate"]  # (1, seq, intermediate_size)
            hidden_input = layer_data[li]["input"]          # (1, seq, hidden_size)

            # Oracle: magnitude top-k
            mag = intermediate.abs()
            k = int(intermediate_size * sparsity)
            _, topk_idx = mag.topk(k, dim=-1)
            oracle_mask = torch.zeros_like(intermediate, dtype=torch.bool)
            oracle_mask.scatter_(-1, topk_idx, True)

            for name, preds in predictors.items():
                logits = preds[li](hidden_input)
                pred_mask = logits > 0  # hard threshold at inference

                tp = (pred_mask & oracle_mask).sum().item()
                fp = (pred_mask & ~oracle_mask).sum().item()
                fn = (~pred_mask & oracle_mask).sum().item()

                stats[name]["tp"][li] += tp
                stats[name]["fp"][li] += fp
                stats[name]["fn"][li] += fn

        n_processed += 1
        if n_processed % 10 == 0:
            print(f"  Processed {n_processed}/{num_samples} samples")

    for h in hooks:
        h.remove()

    # Compute metrics
    results = {}
    for name in predictors:
        tp = stats[name]["tp"]
        fp = stats[name]["fp"]
        fn = stats[name]["fn"]

        precision_per_layer = tp / (tp + fp + 1e-8)
        recall_per_layer = tp / (tp + fn + 1e-8)
        f1_per_layer = 2 * precision_per_layer * recall_per_layer / (precision_per_layer + recall_per_layer + 1e-8)

        tp_total = tp.sum().item()
        fp_total = fp.sum().item()
        fn_total = fn.sum().item()

        precision = tp_total / (tp_total + fp_total + 1e-8)
        recall = tp_total / (tp_total + fn_total + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        results[name] = {
            "f1": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "per_layer_f1": [round(x, 4) for x in f1_per_layer.tolist()],
            "per_layer_precision": [round(x, 4) for x in precision_per_layer.tolist()],
            "per_layer_recall": [round(x, 4) for x in recall_per_layer.tolist()],
        }

    # Random baseline
    expected_f1_random = 2 * sparsity * sparsity / (sparsity + sparsity)  # = sparsity for 50%
    results["random_baseline"] = {
        "f1": round(expected_f1_random, 4),
        "precision": round(sparsity, 4),
        "recall": round(sparsity, 4),
    }

    output = {
        "config": {
            "num_samples": num_samples,
            "seq_len": seq_len,
            "sparsity_target": sparsity,
            "model": model_path,
        },
        **results,
    }

    # Summary
    print("\n=== F1 vs Oracle Magnitude Mask (top-50%) ===")
    for name in list(predictors.keys()) + ["random_baseline"]:
        r = results[name]
        print(f"  {name:20s}: F1={r['f1']:.4f}  P={r['precision']:.4f}  R={r['recall']:.4f}")

    out_path = "/root/distill_sparse_swiglu/artifacts/kl_bce_f1_oracle.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    return output


if __name__ == "__main__":
    compute_f1_vs_oracle()

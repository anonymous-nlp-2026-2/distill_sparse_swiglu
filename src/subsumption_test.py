"""SPON Subsumption Linearity Test.

Measures whether the KL-trained predictor's sparsity mask subsumes the SPON bias
correction signal, by testing additivity of their logit-space effects.

For each input, computes four model forward passes (dense, sparse-only, spon-only,
sparse+spon) and measures:
  - linearity_ratio: ||Δ(s+c) - Δ(s) - Δ(c)|| / (||Δ(s)|| + ||Δ(c)||)
  - magnitude_ratio: ||Δ(c)|| / ||Δ(s)||
  - cos_sim: cosine similarity between Δ(s) and Δ(c)

Isolated per-layer analysis uses dense-pass MLP inputs to decompose mask vs bias
effects per layer. (Per-layer effects are exactly additive since down_proj is linear;
non-additivity at logit level arises from cross-layer cascading.)

Inputs: LLaMA model, KL predictor checkpoint, SPON bias checkpoint, WikiText-2.
Output: JSON report with global and per-layer statistics.

Dependencies: transformers, torch, datasets, predictor.py, spon.py, data_utils.py
"""

import argparse
import importlib.util
import json
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import get_eval_dataset
from predictor import SparsityPredictor
from spon import SPONBiasVectors


def _best_attn_impl():
    if importlib.util.find_spec("flash_attn"):
        return "flash_attention_2"
    return "sdpa"


class SubsumptionPatcher:
    """Patches MLP forwards for 4 modes: dense, sparse, spon_only, sparse_spon.

    Optionally captures MLP inputs during dense mode for isolated per-layer analysis.
    """

    MODES = ("dense", "sparse", "spon_only", "sparse_spon")

    def __init__(self, model, predictors, spon_biases):
        self.model = model
        self.predictors = predictors
        self.spon_biases = spon_biases
        self._saved_forwards = {}
        self.mode = "dense"
        self.capture_inputs = False
        self._layer_inputs = {}

    def patch(self):
        for idx, layer in enumerate(self.model.model.layers):
            if idx >= len(self.predictors):
                break
            self._saved_forwards[idx] = layer.mlp.forward
            self._make_patched_fwd(idx, layer.mlp)

    def unpatch(self):
        for idx, fwd in self._saved_forwards.items():
            self.model.model.layers[idx].mlp.forward = fwd
        self._saved_forwards.clear()

    def _make_patched_fwd(self, layer_idx, mlp):
        gate_proj = mlp.gate_proj
        up_proj = mlp.up_proj
        down_proj = mlp.down_proj
        act_fn = mlp.act_fn
        predictor = self.predictors[layer_idx]
        spon = self.spon_biases
        patcher = self

        def fwd(x):
            if patcher.capture_inputs and patcher.mode == "dense":
                patcher._layer_inputs[layer_idx] = x.detach()

            gate = act_fn(gate_proj(x))
            up = up_proj(x)
            intermediate = gate * up

            if patcher.mode == "dense":
                return down_proj(intermediate)
            elif patcher.mode == "sparse":
                with torch.no_grad():
                    mask = (predictor(x) > 0).to(intermediate.dtype)
                return down_proj(intermediate * mask)
            elif patcher.mode == "spon_only":
                return down_proj(spon(layer_idx, intermediate))
            elif patcher.mode == "sparse_spon":
                with torch.no_grad():
                    mask = (predictor(x) > 0).to(intermediate.dtype)
                return down_proj(spon(layer_idx, intermediate * mask))
            else:
                raise ValueError(f"Unknown mode: {patcher.mode}")

        mlp.forward = fwd

    def get_layer_inputs(self):
        return dict(self._layer_inputs)

    def clear_layer_inputs(self):
        self._layer_inputs.clear()

    def __enter__(self):
        self.patch()
        return self

    def __exit__(self, *args):
        self.unpatch()


@torch.no_grad()
def compute_per_layer_isolated(model, predictors, spon_biases, layer_inputs, num_layers):
    """Isolated per-layer mask vs bias effects using dense-pass MLP inputs.

    Since down_proj is linear, mask+bias are perfectly additive at each layer.
    Reports magnitude and direction metrics per layer.
    """
    results = []
    for layer_idx in range(num_layers):
        if layer_idx not in layer_inputs:
            continue
        x = layer_inputs[layer_idx]
        mlp = model.model.layers[layer_idx].mlp

        gate = mlp.act_fn(mlp.gate_proj(x))
        up = mlp.up_proj(x)
        intermediate = gate * up

        mask = (predictors[layer_idx](x) > 0).to(intermediate.dtype)
        bias = spon_biases.biases[layer_idx]

        # mask effect: down_proj(intermediate * (mask - 1))
        delta_s = mlp.down_proj(intermediate * (mask - 1))
        # bias effect: down_proj(bias) — constant across tokens
        delta_c = mlp.down_proj(bias.unsqueeze(0).unsqueeze(0))  # [1, 1, hidden_size]

        norm_s = delta_s.float().norm(dim=-1).mean().item()
        dc_vec = delta_c.squeeze().float()
        norm_c = dc_vec.norm().item()

        ds_2d = delta_s.reshape(-1, delta_s.size(-1)).float()
        cos = F.cosine_similarity(
            ds_2d, dc_vec.unsqueeze(0).expand_as(ds_2d), dim=-1
        ).mean().item()

        sparsity = 1.0 - mask.float().mean().item()

        results.append({
            "layer_id": layer_idx,
            "mask_effect_norm": norm_s,
            "bias_effect_norm": norm_c,
            "magnitude_ratio": norm_c / (norm_s + 1e-10),
            "cos_sim": cos,
            "mask_sparsity": sparsity,
        })

        del x, gate, up, intermediate, mask, delta_s, delta_c, ds_2d, dc_vec

    return results


def parse_args():
    p = argparse.ArgumentParser(description="SPON Subsumption Linearity Test")
    p.add_argument("--model_name_or_path", type=str,
                   default="/root/autodl-tmp/models/llama-3.1-8b")
    p.add_argument("--predictor_checkpoint", type=str,
                   default="/root/distill_sparse_swiglu/checkpoints/mvp_kl_norm_v2_s42/predictor_kl_normalized.pt")
    p.add_argument("--spon_checkpoint", type=str,
                   default="/root/distill_sparse_swiglu/checkpoints/spon_2048_s42/spon_biases.pt")
    p.add_argument("--dataset", type=str, default="wikitext2")
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--num_samples", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output_dir", type=str,
                   default="/root/distill_sparse_swiglu/results/subsumption_test")
    p.add_argument("--dry_run", action="store_true",
                   help="Validate setup without running evaluation")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    torch.manual_seed(args.seed)

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

    print(f"Loading predictor from {args.predictor_checkpoint} ...")
    predictors = torch.nn.ModuleList([
        SparsityPredictor(hidden_size, intermediate_size, 128)
        for _ in range(num_layers)
    ])
    pred_ckpt = torch.load(args.predictor_checkpoint, map_location=device, weights_only=True)
    predictors.load_state_dict(pred_ckpt["predictors"])
    predictors.to(device=device, dtype=torch.bfloat16)
    predictors.eval()
    for param in predictors.parameters():
        param.requires_grad = False

    print(f"Loading SPON biases from {args.spon_checkpoint} ...")
    spon_biases = SPONBiasVectors(num_layers, intermediate_size)
    spon_ckpt = torch.load(args.spon_checkpoint, map_location=device, weights_only=True)
    spon_biases.load_state_dict(spon_ckpt["spon_biases"])
    spon_biases.to(device=device, dtype=torch.bfloat16)
    spon_biases.eval()

    bias_norms = [b.data.norm().item() for b in spon_biases.biases]
    print(f"SPON bias stats: mean_norm={sum(bias_norms)/len(bias_norms):.6f}, "
          f"max_norm={max(bias_norms):.6f}")

    print(f"Loading {args.dataset} ...")
    dataset = get_eval_dataset(args.dataset, tokenizer, args.seq_len, args.num_samples)
    print(f"  {len(dataset)} sequences, seq_len={args.seq_len}")

    if args.dry_run:
        print("\n[DRY RUN] Setup validated.")
        print(f"  Model: {num_layers} layers, hidden={hidden_size}, intermediate={intermediate_size}")
        print(f"  Predictor params: {sum(p.numel() for p in predictors.parameters()):,}")
        print(f"  SPON bias params: {spon_biases.param_count():,}")
        print(f"  Dataset: {len(dataset)} sequences x {args.seq_len} tokens")
        return

    patcher = SubsumptionPatcher(model, predictors, spon_biases)
    patcher.patch()

    # Accumulators
    all_linearity = []
    all_magnitude = []
    all_cos_sim = []
    all_norm_s = []
    all_norm_c = []
    all_norm_int = []
    per_layer_accum = {
        i: {"magnitude": [], "cos_sim": [], "mask_sparsity": [],
            "mask_norm": [], "bias_norm": []}
        for i in range(num_layers)
    }

    print(f"\nRunning subsumption test on {len(dataset)} samples ...")
    for idx, sample in enumerate(dataset):
        input_ids = sample["input_ids"].unsqueeze(0).to(device)

        # 1) Dense forward with input capture for per-layer analysis
        patcher.mode = "dense"
        patcher.capture_inputs = True
        patcher.clear_layer_inputs()
        with torch.no_grad():
            logits_dense = model(input_ids=input_ids).logits
        layer_inputs = {k: v.clone() for k, v in patcher.get_layer_inputs().items()}
        patcher.capture_inputs = False

        # 2) Sparse (predictor mask only)
        patcher.mode = "sparse"
        with torch.no_grad():
            logits_sparse = model(input_ids=input_ids).logits

        # 3) SPON only (bias, no mask)
        patcher.mode = "spon_only"
        with torch.no_grad():
            logits_spon = model(input_ids=input_ids).logits

        # 4) Combined (mask + bias)
        patcher.mode = "sparse_spon"
        with torch.no_grad():
            logits_combined = model(input_ids=input_ids).logits

        # --- Logit-level analysis ---
        delta_s = (logits_sparse - logits_dense).float()
        delta_c = (logits_spon - logits_dense).float()
        delta_sc = (logits_combined - logits_dense).float()
        interaction = delta_sc - delta_s - delta_c

        norm_s = delta_s.norm(dim=-1).mean().item()
        norm_c = delta_c.norm(dim=-1).mean().item()
        norm_int = interaction.norm(dim=-1).mean().item()

        linearity = norm_int / (norm_s + norm_c + 1e-10)
        magnitude = norm_c / (norm_s + 1e-10)
        cos = F.cosine_similarity(
            delta_s.reshape(1, -1), delta_c.reshape(1, -1)
        ).item()

        all_linearity.append(linearity)
        all_magnitude.append(magnitude)
        all_cos_sim.append(cos)
        all_norm_s.append(norm_s)
        all_norm_c.append(norm_c)
        all_norm_int.append(norm_int)

        del logits_dense, logits_sparse, logits_spon, logits_combined
        del delta_s, delta_c, delta_sc, interaction

        # --- Isolated per-layer analysis ---
        layer_stats = compute_per_layer_isolated(
            model, predictors, spon_biases, layer_inputs, num_layers)
        for ls in layer_stats:
            lid = ls["layer_id"]
            per_layer_accum[lid]["magnitude"].append(ls["magnitude_ratio"])
            per_layer_accum[lid]["cos_sim"].append(ls["cos_sim"])
            per_layer_accum[lid]["mask_sparsity"].append(ls["mask_sparsity"])
            per_layer_accum[lid]["mask_norm"].append(ls["mask_effect_norm"])
            per_layer_accum[lid]["bias_norm"].append(ls["bias_effect_norm"])

        del layer_inputs, layer_stats
        torch.cuda.empty_cache()

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(dataset)}] linearity={linearity:.4f}, "
                  f"magnitude={magnitude:.6f}, cos_sim={cos:.4f}")

    patcher.unpatch()

    # --- Aggregate ---
    def ms(vals):
        t = torch.tensor(vals)
        return {"mean": round(t.mean().item(), 6), "std": round(t.std().item(), 6)}

    per_layer_stats = []
    for lid in range(num_layers):
        acc = per_layer_accum[lid]
        if not acc["magnitude"]:
            continue
        per_layer_stats.append({
            "layer_id": lid,
            "magnitude_ratio": ms(acc["magnitude"]),
            "cos_sim": ms(acc["cos_sim"]),
            "mask_sparsity": ms(acc["mask_sparsity"]),
            "mask_effect_norm": ms(acc["mask_norm"]),
            "bias_effect_norm": ms(acc["bias_norm"]),
        })

    results = {
        "config": {
            "model": args.model_name_or_path,
            "predictor_checkpoint": args.predictor_checkpoint,
            "spon_checkpoint": args.spon_checkpoint,
            "dataset": args.dataset,
            "seq_len": args.seq_len,
            "num_samples": len(dataset),
            "seed": args.seed,
        },
        "global_linearity_ratio": ms(all_linearity),
        "global_magnitude_ratio": ms(all_magnitude),
        "global_cos_sim": ms(all_cos_sim),
        "global_delta_s_norm": ms(all_norm_s),
        "global_delta_c_norm": ms(all_norm_c),
        "global_interaction_norm": ms(all_norm_int),
        "per_layer_stats": per_layer_stats,
        "note": (
            "Per-layer effects are exactly additive (down_proj is linear). "
            "Global linearity_ratio captures only cross-layer cascading non-linearity."
        ),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "subsumption_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # --- Summary ---
    lr = results["global_linearity_ratio"]
    mr = results["global_magnitude_ratio"]
    cs = results["global_cos_sim"]
    print(f"\n{'='*60}")
    print("SPON Subsumption Linearity Test Results")
    print(f"{'='*60}")
    print(f"Linearity ratio:   {lr['mean']:.4f} +/- {lr['std']:.4f}  (0 = additive)")
    print(f"Magnitude ratio:   {mr['mean']:.6f} +/- {mr['std']:.6f}  (||Dc||/||Ds||)")
    print(f"Cosine similarity: {cs['mean']:.4f} +/- {cs['std']:.4f}")
    print(f"||D(s)|| mean:     {results['global_delta_s_norm']['mean']:.4f}")
    print(f"||D(c)|| mean:     {results['global_delta_c_norm']['mean']:.4f}")
    print(f"||interaction||:   {results['global_interaction_norm']['mean']:.4f}")

    if lr["mean"] < 0.05 and mr["mean"] < 0.01:
        print("\n-> Strong subsumption: SPON correction is negligible and additive.")
        print("   KL predictor has already captured the SPON signal.")
    elif lr["mean"] < 0.1:
        print("\n-> Moderate subsumption: effects are approximately additive.")
    else:
        print("\n-> Weak subsumption: significant non-linear interaction detected.")


if __name__ == "__main__":
    main()

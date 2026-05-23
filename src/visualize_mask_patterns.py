"""Visualize KL-predicted vs magnitude-based mask patterns.

Generates 3 figures for EMNLP 2026 paper:
  1. Layer-wise Divergence Heatmap (Jaccard distance across layers and sparsity levels)
  2. Neuron Importance Scatter (KL vs magnitude scores for a representative layer)
  3. Concrete Mask Examples (binary mask heatmaps for early/middle/late layers)

Usage:
  cd /root/distill_sparse_swiglu
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/root/autodl-tmp/.hf_cache \
    python src/visualize_mask_patterns.py [--device cuda:1] [--calib_samples 16]
"""

import os
import sys
import gc
import argparse

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predictor import PredictorWrapper
from data_utils import get_eval_dataset
from evaluate import _collect_swiglu_magnitudes, _best_attn_impl
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = "/root/distill_sparse_swiglu"
MODEL_PATH = "/root/autodl-tmp/models/llama-3.1-8b"
CKPT_50 = os.path.join(BASE_DIR, "checkpoints/mvp_kl_norm_v2_s42/predictor_kl_normalized.pt")
CKPT_30 = os.path.join(BASE_DIR, "checkpoints/kl_sparsity30_s42_v3/predictor_kl_normalized.pt")
FIG_DIR = os.path.join(BASE_DIR, "figures")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--calib_samples", type=int, default=16)
    return p.parse_args()


def load_model_and_tokenizer(device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation=_best_attn_impl(),
        device_map={"": device},
    )
    model.eval()
    return model, tokenizer


def get_calib_ids(tokenizer, seq_len, n_samples, device):
    data = get_eval_dataset("wikitext2", tokenizer, seq_len=seq_len, max_samples=n_samples)
    return torch.stack([ex["input_ids"] for ex in data[:n_samples]]).to(device)


def _restore_mlps(model):
    """Restore original MLP forwards after PredictorWrapper patching."""
    for layer in model.model.layers:
        mlp = layer.mlp
        gp, up, dp, af = mlp.gate_proj, mlp.up_proj, mlp.down_proj, mlp.act_fn
        def _make_fwd(g, u, d, a):
            def fwd(x):
                return d(a(g(x)) * u(x))
            return fwd
        mlp.forward = _make_fwd(gp, up, dp, af)


def get_kl_scores(model, ckpt_path, calib_ids, device):
    """Per-layer mean sigmoid scores from KL predictor over calibration data."""
    wrapper = PredictorWrapper(model, bottleneck_size=128)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    wrapper.predictors.load_state_dict(ckpt["predictors"])
    wrapper.predictors.to(device=device, dtype=torch.bfloat16)
    wrapper.predictors.eval()

    layer_scores = {}
    n_batches = 0

    for start in range(0, calib_ids.size(0), 2):
        batch = calib_ids[start:start + 2]
        wrapper.forward_dense(batch, capture_intermediates=True)
        layer_inputs = wrapper.get_layer_inputs()
        with torch.no_grad():
            for li in range(len(wrapper.predictors)):
                if li not in layer_inputs:
                    continue
                logits = wrapper.predictors[li](layer_inputs[li].to(device))
                sig = torch.sigmoid(logits.float()).mean(dim=(0, 1))
                if li not in layer_scores:
                    layer_scores[li] = sig.cpu()
                else:
                    layer_scores[li] = layer_scores[li] + sig.cpu()
        wrapper._layer_intermediates.clear()
        wrapper._layer_inputs.clear()
        torch.cuda.empty_cache()
        n_batches += 1

    # Unpatch and clean up
    _restore_mlps(model)
    del wrapper
    gc.collect()
    torch.cuda.empty_cache()

    return {li: layer_scores[li] / n_batches for li in sorted(layer_scores)}


def get_magnitude_scores(model, calib_ids):
    """Per-layer mean |act(gate)*up| per neuron across calibration data."""
    return _collect_swiglu_magnitudes(model, calib_ids, calib_batch_size=2)


def scores_to_mask(scores, sparsity):
    s = scores.float()
    n = s.numel()
    k = int(n * (1.0 - sparsity))
    thr = torch.topk(s, k).values[-1]
    return (s >= thr).int()


def jaccard_distance(mask_a, mask_b):
    a = mask_a.bool()
    b = mask_b.bool()
    intersection = (a & b).sum().float()
    union = (a | b).sum().float()
    if union == 0:
        return 0.0
    return 1.0 - (intersection / union).item()


# ──────────────────────────────────────────────────────────────
# Figure 1: Layer-wise Divergence Heatmap
# Caption: Jaccard distance between KL-predicted and magnitude-based masks
# across layers and sparsity levels. Higher values indicate greater
# disagreement, revealing that KL distillation discovers fundamentally
# different sparsity patterns from activation magnitude pruning.
# ──────────────────────────────────────────────────────────────
def plot_layerwise_divergence(kl_scores_50, kl_scores_30, mag_scores, out_dir):
    num_layers = len(mag_scores)
    sparsities = [0.3, 0.5]
    kl_scores_map = {0.3: kl_scores_30, 0.5: kl_scores_50}

    heatmap = np.zeros((len(sparsities), num_layers))
    for si, sp in enumerate(sparsities):
        kl_s = kl_scores_map[sp]
        for li in range(num_layers):
            kl_mask = scores_to_mask(kl_s[li], sp)
            mag_mask = scores_to_mask(mag_scores[li].cpu(), sp)
            heatmap[si, li] = jaccard_distance(kl_mask, mag_mask)

    fig, ax = plt.subplots(figsize=(6.75, 1.6))
    im = ax.imshow(heatmap, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["30%", "50%"], fontsize=8)
    ax.set_ylabel("Sparsity", fontsize=8)
    ax.set_xlabel("Layer index", fontsize=8)
    ax.set_xticks(range(0, num_layers, 4))
    ax.set_xticklabels(range(0, num_layers, 4), fontsize=7)
    ax.tick_params(axis="both", length=2)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("Jaccard distance", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.3)
    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(out_dir, f"mask_divergence_heatmap.{fmt}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved mask_divergence_heatmap.pdf/png")
    print(f"  Jaccard dist range: {heatmap.min():.3f} - {heatmap.max():.3f}")
    print(f"  Mean Jaccard (30%): {heatmap[0].mean():.3f}, (50%): {heatmap[1].mean():.3f}")


# ──────────────────────────────────────────────────────────────
# Figure 2: Neuron Importance Scatter
# Caption: Per-neuron importance comparison between magnitude-based scoring
# (x-axis) and KL-predictor sigmoid probability (y-axis) at layer 16.
# Points are colored by mask agreement category at 50% sparsity.
# Substantial disagreement regions reveal neurons that KL distillation
# identifies as important despite low activation magnitude, and vice versa.
# ──────────────────────────────────────────────────────────────
def plot_neuron_scatter(kl_scores, mag_scores, layer_idx, sparsity, out_dir):
    kl_s = kl_scores[layer_idx].cpu().float().numpy()
    mag_s = mag_scores[layer_idx].cpu().float().numpy()

    kl_mask = scores_to_mask(kl_scores[layer_idx], sparsity).numpy().astype(bool)
    mag_mask = scores_to_mask(mag_scores[layer_idx].cpu(), sparsity).numpy().astype(bool)

    both_keep = kl_mask & mag_mask
    both_prune = (~kl_mask) & (~mag_mask)
    kl_only = kl_mask & (~mag_mask)
    mag_only = (~kl_mask) & mag_mask

    agree_pct = (both_keep.sum() + both_prune.sum()) / len(kl_mask) * 100
    disagree_pct = 100 - agree_pct

    mag_norm = (mag_s - mag_s.min()) / (mag_s.max() - mag_s.min() + 1e-12)

    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    s = 3
    alpha = 0.4
    ax.scatter(mag_norm[both_prune], kl_s[both_prune], s=s, alpha=alpha*0.5,
               c="#bbbbbb", label=f"Both prune ({both_prune.sum()})", rasterized=True)
    ax.scatter(mag_norm[both_keep], kl_s[both_keep], s=s, alpha=alpha,
               c="#2166ac", label=f"Both keep ({both_keep.sum()})", rasterized=True)
    ax.scatter(mag_norm[kl_only], kl_s[kl_only], s=s+2, alpha=0.6,
               c="#b2182b", label=f"KL-only keep ({kl_only.sum()})", rasterized=True)
    ax.scatter(mag_norm[mag_only], kl_s[mag_only], s=s+2, alpha=0.6,
               c="#f4a582", label=f"Mag-only keep ({mag_only.sum()})", rasterized=True)

    ax.axhline(0.5, color="gray", lw=0.5, ls="--", alpha=0.5)

    ax.set_xlabel("Magnitude importance (normalized)", fontsize=8)
    ax.set_ylabel("KL predictor P(keep)", fontsize=8)
    ax.set_title(f"Layer {layer_idx}, {int(sparsity*100)}% sparsity "
                 f"({disagree_pct:.1f}% disagree)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="upper left", markerscale=2.5, handletextpad=0.3,
              borderpad=0.3, labelspacing=0.3)
    fig.tight_layout(pad=0.3)
    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(out_dir, f"neuron_scatter_layer{layer_idx}.{fmt}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved neuron_scatter_layer{layer_idx}.pdf/png")
    print(f"  Agreement: {agree_pct:.1f}%, Disagree: {disagree_pct:.1f}%")
    print(f"  KL-only: {kl_only.sum()}, Mag-only: {mag_only.sum()}")


# ──────────────────────────────────────────────────────────────
# Figure 3: Concrete Mask Examples
# Caption: Binary mask comparison between KL predictor and magnitude
# pruning at 50% sparsity for layers 2 (early), 16 (middle), and 28
# (late). Each row shows a 512-neuron slice; white = keep, black = prune.
# KL masks exhibit structured patterns distinct from magnitude ordering.
# ──────────────────────────────────────────────────────────────
def plot_mask_comparison(kl_scores, mag_scores, layers, sparsity, out_dir):
    n_layers = len(layers)
    n_show = 512

    fig, axes = plt.subplots(n_layers, 2, figsize=(6.75, 0.5 + 0.7 * n_layers))
    if n_layers == 1:
        axes = axes.reshape(1, -1)

    for i, li in enumerate(layers):
        kl_mask = scores_to_mask(kl_scores[li], sparsity).numpy()[:n_show]
        mag_mask = scores_to_mask(mag_scores[li].cpu(), sparsity).numpy()[:n_show]

        rows = 8
        cols = n_show // rows
        kl_2d = kl_mask.reshape(rows, cols)
        mag_2d = mag_mask.reshape(rows, cols)

        axes[i, 0].imshow(kl_2d, cmap="gray_r", aspect="auto", interpolation="nearest")
        axes[i, 0].set_ylabel(f"L{li}", fontsize=8, rotation=0, labelpad=15)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        axes[i, 1].imshow(mag_2d, cmap="gray_r", aspect="auto", interpolation="nearest")
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])

        if i == 0:
            axes[i, 0].set_title("KL predictor", fontsize=8)
            axes[i, 1].set_title("Magnitude", fontsize=8)

        jacc = jaccard_distance(torch.tensor(kl_mask), torch.tensor(mag_mask))
        axes[i, 1].text(cols + 2, rows/2, f"J={jacc:.2f}", fontsize=7, va="center")

    fig.tight_layout(pad=0.4, h_pad=0.3)
    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(out_dir, f"mask_comparison.{fmt}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved mask_comparison.pdf/png")


def main():
    args = parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    device = args.device

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(device)

    print("Preparing calibration data...")
    calib_ids = get_calib_ids(tokenizer, args.seq_len, args.calib_samples, device)
    print(f"  Calibration shape: {calib_ids.shape}")

    print("Computing magnitude importance scores...")
    mag_scores = get_magnitude_scores(model, calib_ids)
    # Move to CPU to free GPU memory
    mag_scores = {li: s.cpu() for li, s in mag_scores.items()}
    torch.cuda.empty_cache()
    print(f"  Got scores for {len(mag_scores)} layers, dim={mag_scores[0].numel()}")

    print("Loading KL predictor (50% sparsity)...")
    kl_scores_50 = get_kl_scores(model, CKPT_50, calib_ids, device)
    print(f"  Done, {len(kl_scores_50)} layers")

    print("Loading KL predictor (30% sparsity)...")
    kl_scores_30 = get_kl_scores(model, CKPT_30, calib_ids, device)
    print(f"  Done, {len(kl_scores_30)} layers")

    print("\n=== Figure 1: Layer-wise Divergence Heatmap ===")
    plot_layerwise_divergence(kl_scores_50, kl_scores_30, mag_scores, FIG_DIR)

    print("\n=== Figure 2: Neuron Importance Scatter (Layer 16) ===")
    plot_neuron_scatter(kl_scores_50, mag_scores, layer_idx=16, sparsity=0.5, out_dir=FIG_DIR)

    print("\n=== Figure 3: Concrete Mask Comparison ===")
    plot_mask_comparison(kl_scores_50, mag_scores, layers=[2, 16, 28], sparsity=0.5, out_dir=FIG_DIR)

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()

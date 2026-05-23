import argparse
import importlib.util
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import get_eval_dataset
from predictor import PredictorWrapper
from baselines import wina_greedy_allocation_masks


def _best_attn_impl():
    if importlib.util.find_spec("flash_attn"):
        return "flash_attention_2"
    return "sdpa"

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate sparsity predictor and baselines")
    p.add_argument("--model_name_or_path", type=str,
                    default="/root/autodl-tmp/models/llama-3.1-8b")
    p.add_argument("--checkpoint", type=str, default=None,
                    help="Path to predictor .pt checkpoint (required for predictor eval)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--sparsity_target", type=float, default=0.5)
    p.add_argument("--use_random_data", action="store_true",
                    help="Use random data for dry-run eval")
    p.add_argument("--max_eval_samples", type=int, default=200)
    p.add_argument("--calibration_samples", type=int, default=32,
                    help="Number of calibration sequences for masks")
    p.add_argument("--skip_c4", action="store_true",
                    help="Skip C4 evaluation")
    p.add_argument("--baseline_mode", type=str, default=None,
                    help="Comma-separated baselines: teal_vanilla,teal_greedy,wina,wina_greedy,rsparse")
    p.add_argument("--svd_rank", type=int, default=256,
                    help="SVD rank for R-Sparse scoring")
    return p.parse_args()


def _make_original_fwd(mlp):
    """Reconstruct original LlamaMLP forward without any mask."""
    gp, up, dp, af = mlp.gate_proj, mlp.up_proj, mlp.down_proj, mlp.act_fn
    def fwd(x):
        return dp(af(gp(x)) * up(x))
    return fwd


def collect_layer_inputs(wrapper, input_ids):
    wrapper.forward_dense(input_ids, capture_intermediates=True)
    return wrapper.get_layer_inputs()


def teal_global_masks(wrapper, input_ids, sparsity_target, calib_batch_size=4, allocation="global_topk"):
    """Predictor-based TEAL: score neurons by mean sigmoid activation probability.

    Score each neuron by mean(sigmoid(logit)) across calibration tokens, then
    allocate sparsity via global top-k or uniform per-layer top-k.
    """
    accum_scores = {}
    n_batches = 0

    for start in range(0, input_ids.size(0), calib_batch_size):
        batch = input_ids[start:start + calib_batch_size]
        wrapper.forward_dense(batch, capture_intermediates=True)
        intermediates = wrapper.get_intermediates()
        layer_inputs = wrapper.get_layer_inputs()
        with torch.no_grad():
            for layer_idx in range(len(wrapper.predictors)):
                if layer_idx not in layer_inputs or layer_idx not in intermediates:
                    continue
                pred = wrapper.predictors[layer_idx]
                pred_device = next(pred.parameters()).device
                logits = pred(layer_inputs[layer_idx].to(pred_device))
                scores = torch.sigmoid(logits).mean(dim=(0, 1))
                if layer_idx not in accum_scores:
                    accum_scores[layer_idx] = scores
                else:
                    accum_scores[layer_idx] = accum_scores[layer_idx] + scores
        n_batches += 1

    layer_scores = {li: accum_scores[li] / n_batches for li in sorted(accum_scores)}
    return _dispatch_masks(layer_scores, sparsity_target, allocation)


# ---------------------------------------------------------------------------
# Training-free baseline infrastructure
# ---------------------------------------------------------------------------

@torch.no_grad()
def _collect_swiglu_magnitudes(model, input_ids, calib_batch_size=4):
    """Per-neuron mean |act(gate(x)) * up(x)| across calibration data.

    Shared calibration pass for vanilla TEAL, WINA, and R-Sparse.
    """
    accum = {}
    n_batches = 0
    saved = {}

    for li, layer in enumerate(model.model.layers):
        saved[li] = layer.mlp.forward
        mlp = layer.mlp
        gp, up, dp, af = mlp.gate_proj, mlp.up_proj, mlp.down_proj, mlp.act_fn

        def _make(idx, _gp, _up, _dp, _af):
            def fwd(x):
                h = _af(_gp(x)) * _up(x)
                mag = h.abs().mean(dim=(0, 1))
                if idx in accum:
                    accum[idx] = accum[idx] + mag
                else:
                    accum[idx] = mag
                return _dp(h)
            return fwd

        layer.mlp.forward = _make(li, gp, up, dp, af)

    for start in range(0, input_ids.size(0), calib_batch_size):
        model(input_ids[start:start + calib_batch_size])
        n_batches += 1

    for li, fwd in saved.items():
        model.model.layers[li].mlp.forward = fwd

    return {li: accum[li] / n_batches for li in sorted(accum)}


def _global_topk_masks(layer_scores, sparsity_target):
    """Global top-K mask allocation with per-layer mean normalization."""
    items = sorted(layer_scores.items())
    normalized = []
    for li, s in items:
        s_f = s.float()
        mu = s_f.mean()
        normalized.append((li, s_f / mu if mu > 1e-12 else s_f))
    all_s = torch.cat([s for _, s in normalized])
    num_keep = int(len(all_s) * (1.0 - sparsity_target))
    thr = torch.topk(all_s, num_keep).values[-1]
    return {li: (s >= thr).to(torch.bfloat16) for li, s in normalized}



def _uniform_topk_masks(layer_scores, sparsity_target):
    """Uniform per-layer allocation: each layer independently keeps top-(1-sparsity) neurons.

    Unlike global top-k which applies a single threshold across all layers (causing
    sparsity to concentrate in later layers), this ensures every layer has exactly
    the same sparsity ratio equal to sparsity_target.
    """
    masks = {}
    for li in sorted(layer_scores):
        s = layer_scores[li].float()
        num_keep = int(len(s) * (1.0 - sparsity_target))
        if num_keep >= len(s):
            masks[li] = torch.ones_like(s, dtype=torch.bfloat16)
        elif num_keep <= 0:
            masks[li] = torch.zeros_like(s, dtype=torch.bfloat16)
        else:
            thr = torch.topk(s, num_keep).values[-1]
            masks[li] = (s >= thr).to(torch.bfloat16)
        actual_sparsity = 1 - masks[li].float().mean()
        if abs(actual_sparsity - sparsity_target) >= 0.10:
            import warnings; warnings.warn(f"Layer {li}: sparsity {actual_sparsity:.3f} deviates from target {sparsity_target:.3f} by {abs(actual_sparsity - sparsity_target):.3f}")
    return masks


def _dispatch_masks(layer_scores, sparsity_target, allocation="global_topk"):
    """Route to global top-k or uniform per-layer mask allocation."""
    if allocation == "uniform":
        return _uniform_topk_masks(layer_scores, sparsity_target)
    return _global_topk_masks(layer_scores, sparsity_target)


def _print_mask_stats(name, masks):
    """Print per-layer and model-level sparsity."""
    print(f"\nPer-layer sparsity ({name}):")
    for idx in sorted(masks):
        sp = 1.0 - masks[idx].mean().item()
        print(f"  Layer {idx:2d}: {sp:.3f}")
    total_n = sum(m.numel() for m in masks.values())
    total_z = sum((m == 0).sum().item() for m in masks.values())
    sp = total_z / total_n
    print(f"  Model-level: {sp:.3f}")
    return sp


# ---------------------------------------------------------------------------
# Training-free baselines
# ---------------------------------------------------------------------------

@torch.no_grad()
def vanilla_teal_global_masks(model, input_ids, sparsity_target, calib_batch_size=4,
                               avg_mag=None, allocation="global_topk"):
    """Vanilla TEAL: score_j = mean(|act(gate(x))_j * up(x)_j|).

    Global greedy allocation by activation magnitude only (no predictor).
    Reference: TEAL (Liu et al., ICLR 2025, arxiv 2408.14690)
    """
    if avg_mag is None:
        avg_mag = _collect_swiglu_magnitudes(model, input_ids, calib_batch_size)
    return _dispatch_masks(avg_mag, sparsity_target, allocation)


@torch.no_grad()
def teal_activation_magnitude_masks(model, input_ids, sparsity_target=0.5,
                                     step_size=0.05, calib_batch_size=4):
    """Activation-magnitude scoring with greedy per-layer sparsity allocation.

    Scores neurons by mean |SiLU(gate(x)) * up(x)| on calibration data.
    Allocates per-layer sparsity by greedily increasing sparsity for the layer
    with lowest marginal reconstruction error until the target is reached.

    Args:
        model: HuggingFace CausalLM with SwiGLU MLP layers.
        input_ids: Calibration token IDs, shape (num_seqs, seq_len).
        sparsity_target: Target average sparsity across layers (0-1).
        step_size: Sparsity granularity for error curve (default 0.05).
        calib_batch_size: Batch size for calibration forward passes.

    Returns:
        dict[int, Tensor]: layer_idx -> 1-D binary mask (intermediate_size,).
    """
    n_layers = len(model.model.layers)
    inter_dim = model.model.layers[0].mlp.gate_proj.out_features
    device = next(model.parameters()).device

    # Collect MLP inputs via hooks
    bufs = {i: [] for i in range(n_layers)}
    hooks = []
    for idx, layer in enumerate(model.model.layers):
        def _hook(idx_):
            def fn(module, inp, out):
                bufs[idx_].append(inp[0].detach().cpu())
            return fn
        hooks.append(layer.mlp.register_forward_hook(_hook(idx)))

    for s in range(0, input_ids.size(0), calib_batch_size):
        model(input_ids[s:s + calib_batch_size])

    for h in hooks:
        h.remove()

    mlp_inputs = {i: torch.cat(bufs[i], dim=0) for i in range(n_layers)}
    del bufs

    # Per-layer: compute mean magnitude and reconstruction error curves
    max_sp = 0.90
    levels = [round(i * step_size, 4) for i in range(int(max_sp / step_size) + 1)]
    if levels[0] != 0.0:
        levels.insert(0, 0.0)
    error_table = {}
    mean_mags = {}
    chunk = 4

    for li in range(n_layers):
        mlp = model.model.layers[li].mlp
        gp, up, dp, af = mlp.gate_proj, mlp.up_proj, mlp.down_proj, mlp.act_fn

        accum_mag = torch.zeros(inter_dim, dtype=torch.float32)
        inter_chunks, dense_chunks = [], []
        total_tokens = 0

        for s in range(0, mlp_inputs[li].size(0), chunk):
            x = mlp_inputs[li][s:s + chunk].to(device)
            inter = af(gp(x)) * up(x)
            y_dense = dp(inter)
            accum_mag += inter.abs().float().sum(dim=(0, 1)).cpu()
            total_tokens += x.size(0) * x.size(1)
            inter_chunks.append(inter.cpu())
            dense_chunks.append(y_dense.cpu())
            del x, inter, y_dense

        mean_mag = accum_mag / total_tokens
        mean_mags[li] = mean_mag
        all_inter = torch.cat(inter_chunks, dim=0)
        all_dense = torch.cat(dense_chunks, dim=0)
        del inter_chunks, dense_chunks

        dense_mse = all_dense.float().pow(2).mean().item()
        errors = []
        for sp in levels:
            if sp <= 0.0:
                errors.append(0.0)
                continue
            nk = max(int(inter_dim * (1.0 - sp)), 1)
            thr = torch.topk(mean_mag, nk).values[-1]
            mask_1d = mean_mag >= thr

            total_err = 0.0
            total_elem = 0
            for s in range(0, all_inter.size(0), chunk):
                ic = all_inter[s:s + chunk].to(device)
                dc = all_dense[s:s + chunk].to(device)
                m = mask_1d.unsqueeze(0).unsqueeze(0).to(
                    device=device, dtype=ic.dtype)
                y_sp = dp(ic * m)
                total_err += (dc - y_sp).pow(2).sum().item()
                total_elem += dc.numel()
                del ic, dc, y_sp
            errors.append(total_err / max(total_elem, 1) / max(dense_mse, 1e-12))

        error_table[li] = errors
        del all_inter, all_dense

    del mlp_inputs
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Greedy per-layer allocation
    cur = {i: 0 for i in range(n_layers)}

    def avg_sp():
        return sum(levels[cur[i]] for i in range(n_layers)) / n_layers

    while avg_sp() < sparsity_target:
        best_l, best_m = None, float("inf")
        for li in range(n_layers):
            ci = cur[li]
            if ci + 1 >= len(levels):
                continue
            marginal = error_table[li][ci + 1] - error_table[li][ci]
            if marginal < best_m:
                best_m = marginal
                best_l = li
        if best_l is None:
            break
        cur[best_l] += 1

    masks = {}
    for li in range(n_layers):
        sp = levels[cur[li]]
        if sp <= 0:
            masks[li] = torch.ones(inter_dim, dtype=torch.bfloat16)
        else:
            nk = max(int(inter_dim * (1.0 - sp)), 1)
            thr = torch.topk(mean_mags[li], nk).values[-1]
            masks[li] = (mean_mags[li] >= thr).to(torch.bfloat16)

    return masks


@torch.no_grad()
def wina_global_masks(model, input_ids, sparsity_target, calib_batch_size=4,
                       avg_mag=None, allocation="global_topk"):
    """WINA: score_j = mean(|h_j|) * ||W_down[:, j]||_2.

    Jointly considers activation magnitude and down_proj column-wise L2 norms so
    neurons feeding into high-norm weight columns are preferentially retained.
    Reference: WINA (Chen et al., arxiv 2505.19427)
    """
    if avg_mag is None:
        avg_mag = _collect_swiglu_magnitudes(model, input_ids, calib_batch_size)
    scores = {}
    for li in avg_mag:
        # down_proj.weight: [hidden_size, intermediate_size]
        w = model.model.layers[li].mlp.down_proj.weight
        cn = w.float().norm(dim=0)  # column-wise L2 norm, shape [intermediate_size]
        scores[li] = avg_mag[li].float() * cn.to(avg_mag[li].device)
    return _dispatch_masks(scores, sparsity_target, allocation)


@torch.no_grad()
def rsparse_global_masks(model, input_ids, sparsity_target, calib_batch_size=4,
                          svd_rank=256, avg_mag=None, allocation="global_topk"):
    """R-Sparse: score_j = mean(|h_j|) * sum_{i<r} sigma_i * |Vh[i,j]|.

    Rank-aware scoring weights each neuron by its contribution to the top-r SVD
    components of down_proj, capturing the low-rank structure of the weight.
    Reference: R-Sparse (Zhang et al., ICLR 2025, arxiv 2504.19449)
    """
    if avg_mag is None:
        avg_mag = _collect_swiglu_magnitudes(model, input_ids, calib_batch_size)
    scores = {}
    for li in avg_mag:
        w = model.model.layers[li].mlp.down_proj.weight  # [hidden_size, intermediate_size]
        # SVD: W = U @ diag(S) @ Vh, Vh shape [min(H,I), I]
        _, S, Vh = torch.linalg.svd(w.float(), full_matrices=False)
        r = min(svd_rank, S.shape[0])
        # weight_score_j = sum_{i<r} S[i] * |Vh[i, j]|
        ws = (S[:r].unsqueeze(1) * Vh[:r].abs()).sum(dim=0)  # [intermediate_size]
        scores[li] = avg_mag[li].float() * ws.to(avg_mag[li].device)
    return _dispatch_masks(scores, sparsity_target, allocation)


# ---------------------------------------------------------------------------
# Perplexity evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_perplexity(model, examples, device, global_masks=None):
    """Compute perplexity with optional fixed masks on SwiGLU intermediates."""
    original_forwards = {}

    if global_masks is not None:
        for layer_idx, layer in enumerate(model.model.layers):
            if layer_idx not in global_masks:
                continue
            original_forwards[layer_idx] = layer.mlp.forward
            mask = global_masks[layer_idx].to(device=device,
                                               dtype=layer.mlp.gate_proj.weight.dtype)
            mlp = layer.mlp

            def make_masked_fwd(gp, up, dp, af, m):
                def fwd(x):
                    intermediate = af(gp(x)) * up(x)
                    return dp(intermediate * m.unsqueeze(0).unsqueeze(0))
                return fwd

            layer.mlp.forward = make_masked_fwd(
                mlp.gate_proj, mlp.up_proj, mlp.down_proj, mlp.act_fn, mask
            )

    total_loss = 0.0
    total_tokens = 0
    for ex in examples:
        ids = ex["input_ids"].unsqueeze(0).to(device)
        labels = ex["labels"].unsqueeze(0).to(device)
        logits = model(input_ids=ids).logits
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), reduction="sum"
        )
        total_loss += loss.item()
        total_tokens += labels.numel()

    for layer_idx, fwd in original_forwards.items():
        model.model.layers[layer_idx].mlp.forward = fwd

    return torch.exp(torch.tensor(total_loss / total_tokens)).item()


@torch.no_grad()
def evaluate_true_dense(model, examples, device):
    """Evaluate with original MLP forwards — no predictor, no mask."""
    patched_forwards = {}
    for idx, layer in enumerate(model.model.layers):
        patched_forwards[idx] = layer.mlp.forward
        layer.mlp.forward = _make_original_fwd(layer.mlp)

    ppl = evaluate_perplexity(model, examples, device)

    for idx, fwd in patched_forwards.items():
        model.model.layers[idx].mlp.forward = fwd

    return ppl


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")

    baselines = []
    if args.baseline_mode:
        baselines = [b.strip() for b in args.baseline_mode.split(",")]

    if not args.checkpoint and not baselines:
        print("Error: specify --checkpoint for predictor eval and/or --baseline_mode")
        sys.exit(1)

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

    if args.use_random_data:
        import random
        vocab_size = tokenizer.vocab_size
        n_samples = min(args.max_eval_samples, 10)
        def _rand_examples(n, sl, vs):
            exs = []
            for _ in range(n):
                t = [random.randint(0, vs - 1) for _ in range(sl + 1)]
                exs.append({"input_ids": torch.tensor(t[:-1], dtype=torch.long),
                            "labels": torch.tensor(t[1:], dtype=torch.long)})
            return exs
        wt2 = _rand_examples(n_samples, args.seq_len, vocab_size)
        print(f"Using random eval data: {n_samples} sequences")
    else:
        print("Loading WikiText-2 ...")
        wt2 = get_eval_dataset("wikitext2", tokenizer, args.seq_len)
        print(f"  {len(wt2)} sequences")

    calib_ids = torch.stack(
        [wt2[i]["input_ids"] for i in range(min(args.calibration_samples, len(wt2)))]
    ).to(device)

    results = {}

    # ---- Predictor-based evaluation ----
    if args.checkpoint:
        wrapper = PredictorWrapper(model, bottleneck_size=128)
        print(f"Loading predictor from {args.checkpoint} ...")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        wrapper.predictors.load_state_dict(ckpt["predictors"])
        wrapper.predictors.to(device=device, dtype=torch.bfloat16)
        wrapper.predictors.eval()
        wrapper.gumbel_mask.hard = True

        print("Computing TEAL global masks ...")
        global_masks = teal_global_masks(wrapper, calib_ids, args.sparsity_target)
        _print_mask_stats("TEAL-predictor", global_masks)

        print("\n[Pred 1/3] True Dense (WikiText-2) ...")
        results["true_dense"] = evaluate_true_dense(model, wt2, device)
        print(f"  Perplexity: {results['true_dense']:.2f}")

        print("[Pred 2/3] Predictor Hard Mask (WikiText-2) ...")
        results["predictor"] = evaluate_perplexity(model, wt2, device)
        print(f"  Perplexity: {results['predictor']:.2f}")

        print("[Pred 3/3] TEAL Sparse (WikiText-2) ...")
        results["teal_predictor"] = evaluate_perplexity(model, wt2, device, global_masks)
        print(f"  Perplexity: {results['teal_predictor']:.2f}")

    # ---- Training-free baselines ----
    if baselines:
        if "true_dense" not in results:
            print("\n[Dense] True Dense (WikiText-2) ...")
            results["true_dense"] = evaluate_true_dense(model, wt2, device)
            print(f"  Perplexity: {results['true_dense']:.2f}")

        print("\nCollecting calibration activation magnitudes ...")
        avg_mag = _collect_swiglu_magnitudes(model, calib_ids)

        for bl in baselines:
            if bl == "teal_vanilla":
                print(f"\n[Baseline] Vanilla TEAL ...")
                masks = vanilla_teal_global_masks(
                    model, calib_ids, args.sparsity_target, avg_mag=avg_mag)
                _print_mask_stats("Vanilla-TEAL", masks)
                ppl = evaluate_perplexity(model, wt2, device, masks)
                print(f"  Perplexity: {ppl:.2f}")
                results["vanilla_teal"] = ppl

            elif bl == "wina":
                print(f"\n[Baseline] WINA ...")
                masks = wina_global_masks(
                    model, calib_ids, args.sparsity_target, avg_mag=avg_mag)
                _print_mask_stats("WINA", masks)
                ppl = evaluate_perplexity(model, wt2, device, masks)
                print(f"  Perplexity: {ppl:.2f}")
                results["wina"] = ppl

            elif bl == "rsparse":
                print(f"\n[Baseline] R-Sparse (svd_rank={args.svd_rank}) ...")
                masks = rsparse_global_masks(
                    model, calib_ids, args.sparsity_target,
                    svd_rank=args.svd_rank, avg_mag=avg_mag)
                _print_mask_stats("R-Sparse", masks)
                ppl = evaluate_perplexity(model, wt2, device, masks)
                print(f"  Perplexity: {ppl:.2f}")
                results["rsparse"] = ppl

            elif bl == "teal_greedy":
                print(f"\n[Baseline] TEAL Greedy Per-Layer ...")
                masks = teal_activation_magnitude_masks(
                    model, calib_ids, args.sparsity_target)
                _print_mask_stats("TEAL-greedy", masks)
                ppl = evaluate_perplexity(model, wt2, device, masks)
                print(f"  Perplexity: {ppl:.2f}")
                results["teal_greedy"] = ppl

            elif bl == "wina_greedy":
                print(f"\n[Baseline] WINA Greedy Per-Layer ...")
                masks = wina_greedy_allocation_masks(
                    model, calib_ids, args.sparsity_target, device=device)
                _print_mask_stats("WINA-greedy", masks)
                ppl = evaluate_perplexity(model, wt2, device, masks)
                print(f"  Perplexity: {ppl:.2f}")
                results["wina_greedy"] = ppl

            else:
                print(f"  Unknown baseline: {bl}, skipping")

    # ---- Summary ----
    print(f"\n=== Summary (sparsity_target={args.sparsity_target*100:.0f}%) ===")
    for k, v in results.items():
        print(f"  {k}: {v:.2f}")


if __name__ == "__main__":
    main()

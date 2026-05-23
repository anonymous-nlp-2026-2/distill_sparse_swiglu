"""
Latency benchmark: measure inference throughput for dense vs sparse modes.
Tests: dense / predictor-only (KL sparse) / predictor+compensation
Config: batch=1, seq_len=2048, warmup=10, timing=100 forward passes
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from predictor import SparsityPredictor, CompensationNetwork, GumbelSigmoidMask

# === Config ===
MODEL_PATH = "/root/autodl-tmp/models/llama-3.1-8b"
DEVICE = "cuda:0"
BATCH_SIZE = 1
SEQ_LEN = 2048
WARMUP_ITERS = 10
TIMING_ITERS = 100

CKPT_30 = "/root/distill_sparse_swiglu/checkpoints/kl_sparsity30_s42_v3/predictor_kl_normalized.pt"
CKPT_50_COMP = "/root/distill_sparse_swiglu/checkpoints/bce_comp_staged_s42/predictor_bce_comp.pt"

# Llama-3.1-8B config
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 14336
NUM_LAYERS = 32
BOTTLENECK_SIZE = 128
COMP_BOTTLENECK = 256


def measure_latency(fn, warmup=WARMUP_ITERS, iters=TIMING_ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def load_model():
    print(f"Loading model from {MODEL_PATH} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        device_map={"": DEVICE},
    )
    model.eval()
    return model


def load_predictors(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    pred_sd = ckpt["predictors"]
    predictors = torch.nn.ModuleList([
        SparsityPredictor(HIDDEN_SIZE, INTERMEDIATE_SIZE, BOTTLENECK_SIZE)
        for _ in range(NUM_LAYERS)
    ])
    predictors.load_state_dict(pred_sd)
    predictors = predictors.half().to(device).eval()
    return predictors


def load_comp_network(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    comp_sd = ckpt["comp_network"]
    comp_net = CompensationNetwork(NUM_LAYERS, HIDDEN_SIZE, COMP_BOTTLENECK)
    comp_net.load_state_dict(comp_sd)
    comp_net = comp_net.half().to(device).eval()
    return comp_net


def patch_mlps_sparse(model, predictors, comp_net=None):
    """Patch MLPs with predictor masks (and optionally compensation)."""
    mask_fn = GumbelSigmoidMask(hard=True)
    hooks = []

    for layer_idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        gate_proj = mlp.gate_proj
        up_proj = mlp.up_proj
        down_proj = mlp.down_proj
        act_fn = mlp.act_fn
        predictor = predictors[layer_idx]
        lidx = layer_idx

        def make_patched_fwd(gp, up, dp, af, pred, cn, idx):
            def patched_forward(x):
                gate = af(gp(x))
                up_out = up(x)
                intermediate = gate * up_out
                logits = pred(x)
                mask = (logits > 0).to(x.dtype)
                result = dp(intermediate * mask)
                if cn is not None:
                    result = result + cn.forward_layer(idx, x)
                return result
            return patched_forward

        original_fwd = mlp.forward
        mlp.forward = make_patched_fwd(
            gate_proj, up_proj, down_proj, act_fn, predictor, comp_net, lidx
        )
        hooks.append((mlp, original_fwd))

    return hooks


def restore_mlps(hooks):
    for mlp, original_fwd in hooks:
        mlp.forward = original_fwd


def measure_predictor_overhead(predictors, device):
    """Measure per-layer predictor forward time."""
    x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, dtype=torch.float16, device=device)
    
    def pred_fwd():
        for p in predictors:
            _ = p(x)
    
    times = measure_latency(pred_fwd, warmup=WARMUP_ITERS, iters=TIMING_ITERS)
    return times


def measure_sparse_ffn_breakdown(model, predictors, device):
    """Measure per-layer: dense FFN vs sparse FFN (with predictor) time."""
    x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, dtype=torch.float16, device=device)
    layer = model.model.layers[0]
    mlp = layer.mlp
    predictor = predictors[0]

    # Dense FFN (single layer)
    def dense_ffn():
        _ = mlp(x)
    
    dense_times = measure_latency(dense_ffn, warmup=WARMUP_ITERS, iters=TIMING_ITERS)

    # Sparse FFN with predictor (single layer)
    gate_proj = mlp.gate_proj
    up_proj = mlp.up_proj
    down_proj = mlp.down_proj
    act_fn = mlp.act_fn

    def sparse_ffn():
        gate = act_fn(gate_proj(x))
        up_out = up_proj(x)
        intermediate = gate * up_out
        logits = predictor(x)
        mask = (logits > 0).to(x.dtype)
        _ = down_proj(intermediate * mask)

    sparse_times = measure_latency(sparse_ffn, warmup=WARMUP_ITERS, iters=TIMING_ITERS)

    return dense_times, sparse_times


def run_benchmark():
    model = load_model()
    
    # Generate dummy input
    input_ids = torch.randint(0, 32000, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    
    print(f"\nConfig: batch={BATCH_SIZE}, seq_len={SEQ_LEN}, warmup={WARMUP_ITERS}, iters={TIMING_ITERS}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"dtype: float16, attn: flash_attention_2\n")

    # === Mode 1: Dense ===
    print("--- Mode 1: Dense forward ---")
    @torch.no_grad()
    def dense_fwd():
        _ = model(input_ids)
    
    dense_times = measure_latency(dense_fwd, WARMUP_ITERS, TIMING_ITERS)
    dense_mean = np.mean(dense_times)
    dense_std = np.std(dense_times)
    print(f"  Mean: {dense_mean*1000:.2f} ± {dense_std*1000:.2f} ms")

    # === Mode 2: Predictor-only 30% ===
    print("\n--- Mode 2: KL predictor-only 30% ---")
    predictors_30 = load_predictors(CKPT_30, DEVICE)
    hooks = patch_mlps_sparse(model, predictors_30, comp_net=None)
    
    @torch.no_grad()
    def sparse30_fwd():
        _ = model(input_ids)
    
    sparse30_times = measure_latency(sparse30_fwd, WARMUP_ITERS, TIMING_ITERS)
    sparse30_mean = np.mean(sparse30_times)
    sparse30_std = np.std(sparse30_times)
    print(f"  Mean: {sparse30_mean*1000:.2f} ± {sparse30_std*1000:.2f} ms")
    restore_mlps(hooks)

    # === Mode 3: Predictor-only 50% ===
    print("\n--- Mode 3: KL predictor-only 50% ---")
    predictors_50 = load_predictors(CKPT_50_COMP, DEVICE)
    hooks = patch_mlps_sparse(model, predictors_50, comp_net=None)
    
    @torch.no_grad()
    def sparse50_fwd():
        _ = model(input_ids)
    
    sparse50_times = measure_latency(sparse50_fwd, WARMUP_ITERS, TIMING_ITERS)
    sparse50_mean = np.mean(sparse50_times)
    sparse50_std = np.std(sparse50_times)
    print(f"  Mean: {sparse50_mean*1000:.2f} ± {sparse50_std*1000:.2f} ms")
    restore_mlps(hooks)

    # === Mode 4: Predictor + compensation 30% ===
    print("\n--- Mode 4: Predictor+compensation 30% ---")
    # No comp checkpoint at 30%, use dummy comp network
    comp_net_30 = CompensationNetwork(NUM_LAYERS, HIDDEN_SIZE, COMP_BOTTLENECK).half().to(DEVICE).eval()
    hooks = patch_mlps_sparse(model, predictors_30, comp_net=comp_net_30)
    
    @torch.no_grad()
    def comp30_fwd():
        _ = model(input_ids)
    
    comp30_times = measure_latency(comp30_fwd, WARMUP_ITERS, TIMING_ITERS)
    comp30_mean = np.mean(comp30_times)
    comp30_std = np.std(comp30_times)
    print(f"  Mean: {comp30_mean*1000:.2f} ± {comp30_std*1000:.2f} ms")
    restore_mlps(hooks)

    # === Mode 5: Predictor + compensation 50% ===
    print("\n--- Mode 5: Predictor+compensation 50% ---")
    comp_net_50 = load_comp_network(CKPT_50_COMP, DEVICE)
    hooks = patch_mlps_sparse(model, predictors_50, comp_net=comp_net_50)
    
    @torch.no_grad()
    def comp50_fwd():
        _ = model(input_ids)
    
    comp50_times = measure_latency(comp50_fwd, WARMUP_ITERS, TIMING_ITERS)
    comp50_mean = np.mean(comp50_times)
    comp50_std = np.std(comp50_times)
    print(f"  Mean: {comp50_mean*1000:.2f} ± {comp50_std*1000:.2f} ms")
    restore_mlps(hooks)

    # === Predictor overhead (isolated) ===
    print("\n--- Predictor overhead (all 32 layers, isolated) ---")
    pred_times = measure_predictor_overhead(predictors_30, DEVICE)
    pred_mean = np.mean(pred_times)
    pred_std = np.std(pred_times)
    print(f"  All-layer predictor forward: {pred_mean*1000:.2f} ± {pred_std*1000:.2f} ms")
    print(f"  Per-layer average: {pred_mean*1000/NUM_LAYERS:.3f} ms")

    # === Per-layer FFN breakdown ===
    print("\n--- Per-layer FFN breakdown (layer 0) ---")
    dense_ffn_t, sparse_ffn_t = measure_sparse_ffn_breakdown(model, predictors_30, DEVICE)
    print(f"  Dense FFN (1 layer): {np.mean(dense_ffn_t)*1000:.3f} ± {np.std(dense_ffn_t)*1000:.3f} ms")
    print(f"  Sparse FFN+predictor (1 layer): {np.mean(sparse_ffn_t)*1000:.3f} ± {np.std(sparse_ffn_t)*1000:.3f} ms")

    # === Summary table ===
    tokens = BATCH_SIZE * SEQ_LEN
    results = [
        ("Dense", "0%", dense_mean, dense_std),
        ("KL predictor-only", "30%", sparse30_mean, sparse30_std),
        ("KL predictor-only", "50%", sparse50_mean, sparse50_std),
        ("Predictor+compensation", "30%", comp30_mean, comp30_std),
        ("Predictor+compensation", "50%", comp50_mean, comp50_std),
    ]

    print("\n" + "="*80)
    print(f"| {'Method':<25} | {'Sparsity':<8} | {'tokens/s':<10} | {'ms/token':<10} | {'speedup vs dense':<18} |")
    print(f"|{'-'*27}|{'-'*10}|{'-'*12}|{'-'*12}|{'-'*20}|")
    for name, sp, mean_t, std_t in results:
        tps = tokens / mean_t
        ms_per_tok = mean_t * 1000 / tokens
        speedup = dense_mean / mean_t
        print(f"| {name:<25} | {sp:<8} | {tps:<10.1f} | {ms_per_tok:<10.4f} | {speedup:<18.3f}x |")
    print("="*80)

    print(f"\n--- Raw data ---")
    print(f"Predictor overhead (32 layers): {pred_mean*1000:.2f} ms ({pred_mean/dense_mean*100:.1f}% of dense forward)")
    print(f"Dense FFN per layer: {np.mean(dense_ffn_t)*1000:.3f} ms")
    print(f"Sparse FFN+pred per layer: {np.mean(sparse_ffn_t)*1000:.3f} ms")
    print(f"Compensation overhead estimate: {(comp30_mean - sparse30_mean)*1000:.2f} ms")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERS)
    parser.add_argument("--iters", type=int, default=TIMING_ITERS)
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN)
    args = parser.parse_args()
    
    WARMUP_ITERS = args.warmup
    TIMING_ITERS = args.iters
    SEQ_LEN = args.seq_len
    
    with torch.no_grad():
        run_benchmark()

"""Measure per-step wall time for BCE training only."""
import os, sys, time, math, gc, torch
sys.path.insert(0, "/root/distill_sparse_swiglu/src")

from transformers import AutoModelForCausalLM, AutoTokenizer
from predictor import PredictorWrapper
from losses import BCESparsityLoss, SparsityRegularizer
from data_utils import get_random_loader

DEVICE = torch.device("cuda:0")
MODEL_PATH = "/root/autodl-tmp/models/llama-3.1-8b"
BATCH_SIZE = 4
SEQ_LEN = 2048
NUM_WARMUP = 3
NUM_MEASURE = 10

def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map={"": DEVICE},
        attn_implementation="sdpa",
    )
    model.eval()

    wrapper = PredictorWrapper(model, bottleneck_size=128)
    wrapper.predictors.to(device=DEVICE, dtype=torch.bfloat16)

    dataloader = get_random_loader(tokenizer.vocab_size, BATCH_SIZE, SEQ_LEN, seed=42)

    bce_fn = BCESparsityLoss(sparsity_target=0.5)
    reg_fn = SparsityRegularizer(0.5, 0.1)
    params = list(wrapper.predictors.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    wrapper.predictors.train()

    data_iter = iter(dataloader)
    num_layers = len(wrapper.predictors)
    times = []

    print(f"=== Profiling BCE ({NUM_MEASURE} steps, {NUM_WARMUP} warmup) ===")
    for i in range(NUM_WARMUP + NUM_MEASURE):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(DEVICE)
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        wrapper.forward_dense(input_ids, capture_intermediates=True)
        intermediates = wrapper.get_intermediates()
        layer_inputs = wrapper.get_layer_inputs()

        total_bce = torch.tensor(0.0, device=DEVICE)
        all_masks = {}
        for layer_idx in range(num_layers):
            if layer_idx not in intermediates:
                continue
            dense_act = intermediates[layer_idx]
            inp = layer_inputs[layer_idx]
            pred_logits = wrapper.predictors[layer_idx](inp)
            total_bce = total_bce + bce_fn(pred_logits, dense_act)
            with torch.no_grad():
                all_masks[layer_idx] = (pred_logits > 0).float()

        total_bce = total_bce / max(num_layers, 1)
        reg = reg_fn(all_masks)
        loss = total_bce + reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        if i >= NUM_WARMUP:
            times.append(elapsed)
            print(f"  BCE step {i-NUM_WARMUP}: {elapsed:.4f}s")

    bce_mean = sum(times) / len(times)
    bce_std = torch.tensor(times).std().item()

    # KL results from previous run
    kl_times = [4.5675, 4.5624, 4.5669, 4.5534, 4.5700, 4.6393, 4.5825, 4.5861, 4.5887, 4.5884]
    kl_mean = sum(kl_times) / len(kl_times)
    kl_std = torch.tensor(kl_times).std().item()
    ratio = kl_mean / bce_mean

    print(f"\n{'='*50}")
    print(f"KL  per step: {kl_mean:.4f}s (std={kl_std:.4f})")
    print(f"BCE per step: {bce_mean:.4f}s (std={bce_std:.4f})")
    print(f"Ratio (KL/BCE): {ratio:.3f}")
    print(f"\nIf KL runs 1000 steps, iso-compute BCE steps = ceil(1000 * {ratio:.3f}) = {math.ceil(1000 * ratio)}")

    # Theoretical
    config = model.config
    H = config.hidden_size
    I = config.intermediate_size
    V = config.vocab_size
    L = config.num_hidden_layers
    T = BATCH_SIZE * SEQ_LEN

    attn_flops = 2 * T * (4 * H * H)
    mlp_flops = 2 * T * (3 * H * I)
    layer_flops = attn_flops + mlp_flops
    lm_head_flops = 2 * T * H * V
    model_fwd_flops = L * layer_flops + lm_head_flops
    pred_per_layer = 2 * T * (H * 128 + 128 * I)
    pred_total = L * pred_per_layer

    kl_theory = 4 * model_fwd_flops + 3 * pred_total
    bce_theory = model_fwd_flops + 3 * pred_total
    theory_ratio = kl_theory / bce_theory

    print(f"\n--- Theoretical estimate ---")
    print(f"Model fwd FLOPs: {model_fwd_flops/1e12:.2f} TFLOPs")
    print(f"Predictor total FLOPs: {pred_total/1e12:.4f} TFLOPs ({100*pred_total/model_fwd_flops:.2f}% of model)")
    print(f"KL theory: {kl_theory/1e12:.2f} TFLOPs")
    print(f"BCE theory: {bce_theory/1e12:.2f} TFLOPs")
    print(f"Theory ratio: {theory_ratio:.3f}")

if __name__ == "__main__":
    main()

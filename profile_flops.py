"""Measure per-step wall time for KL vs BCE training to derive FLOPs ratio."""
import os, sys, time, math, torch
sys.path.insert(0, "/root/distill_sparse_swiglu/src")

from transformers import AutoModelForCausalLM, AutoTokenizer
from predictor import PredictorWrapper
from losses import KLDistillLoss, BCESparsityLoss, SparsityRegularizer
from data_utils import get_random_loader

DEVICE = torch.device("cuda:0")
MODEL_PATH = "/root/autodl-tmp/models/llama-3.1-8b"
BATCH_SIZE = 4
SEQ_LEN = 2048
NUM_WARMUP = 3
NUM_MEASURE = 10

def load_model_and_wrapper():
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
    return tokenizer, model, wrapper

def profile_kl(wrapper, dataloader):
    """Profile KL training step: dense fwd + sparse fwd + backward."""
    kl_fn = KLDistillLoss(temperature=1.0)
    params = list(wrapper.predictors.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    wrapper.predictors.train()
    wrapper.model.train()
    wrapper.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    wrapper.gumbel_mask.tau = 0.5

    data_iter = iter(dataloader)
    times = []

    for i in range(NUM_WARMUP + NUM_MEASURE):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(DEVICE)
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Dense forward (no grad)
        dense_logits = wrapper.forward_dense(input_ids).logits.detach()
        # Sparse forward (with grad)
        sparse_logits = wrapper.forward_sparse(input_ids).logits
        # Loss + backward
        kl = kl_fn(dense_logits, sparse_logits)
        masks = wrapper.get_masks()
        actual_sparsity = 1.0 - torch.stack([m.mean() for m in masks.values()]).mean()
        loss = kl + 0.1 * torch.clamp(0.5 - actual_sparsity, min=0.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        if i >= NUM_WARMUP:
            times.append(elapsed)
            print(f"  KL step {i-NUM_WARMUP}: {elapsed:.4f}s")

    wrapper.model.gradient_checkpointing_disable()
    wrapper.model.eval()
    return times

def profile_bce(wrapper, dataloader):
    """Profile BCE training step: dense fwd (capture intermediates) + predictor fwd/bwd."""
    bce_fn = BCESparsityLoss(sparsity_target=0.5)
    reg_fn = SparsityRegularizer(0.5, 0.1)
    params = list(wrapper.predictors.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    wrapper.predictors.train()

    data_iter = iter(dataloader)
    times = []
    num_layers = len(wrapper.predictors)

    for i in range(NUM_WARMUP + NUM_MEASURE):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(DEVICE)
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Dense forward (no grad, capture intermediates)
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

    return times

def main():
    print("Loading model...")
    tokenizer, model, wrapper = load_model_and_wrapper()
    vocab_size = tokenizer.vocab_size
    dataloader = get_random_loader(vocab_size, BATCH_SIZE, SEQ_LEN, seed=42)

    print(f"\n=== Profiling KL ({NUM_MEASURE} steps, {NUM_WARMUP} warmup) ===")
    kl_times = profile_kl(wrapper, dataloader)

    # Reset predictor weights for fair comparison
    for p in wrapper.predictors.parameters():
        if p.dim() >= 2:
            torch.nn.init.kaiming_uniform_(p)
        else:
            torch.nn.init.zeros_(p)

    print(f"\n=== Profiling BCE ({NUM_MEASURE} steps, {NUM_WARMUP} warmup) ===")
    bce_times = profile_bce(wrapper, dataloader)

    kl_mean = sum(kl_times) / len(kl_times)
    bce_mean = sum(bce_times) / len(bce_times)
    ratio = kl_mean / bce_mean

    print(f"\n{'='*50}")
    print(f"KL  per step: {kl_mean:.4f}s (std={torch.tensor(kl_times).std().item():.4f})")
    print(f"BCE per step: {bce_mean:.4f}s (std={torch.tensor(bce_times).std().item():.4f})")
    print(f"Ratio (KL/BCE): {ratio:.3f}")
    print(f"\nIf KL runs 1000 steps, iso-compute BCE steps = ceil(1000 * {ratio:.3f}) = {math.ceil(1000 * ratio)}")

    # Also compute theoretical ratio based on model architecture
    config = model.config
    H = config.hidden_size       # 4096
    I = config.intermediate_size # 14336
    V = config.vocab_size        # 128256
    L = config.num_hidden_layers # 32
    T = BATCH_SIZE * SEQ_LEN     # tokens per step

    # Per-layer FLOPs (forward only, 2*M*N per matmul)
    attn_flops = 2 * T * (4 * H * H)  # Q,K,V,O projections (simplified, ignoring GQA)
    mlp_flops = 2 * T * (3 * H * I)   # gate + up + down
    layer_flops = attn_flops + mlp_flops
    lm_head_flops = 2 * T * H * V
    model_fwd_flops = L * layer_flops + lm_head_flops

    pred_per_layer = 2 * T * (H * 128 + 128 * I)  # down + up
    pred_total = L * pred_per_layer

    # KL: dense_fwd(1x) + sparse_fwd(1x) + backward(~2x with grad ckpt) = ~4x model fwd
    kl_theory = 4 * model_fwd_flops + 3 * pred_total
    # BCE: dense_fwd(1x) + pred_fwd+bwd(~3x pred) 
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

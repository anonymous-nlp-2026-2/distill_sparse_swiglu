"""Diagnostic: verify dynamic predictor mask in eval_benchmark.py patching."""
import os, sys, torch, torch.nn as nn
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/.hf_cache")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predictor import SparsityPredictor
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda:0")
MODEL_PATH = "/root/autodl-tmp/models/llama-3.1-8b"
CKPT_PATH = "/root/distill_sparse_swiglu/checkpoints/mvp_kl_norm_v2_s42/predictor_kl_normalized.pt"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, device_map={"": device}, attn_implementation="sdpa")
model.eval()

config = model.config
print(f"hidden_size={config.hidden_size}, intermediate_size={config.intermediate_size}, layers={config.num_hidden_layers}")

# Load predictors (same as eval_benchmark.py)
predictors = nn.ModuleList([
    SparsityPredictor(config.hidden_size, config.intermediate_size, 128)
    for _ in range(config.num_hidden_layers)
])
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
predictors.load_state_dict(ckpt["predictors"])
predictors.to(device=device, dtype=torch.bfloat16).eval()
print(f"Loaded {len(predictors)} predictors from checkpoint")

# === Test 1: Check mask shape and sparsity on sample input ===
print("\n=== Test 1: Mask shape & sparsity ===")
text = "Question: What is the capital of France?\nAnswer:"
ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
print(f"Input shape: {ids.shape}")

# Get hidden states from each layer
mask_sparsities = []
with torch.no_grad():
    outputs = model(input_ids=ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # (n_layers+1, batch, seq, hidden)
    
    for li in range(config.num_hidden_layers):
        h = hidden_states[li]  # input to layer li
        logits = predictors[li](h)
        mask = (logits > 0).float()
        sp = 1.0 - mask.mean().item()
        mask_sparsities.append(sp)
        if li < 5 or li >= config.num_hidden_layers - 3:
            print(f"  Layer {li:2d}: mask shape={mask.shape}, sparsity={sp:.4f}, "
                  f"logits range=[{logits.min().item():.2f}, {logits.max().item():.2f}]")
    
    avg_sp = sum(mask_sparsities) / len(mask_sparsities)
    print(f"\n  Average sparsity: {avg_sp:.4f}")
    min_sp = min(mask_sparsities)
    max_sp = max(mask_sparsities)
    print(f"  Min sparsity: {min_sp:.4f} (layer {mask_sparsities.index(min_sp)})")
    print(f"  Max sparsity: {max_sp:.4f} (layer {mask_sparsities.index(max_sp)})")

# === Test 2: Verify patched forward actually applies mask ===
print("\n=== Test 2: Verify patch_predictor applies mask ===")
# Dense forward
with torch.no_grad():
    dense_logits = model(input_ids=ids).logits
    dense_probs = torch.softmax(dense_logits[0, -1], dim=-1)
    dense_top5 = torch.topk(dense_probs, 5)
    print(f"Dense top-5 tokens: {[tokenizer.decode([t]) for t in dense_top5.indices.tolist()]}")
    print(f"Dense top-5 probs:  {dense_top5.values.tolist()}")

# Patch model (same as eval_benchmark.py)
for idx, layer in enumerate(model.model.layers):
    mlp = layer.mlp
    gp, up, dp, af = mlp.gate_proj, mlp.up_proj, mlp.down_proj, mlp.act_fn
    pred = predictors[idx]
    def fwd(x, _gp=gp, _up=up, _dp=dp, _af=af, _pred=pred):
        inter = _af(_gp(x)) * _up(x)
        with torch.no_grad():
            mask = (_pred(x) > 0).to(inter.dtype)
        return _dp(inter * mask)
    mlp.forward = fwd

# Sparse forward
with torch.no_grad():
    sparse_logits = model(input_ids=ids).logits
    sparse_probs = torch.softmax(sparse_logits[0, -1], dim=-1)
    sparse_top5 = torch.topk(sparse_probs, 5)
    print(f"\nSparse top-5 tokens: {[tokenizer.decode([t]) for t in sparse_top5.indices.tolist()]}")
    print(f"Sparse top-5 probs:  {sparse_top5.values.tolist()}")

# Check if outputs differ
diff = (dense_logits - sparse_logits).abs().mean().item()
print(f"\nMean |dense - sparse| logit diff: {diff:.4f}")
if diff < 0.001:
    print("WARNING: Logits barely differ - mask may not be applied!")
else:
    print("OK: Logits differ significantly - mask IS being applied")

# === Test 3: Check mask behavior with HFLM-style usage ===
print("\n=== Test 3: MMLU-style log-likelihood comparison ===")
prompt = "The capital of France is"
options = [" Paris", " London", " Berlin", " Madrid"]
for opt in options:
    full_text = prompt + opt
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        logits = model(input_ids=full_ids).logits
    # Log-likelihood of the option token(s)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    n_prompt = prompt_ids.shape[1]
    opt_logprobs = 0.0
    for i in range(n_prompt, full_ids.shape[1]):
        token_logprob = torch.log_softmax(logits[0, i-1], dim=-1)[full_ids[0, i]].item()
        opt_logprobs += token_logprob
    print(f"  '{opt}': logprob = {opt_logprobs:.4f}")

# === Test 4: Check sparsity in actual forward with hook ===
print("\n=== Test 4: Actual mask sparsity during forward ===")
actual_sparsities = {}
def make_spy_fwd(idx, _gp, _up, _dp, _af, _pred):
    def fwd(x):
        inter = _af(_gp(x)) * _up(x)
        with torch.no_grad():
            mask = (_pred(x) > 0).to(inter.dtype)
        sp = 1.0 - mask.mean().item()
        actual_sparsities[idx] = sp
        return _dp(inter * mask)
    return fwd

for idx, layer in enumerate(model.model.layers):
    mlp = layer.mlp
    gp, up, dp, af = mlp.gate_proj, mlp.up_proj, mlp.down_proj, mlp.act_fn
    pred = predictors[idx]
    mlp.forward = make_spy_fwd(idx, gp, up, dp, af, pred)

with torch.no_grad():
    _ = model(input_ids=ids)

print(f"  Layers with mask: {len(actual_sparsities)}")
sp_vals = list(actual_sparsities.values())
print(f"  Average actual sparsity: {sum(sp_vals)/len(sp_vals):.4f}")
print(f"  Range: [{min(sp_vals):.4f}, {max(sp_vals):.4f}]")

# Check if sparsity is ~50% as expected
if abs(sum(sp_vals)/len(sp_vals) - 0.5) > 0.15:
    print(f"  WARNING: Average sparsity deviates significantly from target 0.5!")

print("\nDone.")

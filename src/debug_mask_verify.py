"""Quick diagnostic: verify mask is actually applied during forward passes."""
import os
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/.hf_cache")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from predictor import SparsityPredictor
import torch.nn as nn

MODEL_PATH = "/root/autodl-tmp/models/llama-3.1-8b"
CKPT_PATH = "/root/distill_sparse_swiglu/checkpoints/mvp_kl_norm_v2_s42/predictor_kl_normalized.pt"
device = "cuda:0"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map={"": device})
model.eval()

# Test input
text = "The capital of France is"
inputs = tokenizer(text, return_tensors="pt").to(device)

# 1. Dense forward
print("\n--- Dense forward ---")
with torch.no_grad():
    dense_out = model(**inputs)
dense_logits = dense_out.logits[0, -1]  # last token logits
top5_dense = torch.topk(dense_logits, 5)
print(f"Top-5 tokens: {[tokenizer.decode(t) for t in top5_dense.indices]}")
print(f"Top-5 logits: {top5_dense.values.tolist()}")

# 2. Load predictors
print("\n--- Loading predictor ---")
cfg = model.config
predictors = nn.ModuleList([
    SparsityPredictor(cfg.hidden_size, cfg.intermediate_size, 128)
    for _ in range(cfg.num_hidden_layers)
])
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
print(f"Checkpoint keys: {list(ckpt.keys())}")
predictors.load_state_dict(ckpt["predictors"])
predictors.to(device=device, dtype=torch.bfloat16).eval()

# 3. Check predictor output on a sample layer
# Capture hidden states from layer 0 input
hook_data = {}
def make_hook(name):
    def hook_fn(module, input, output):
        hook_data[name] = input[0].detach()
    return hook_fn

handle = model.model.layers[0].mlp.register_forward_hook(make_hook("layer0_mlp"))
with torch.no_grad():
    _ = model(**inputs)
handle.remove()

x0 = hook_data["layer0_mlp"]
print(f"\nLayer 0 MLP input shape: {x0.shape}")

with torch.no_grad():
    logits_0 = predictors[0](x0)
    mask_0 = (logits_0 > 0).float()
print(f"Predictor logits range: [{logits_0.min():.4f}, {logits_0.max():.4f}]")
print(f"Mask sparsity (layer 0): {1.0 - mask_0.mean().item():.4f}")
print(f"Per-token sparsity (layer 0):")
for t in range(mask_0.shape[1]):
    sp = 1.0 - mask_0[0, t].mean().item()
    if t < 5 or t == mask_0.shape[1]-1:
        tok = tokenizer.decode(inputs.input_ids[0, t])
        print(f"  Token {t} ('{tok}'): sparsity={sp:.4f}")

# 4. Patch model and verify forward differs
print("\n--- Patching model ---")
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

with torch.no_grad():
    sparse_out = model(**inputs)
sparse_logits = sparse_out.logits[0, -1]
top5_sparse = torch.topk(sparse_logits, 5)
print(f"Top-5 tokens: {[tokenizer.decode(t) for t in top5_sparse.indices]}")
print(f"Top-5 logits: {top5_sparse.values.tolist()}")

logit_diff = (dense_logits - sparse_logits).abs()
print(f"\nLogit diff: mean={logit_diff.mean():.4f}, max={logit_diff.max():.4f}")
print(f"Dense top-1 = Sparse top-1? {top5_dense.indices[0].item() == top5_sparse.indices[0].item()}")

# 5. Check if mask is really active — count zeros in intermediate activations
print("\n--- Verifying mask activation in forward ---")
inter_data = {}
def make_inter_hook(name, pred):
    def hook_fn(module, input, output):
        x = input[0]
        with torch.no_grad():
            logits = pred(x)
            mask = (logits > 0).float()
            sp = 1.0 - mask.mean().item()
            inter_data[name] = sp
    return hook_fn

handles = []
for idx, layer in enumerate(model.model.layers):
    h = layer.mlp.register_forward_hook(make_inter_hook(f"layer{idx}", predictors[idx]))
    handles.append(h)

with torch.no_grad():
    _ = model(**inputs)

for h in handles:
    h.remove()

sparsities = list(inter_data.values())
print(f"Mean per-layer sparsity: {sum(sparsities)/len(sparsities):.4f}")
print(f"Min: {min(sparsities):.4f}, Max: {max(sparsities):.4f}")
for i in [0, 1, 15, 16, 30, 31]:
    print(f"  Layer {i}: {inter_data.get(f'layer{i}', 'N/A'):.4f}")

# 6. Quick lm-eval test: dense only, small limit
print("\n--- Quick lm-eval dense sanity check ---")
# Unpatch
for layer in model.model.layers:
    if "forward" in layer.mlp.__dict__:
        del layer.mlp.__dict__["forward"]

try:
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    lm_obj = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=4)
    out = lm_eval.simple_evaluate(
        model=lm_obj,
        tasks=["mmlu"],
        num_fewshot=5,
        limit=100,
    )
    if "results" in out:
        for task, res in out["results"].items():
            for k, v in res.items():
                if isinstance(v, float):
                    print(f"  {task}/{k}: {v:.4f}")
except Exception as e:
    print(f"  lm-eval error: {e}")

# 7. Re-patch and test sparse on same subset
print("\n--- Quick lm-eval sparse sanity check ---")
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

try:
    lm_obj2 = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=4)
    out2 = lm_eval.simple_evaluate(
        model=lm_obj2,
        tasks=["mmlu"],
        num_fewshot=5,
        limit=100,
    )
    if "results" in out2:
        for task, res in out2["results"].items():
            for k, v in res.items():
                if isinstance(v, float):
                    print(f"  {task}/{k}: {v:.4f}")
except Exception as e:
    print(f"  lm-eval error: {e}")

print("\nDone.")

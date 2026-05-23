"""Diagnostic: Test HFLM wrapper with patched model on MMLU (limit=20).
Compares: dense lm-eval vs sparse lm-eval vs manual sparse."""
import os, sys, torch, torch.nn as nn
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/.hf_cache")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

# === Test 1: Dense MMLU through lm-eval ===
print("\n=== Test 1: Dense MMLU via lm-eval (limit=20) ===")
import lm_eval
from lm_eval.models.huggingface import HFLM

lm_obj = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=4)
out = lm_eval.simple_evaluate(model=lm_obj, tasks=["mmlu"], num_fewshot=5, limit=20)
dense_mmlu = out["results"]["mmlu"]["acc,none"]
print(f"Dense MMLU (lm-eval, 20 samples): {dense_mmlu:.4f}")

# === Test 2: Patch model, then run MMLU through lm-eval ===
print("\n=== Test 2: Sparse MMLU via lm-eval (limit=20) ===")
config = model.config
predictors = nn.ModuleList([
    SparsityPredictor(config.hidden_size, config.intermediate_size, 128)
    for _ in range(config.num_hidden_layers)
])
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
predictors.load_state_dict(ckpt["predictors"])
predictors.to(device=device, dtype=torch.bfloat16).eval()

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

# Verify patch is active
print("Verifying patch...")
test_ids = tokenizer("Hello", return_tensors="pt").input_ids.to(device)
with torch.no_grad():
    out1 = model(test_ids).logits
# Check if forward is patched
print(f"  MLP forward patched: {'forward' in model.model.layers[0].mlp.__dict__}")

# Create NEW HFLM with the patched model
lm_obj_sparse = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=4)

# Check if HFLM preserved the patch
print(f"  HFLM model MLP forward patched: {'forward' in lm_obj_sparse._model.model.layers[0].mlp.__dict__}")
print(f"  Same object? {lm_obj_sparse._model is model}")

out = lm_eval.simple_evaluate(model=lm_obj_sparse, tasks=["mmlu"], num_fewshot=5, limit=20)
sparse_mmlu = out["results"]["mmlu"]["acc,none"]
print(f"Sparse MMLU (lm-eval, 20 samples): {sparse_mmlu:.4f}")

# === Test 3: Check if patch survived after lm-eval ===
print("\n=== Test 3: Post lm-eval patch check ===")
print(f"  MLP forward still patched: {'forward' in model.model.layers[0].mlp.__dict__}")
with torch.no_grad():
    out2 = model(test_ids).logits
diff = (out1 - out2).abs().mean().item()
print(f"  Pre vs post lm-eval logit diff: {diff:.6f} (should be ~0 if patch unchanged)")

print(f"\n=== Summary ===")
print(f"Dense  MMLU (lm-eval): {dense_mmlu:.4f}")
print(f"Sparse MMLU (lm-eval): {sparse_mmlu:.4f}")

"""Quick mask diagnostic - unbuffered."""
import os, sys
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/.hf_cache")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from predictor import SparsityPredictor

def p(msg): print(msg, flush=True)

MODEL_PATH = "/root/autodl-tmp/models/llama-3.1-8b"
CKPT_PATH = "/root/distill_sparse_swiglu/checkpoints/mvp_kl_norm_v2_s42/predictor_kl_normalized.pt"
device = "cuda:0"

p("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, device_map={"": device})
model.eval()
p(f"Model loaded. Config: hidden={model.config.hidden_size}, inter={model.config.intermediate_size}, layers={model.config.num_hidden_layers}")

text = "The capital of France is"
inputs = tokenizer(text, return_tensors="pt").to(device)
p(f"Input tokens: {inputs.input_ids.shape}")

# 1. Dense forward
p("\n=== Dense forward ===")
with torch.no_grad():
    dense_out = model(**inputs)
dense_logits = dense_out.logits[0, -1]
top5 = torch.topk(dense_logits, 5)
p(f"Top-5: {[(tokenizer.decode(t), round(v.item(),2)) for t,v in zip(top5.indices, top5.values)]}")

# 2. Load predictor checkpoint
p("\n=== Loading predictor ===")
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
p(f"Checkpoint keys: {list(ckpt.keys())}")
if "predictors" in ckpt:
    pstate = ckpt["predictors"]
    p(f"Predictor state keys count: {len(pstate)}")
    first_key = list(pstate.keys())[0]
    p(f"First key: {first_key}, shape: {pstate[first_key].shape}")

cfg = model.config
predictors = nn.ModuleList([
    SparsityPredictor(cfg.hidden_size, cfg.intermediate_size, 128)
    for _ in range(cfg.num_hidden_layers)
])
predictors.load_state_dict(ckpt["predictors"])
predictors.to(device=device, dtype=torch.bfloat16).eval()
p(f"Predictors loaded: {len(predictors)} layers")
p(f"Predictor[0]: down={predictors[0].down.weight.shape}, up={predictors[0].up.weight.shape}")

# 3. Check predictor output on layer 0
p("\n=== Predictor output check ===")
hook_data = {}
def make_hook(name):
    def fn(module, inp, out):
        hook_data[name] = inp[0].detach()
    return fn

h = model.model.layers[0].mlp.register_forward_hook(make_hook("l0"))
with torch.no_grad():
    _ = model(**inputs)
h.remove()

x0 = hook_data["l0"]
p(f"Layer 0 MLP input: {x0.shape}, dtype={x0.dtype}")

with torch.no_grad():
    logits_0 = predictors[0](x0)
    mask_0 = (logits_0 > 0).float()

p(f"Predictor logits: shape={logits_0.shape}, range=[{logits_0.min():.4f}, {logits_0.max():.4f}]")
p(f"Mask sparsity (layer 0): {1.0 - mask_0.mean().item():.4f}")
for t in range(min(5, mask_0.shape[1])):
    sp = 1.0 - mask_0[0, t].mean().item()
    tok = tokenizer.decode(inputs.input_ids[0, t])
    p(f"  Token {t} ('{tok}'): sparsity={sp:.4f}")

# 4. Compute per-layer sparsity
p("\n=== Per-layer sparsity ===")
all_sp = []
for idx in range(cfg.num_hidden_layers):
    h = model.model.layers[idx].mlp.register_forward_hook(make_hook(f"l{idx}"))
    
with torch.no_grad():
    _ = model(**inputs)

for idx in range(cfg.num_hidden_layers):
    model.model.layers[idx].mlp._forward_hooks.clear()

for idx in range(cfg.num_hidden_layers):
    x_i = hook_data.get(f"l{idx}")
    if x_i is None:
        p(f"  Layer {idx}: NO DATA")
        continue
    with torch.no_grad():
        lg = predictors[idx](x_i)
        m = (lg > 0).float()
        sp = 1.0 - m.mean().item()
        all_sp.append(sp)
        if idx < 3 or idx > 28:
            p(f"  Layer {idx}: sparsity={sp:.4f}")

p(f"  Mean sparsity: {sum(all_sp)/len(all_sp):.4f}")

# 5. Patch and verify
p("\n=== Patch model ===")
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
top5s = torch.topk(sparse_logits, 5)
p(f"Sparse Top-5: {[(tokenizer.decode(t), round(v.item(),2)) for t,v in zip(top5s.indices, top5s.values)]}")

diff = (dense_logits - sparse_logits).abs()
p(f"Logit diff: mean={diff.mean():.4f}, max={diff.max():.4f}")
p(f"Same top-1? {top5.indices[0].item() == top5s.indices[0].item()}")

# 6. Verify patch persists through HFLM
p("\n=== HFLM verification ===")
try:
    from lm_eval.models.huggingface import HFLM
    lm_obj = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=4)
    # Call the model through HFLM
    test_ids = inputs.input_ids
    with torch.no_grad():
        hflm_logits = lm_obj._model_call(test_ids)
    p(f"HFLM output shape: {hflm_logits.shape}")
    hflm_last = hflm_logits[0, -1]
    diff2 = (sparse_logits - hflm_last).abs()
    p(f"HFLM vs direct sparse: mean_diff={diff2.mean():.6f}, max_diff={diff2.max():.6f}")
    if diff2.max() < 0.01:
        p("HFLM uses patched model correctly (logits match sparse)")
    else:
        p("WARNING: HFLM logits differ from sparse! Patch may not persist!")
        diff3 = (dense_logits - hflm_last).abs()
        p(f"HFLM vs dense: mean_diff={diff3.mean():.6f}")
except Exception as e:
    p(f"HFLM error: {e}")

p("\nDone.")

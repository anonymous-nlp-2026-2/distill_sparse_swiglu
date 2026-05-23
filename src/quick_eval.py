"""Quick eval for remaining conditions: teal_vanilla, wina, rsparse, uniform_50, plus sparsity70 TEAL."""
import os, sys, json, time, torch, torch.nn.functional as F
os.environ["HF_HOME"] = "/root/autodl-tmp/.hf_cache"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import get_eval_dataset
from evaluate import (vanilla_teal_global_masks, wina_global_masks,
                      rsparse_global_masks, _collect_swiglu_magnitudes,
                      teal_global_masks, _global_topk_masks)
from predictor import PredictorWrapper

import importlib.util
from transformers import AutoModelForCausalLM, AutoTokenizer

def _best_attn():
    if importlib.util.find_spec("flash_attn"):
        return "flash_attention_2"
    return "sdpa"

@torch.no_grad()
def compute_ppl(model, dataset, device):
    total_loss = 0.0
    total_tokens = 0
    for sample in dataset:
        ids = sample["input_ids"].unsqueeze(0).to(device)
        labels = sample["labels"].unsqueeze(0).to(device)
        logits = model(input_ids=ids).logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += labels.numel()
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()

device = torch.device("cuda:0")
MODEL_PATH = "/root/autodl-tmp/models/llama-3.1-8b"

print("Loading model...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map={"": device},
    attn_implementation=_best_attn())
model.eval()
for p in model.parameters():
    p.requires_grad = False

print("Loading WikiText-2...", flush=True)
wt2 = get_eval_dataset("wikitext2", tokenizer, 2048, 0)
print(f"  {len(wt2)} sequences", flush=True)

calib_ids = torch.stack([s["input_ids"] for s in wt2[:32]]).to(device)

results = {}

# Collect calibration magnitudes
print("Collecting calibration magnitudes...", flush=True)
avg_mag = _collect_swiglu_magnitudes(model, calib_ids)

# Helper: patch static masks
def patch_and_eval(name, masks):
    originals = {}
    for idx, layer in enumerate(model.model.layers):
        if idx not in masks:
            continue
        originals[idx] = layer.mlp.forward
        mask = masks[idx].to(device=device, dtype=torch.bfloat16).unsqueeze(0).unsqueeze(0)
        mlp = layer.mlp
        gp, up, dp, af = mlp.gate_proj, mlp.up_proj, mlp.down_proj, mlp.act_fn
        def fwd(x, _gp=gp, _up=up, _dp=dp, _af=af, _m=mask):
            return _dp(_af(_gp(x)) * _up(x) * _m)
        mlp.forward = fwd
    t0 = time.time()
    ppl = compute_ppl(model, wt2, device)
    elapsed = time.time() - t0
    for idx, f in originals.items():
        model.model.layers[idx].mlp.forward = f
    results[name] = {"ppl": round(ppl, 4), "elapsed_s": round(elapsed, 1)}
    print(f"  {name}: {ppl:.4f} ({elapsed:.0f}s)", flush=True)
    return ppl

# 1. Vanilla TEAL
print("\n[1/5] Vanilla TEAL...", flush=True)
masks = vanilla_teal_global_masks(model, calib_ids, 0.5, avg_mag=avg_mag)
patch_and_eval("teal_vanilla", masks)

# 2. WINA
print("[2/5] WINA...", flush=True)
masks = wina_global_masks(model, calib_ids, 0.5, avg_mag=avg_mag)
patch_and_eval("wina", masks)

# 3. R-Sparse
print("[3/5] R-Sparse...", flush=True)
masks = rsparse_global_masks(model, calib_ids, 0.5, avg_mag=avg_mag)
patch_and_eval("rsparse", masks)

# 4. Uniform 50%
print("[4/5] Uniform 50%...", flush=True)
n_layers = len(model.model.layers)
inter_dim = model.model.layers[0].mlp.gate_proj.out_features
uniform_masks = {}
for li in range(n_layers):
    mag = avg_mag[li].float()
    nk = max(int(inter_dim * 0.5), 1)
    thr = torch.topk(mag, nk).values[-1]
    uniform_masks[li] = (mag >= thr).to(torch.bfloat16)
patch_and_eval("uniform_50", uniform_masks)

# 5. Sparsity70 TEAL (Part F)
print("[5/5] Sparsity70 TEAL re-eval...", flush=True)
sp70_ckpt = "/root/distill_sparse_swiglu/checkpoints/kl_sparsity70_s42/predictor_kl_normalized.pt"
if os.path.exists(sp70_ckpt):
    wrapper = PredictorWrapper(model.config, model.model.layers)
    ckpt = torch.load(sp70_ckpt, map_location=device, weights_only=True)
    wrapper.predictors.load_state_dict(ckpt["predictors"])
    wrapper.predictors.to(device=device, dtype=torch.bfloat16)
    wrapper.predictors.eval()
    wrapper.gumbel_mask.hard = True
    sp70_masks = teal_global_masks(wrapper, calib_ids, 0.7)
    patch_and_eval("sparsity70_teal", sp70_masks)
else:
    print("  SKIP: checkpoint not found", flush=True)

# Save
out_path = "/root/distill_sparse_swiglu/results/plan_007_fixed_baselines.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nAll results saved to {out_path}", flush=True)
print(json.dumps(results, indent=2), flush=True)

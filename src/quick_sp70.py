"""Sparsity70 TEAL re-eval. Traps SIGTERM."""
import signal, os, sys, json, time, torch, torch.nn.functional as F
signal.signal(signal.SIGTERM, signal.SIG_IGN)

os.environ["HF_HOME"] = "/root/autodl-tmp/.hf_cache"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import get_eval_dataset
from evaluate import teal_global_masks
from predictor import PredictorWrapper
import importlib.util
from transformers import AutoModelForCausalLM, AutoTokenizer

def _best_attn():
    return "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"

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

# Create wrapper with the full model (correct API)
wrapper = PredictorWrapper(model, bottleneck_size=128)
ckpt = torch.load(
    "/root/distill_sparse_swiglu/checkpoints/kl_sparsity70_s42/predictor_kl_normalized.pt",
    map_location=device, weights_only=True)
wrapper.predictors.load_state_dict(ckpt["predictors"])
wrapper.predictors.to(device=device, dtype=torch.bfloat16)
wrapper.predictors.eval()
wrapper.gumbel_mask.hard = True

print("Computing sparsity70 TEAL masks...", flush=True)
sp70_masks = teal_global_masks(wrapper, calib_ids, 0.7)

# Print mask stats
total_n = sum(m.numel() for m in sp70_masks.values())
total_z = sum((m == 0).sum().item() for m in sp70_masks.values())
print(f"  Sparsity: {total_z/total_n:.3f}", flush=True)

# Eval with static masks
originals = {}
for idx, layer in enumerate(model.model.layers):
    if idx not in sp70_masks:
        continue
    originals[idx] = layer.mlp.forward
    mask = sp70_masks[idx].to(device=device, dtype=torch.bfloat16).unsqueeze(0).unsqueeze(0)
    mlp = layer.mlp
    gp, up, dp, af = mlp.gate_proj, mlp.up_proj, mlp.down_proj, mlp.act_fn
    def fwd(x, _gp=gp, _up=up, _dp=dp, _af=af, _m=mask):
        return _dp(_af(_gp(x)) * _up(x) * _m)
    mlp.forward = fwd

print("Evaluating PPL...", flush=True)
t0 = time.time()
ppl = compute_ppl(model, wt2, device)
elapsed = time.time() - t0
print(f"  sparsity70_teal PPL: {ppl:.4f} ({elapsed:.0f}s)", flush=True)

# Restore
for idx, f in originals.items():
    model.model.layers[idx].mlp.forward = f

# Also eval hard mask
wrapper.sparse_mode = True
print("Evaluating hard mask PPL...", flush=True)
t0 = time.time()
# Need to patch for hard mask evaluation
for idx, layer in enumerate(model.model.layers):
    originals[idx] = layer.mlp.forward
    mlp = layer.mlp
    gp, up, dp, af = mlp.gate_proj, mlp.up_proj, mlp.down_proj, mlp.act_fn
    pred = wrapper.predictors[idx]
    def fwd(x, _gp=gp, _up=up, _dp=dp, _af=af, _pred=pred):
        inter = _af(_gp(x)) * _up(x)
        with torch.no_grad():
            mask = (_pred(x) > 0).to(inter.dtype)
        return _dp(inter * mask)
    mlp.forward = fwd

ppl_hard = compute_ppl(model, wt2, device)
elapsed2 = time.time() - t0
print(f"  sparsity70_hard_mask PPL: {ppl_hard:.4f} ({elapsed2:.0f}s)", flush=True)

for idx, f in originals.items():
    model.model.layers[idx].mlp.forward = f

result = {"sparsity70_teal": round(ppl, 4), "sparsity70_hard_mask": round(ppl_hard, 4)}
with open("/root/distill_sparse_swiglu/results/plan_007_sparsity70.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"\nResults: {json.dumps(result, indent=2)}", flush=True)

"""Diagnostic script for training-free baseline PPL anomaly."""
import os, sys, torch
import torch.nn.functional as F
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.insert(0, "/root/distill_sparse_swiglu/src")

from transformers import AutoModelForCausalLM, AutoTokenizer
from data_utils import get_eval_dataset

MODEL_PATH = "/root/autodl-tmp/models/llama-3.1-8b"

print("=== PHASE 1: Load model ===")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"},
    attn_implementation="sdpa"
)
model.eval()
device = torch.device("cuda:0")

print("\n=== PHASE 2: Load eval data ===")
wt2 = get_eval_dataset("wikitext2", tokenizer, 2048, max_samples=20)
print(f"  {len(wt2)} sequences loaded")

print("\n=== PHASE 3: Dense PPL (direct, no patching) ===")
total_loss = 0.0
total_tokens = 0
with torch.no_grad():
    for i, ex in enumerate(wt2):
        ids = ex["input_ids"].unsqueeze(0).to(device)
        labels = ex["labels"].unsqueeze(0).to(device)
        logits = model(input_ids=ids).logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += labels.numel()
        if i < 3:
            avg_loss_i = loss.item() / labels.numel()
            print(f"  Seq {i}: avg_loss={avg_loss_i:.4f} ppl={torch.exp(torch.tensor(avg_loss_i)).item():.2f}")
            print(f"    logits min={logits.min():.3f} max={logits.max():.3f} mean={logits.float().mean():.3f}")
            print(f"    labels[:10]={labels[0,:10].tolist()}")

dense_ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
print(f"\n  Dense PPL ({len(wt2)} seq): {dense_ppl:.4f}")

print("\n=== PHASE 4: Calibration magnitudes ===")
from evaluate import _collect_swiglu_magnitudes, _global_topk_masks, _print_mask_stats

calib_ids = torch.stack([wt2[i]["input_ids"] for i in range(min(16, len(wt2)))]).to(device)
print(f"  Calibration: {calib_ids.shape}")
avg_mag = _collect_swiglu_magnitudes(model, calib_ids)

print("\n  Score statistics (selected layers):")
for li in [0, 1, 15, 16, 30, 31]:
    s = avg_mag[li].float()
    print(f"    Layer {li:2d}: min={s.min():.6f} max={s.max():.6f} mean={s.mean():.6f} "
          f"nan={s.isnan().sum().item()} inf={s.isinf().sum().item()} zeros={( s==0).sum().item()}")

print("\n=== PHASE 5: Vanilla TEAL masks (50% global) ===")
masks = _global_topk_masks(avg_mag, 0.5)
actual_sp = _print_mask_stats("Vanilla-TEAL-diag", masks)

# Check extreme layers
print("\n  Extreme layers (keep ratio):")
for li in sorted(masks):
    kr = masks[li].mean().item()
    if kr < 0.1 or kr > 0.9:
        print(f"    *** Layer {li}: keep_ratio={kr:.4f} (EXTREME!)")

print("\n=== PHASE 6: Sanity check - all-ones masks ===")
inter_dim = model.config.intermediate_size
n_layers = model.config.num_hidden_layers
all_ones = {i: torch.ones(inter_dim, dtype=torch.bfloat16) for i in range(n_layers)}
from evaluate import evaluate_perplexity
ppl_ones = evaluate_perplexity(model, wt2[:5], device, all_ones)
print(f"  PPL with all-ones masks (5 seq): {ppl_ones:.4f}")

# Dense reference on same 5 seq
total_loss2 = 0.0
total_tokens2 = 0
with torch.no_grad():
    for ex in wt2[:5]:
        ids = ex["input_ids"].unsqueeze(0).to(device)
        labels = ex["labels"].unsqueeze(0).to(device)
        logits = model(input_ids=ids).logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction="sum")
        total_loss2 += loss.item()
        total_tokens2 += labels.numel()
ppl_dense5 = torch.exp(torch.tensor(total_loss2 / total_tokens2)).item()
print(f"  Dense PPL (5 seq, direct):       {ppl_dense5:.4f}")
print(f"  Match: {'YES' if abs(ppl_ones - ppl_dense5) < 0.01 else 'NO - BUG!'}")

print("\n=== PHASE 7: Vanilla TEAL masked PPL ===")
ppl_teal = evaluate_perplexity(model, wt2[:5], device, masks)
print(f"  Vanilla TEAL 50% PPL (5 seq): {ppl_teal:.4f}")

# Per-token loss comparison (first sequence only)
print("\n=== PHASE 8: Per-token loss diagnosis (seq 0) ===")
ex0 = wt2[0]
ids0 = ex0["input_ids"].unsqueeze(0).to(device)
labels0 = ex0["labels"].unsqueeze(0).to(device)

# Dense
with torch.no_grad():
    logits_dense = model(input_ids=ids0).logits
loss_dense_per_token = F.cross_entropy(
    logits_dense.view(-1, logits_dense.size(-1)), labels0.view(-1), reduction="none"
)
print(f"  Dense: first 10 token losses = {loss_dense_per_token[:10].tolist()}")
print(f"  Dense: mean loss = {loss_dense_per_token.mean():.4f}")
print(f"  Dense: max loss = {loss_dense_per_token.max():.4f} at pos {loss_dense_per_token.argmax().item()}")

# Masked (apply masks manually)
saved_fwds = {}
for li in range(n_layers):
    saved_fwds[li] = model.model.layers[li].mlp.forward
    mlp = model.model.layers[li].mlp
    m = masks[li].to(device=device, dtype=mlp.gate_proj.weight.dtype)
    gp, up, dp, af = mlp.gate_proj, mlp.up_proj, mlp.down_proj, mlp.act_fn
    def make_fwd(gp_, up_, dp_, af_, m_):
        def fwd(x):
            inter = af_(gp_(x)) * up_(x)
            return dp_(inter * m_.unsqueeze(0).unsqueeze(0))
        return fwd
    model.model.layers[li].mlp.forward = make_fwd(gp, up, dp, af, m)

with torch.no_grad():
    logits_masked = model(input_ids=ids0).logits
loss_masked_per_token = F.cross_entropy(
    logits_masked.view(-1, logits_masked.size(-1)), labels0.view(-1), reduction="none"
)
print(f"\n  Masked: first 10 token losses = {loss_masked_per_token[:10].tolist()}")
print(f"  Masked: mean loss = {loss_masked_per_token.mean():.4f}")
print(f"  Masked: max loss = {loss_masked_per_token.max():.4f} at pos {loss_masked_per_token.argmax().item()}")
print(f"  Masked: num tokens with loss > 15 = {(loss_masked_per_token > 15).sum().item()}")

# Check logits range
print(f"\n  Dense logits: min={logits_dense.min():.3f} max={logits_dense.max():.3f}")
print(f"  Masked logits: min={logits_masked.min():.3f} max={logits_masked.max():.3f}")

# Check for NaN/Inf
print(f"  Dense logits NaN={logits_dense.isnan().sum().item()} Inf={logits_dense.isinf().sum().item()}")
print(f"  Masked logits NaN={logits_masked.isnan().sum().item()} Inf={logits_masked.isinf().sum().item()}")

# Restore
for li, fwd in saved_fwds.items():
    model.model.layers[li].mlp.forward = fwd

print("\n=== DONE ===")

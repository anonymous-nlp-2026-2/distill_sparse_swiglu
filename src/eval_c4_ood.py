"""C4 OOD PPL evaluation: dense vs KL-predictor-sparse vs vanilla-TEAL at 50% sparsity."""
import os, sys, time, json
import torch
import torch.nn.functional as F

os.environ["HF_HOME"] = "/root/autodl-tmp/.hf_cache"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["WANDB_MODE"] = "offline"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import get_eval_dataset
from predictor import PredictorWrapper
from evaluate import vanilla_teal_global_masks, _collect_swiglu_magnitudes, evaluate_perplexity, evaluate_true_dense

import importlib.util
from transformers import AutoModelForCausalLM, AutoTokenizer


def _best_attn():
    if importlib.util.find_spec("flash_attn"):
        return "flash_attention_2"
    return "sdpa"


@torch.no_grad()
def compute_ppl_predictor_dynamic(wrapper, dataset, device):
    """PPL with per-token dynamic predictor masks (actual inference mode)."""
    wrapper.sparse_mode = True
    wrapper.gumbel_mask.hard = True
    total_loss = 0.0
    total_tokens = 0
    for sample in dataset:
        ids = sample["input_ids"].unsqueeze(0).to(device)
        labels = sample["labels"].unsqueeze(0).to(device)
        out = wrapper.forward_sparse(ids)
        logits = out.logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += labels.numel()
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()


def main():
    device = torch.device("cuda:0")
    MODEL_PATH = "/root/autodl-tmp/models/llama-3.1-8b"
    CKPT_PATH = "/root/distill_sparse_swiglu/checkpoints/mvp_kl_norm_v2_s42/predictor_kl_normalized.pt"
    SPARSITY = 0.5
    MAX_SAMPLES = 200

    print("=" * 60)
    print("C4 OOD PPL Evaluation")
    print("=" * 60)

    print("\n[1/6] Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map={"": device},
        attn_implementation=_best_attn())
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print("[2/6] Loading C4 validation (local)...", flush=True)
    c4_data = get_eval_dataset("c4", tokenizer, 2048, max_samples=MAX_SAMPLES)
    print(f"  {len(c4_data)} sequences loaded", flush=True)

    # Calibration data from C4 (OOD calibration - fair comparison)
    calib_ids = torch.stack([s["input_ids"] for s in c4_data[:32]]).to(device)

    results = {}

    # --- Dense ---
    print("\n[3/6] Dense PPL on C4...", flush=True)
    t0 = time.time()
    ppl_dense = evaluate_true_dense(model, c4_data, device)
    results["dense"] = {"ppl": round(ppl_dense, 4), "time_s": round(time.time() - t0, 1)}
    print(f"  Dense PPL: {ppl_dense:.4f} ({results['dense']['time_s']}s)", flush=True)

    # --- KL Predictor Dynamic 50% ---
    print("\n[4/6] KL Predictor 50% (dynamic per-token mask) on C4...", flush=True)
    wrapper = PredictorWrapper(model, bottleneck_size=128)
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    wrapper.predictors.load_state_dict(ckpt["predictors"])
    wrapper.predictors.to(device=device, dtype=torch.bfloat16)
    wrapper.predictors.eval()
    wrapper.gumbel_mask.hard = True

    t0 = time.time()
    ppl_kl = compute_ppl_predictor_dynamic(wrapper, c4_data, device)
    results["kl_predictor_50"] = {"ppl": round(ppl_kl, 4), "time_s": round(time.time() - t0, 1)}
    print(f"  KL Predictor PPL: {ppl_kl:.4f} ({results['kl_predictor_50']['time_s']}s)", flush=True)

    # Restore original MLPs for TEAL eval
    for idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        gp, up, dp, af = mlp.gate_proj, mlp.up_proj, mlp.down_proj, mlp.act_fn
        def _make_orig(g, u, d, a):
            def fwd(x):
                return d(a(g(x)) * u(x))
            return fwd
        mlp.forward = _make_orig(gp, up, dp, af)

    # --- Vanilla TEAL 50% ---
    print("\n[5/6] Vanilla TEAL 50% (static magnitude mask) on C4...", flush=True)
    print("  Collecting calibration magnitudes...", flush=True)
    avg_mag = _collect_swiglu_magnitudes(model, calib_ids)
    teal_masks = vanilla_teal_global_masks(model, calib_ids, SPARSITY, avg_mag=avg_mag)

    t0 = time.time()
    ppl_teal = evaluate_perplexity(model, c4_data, device, teal_masks)
    results["teal_vanilla_50"] = {"ppl": round(ppl_teal, 4), "time_s": round(time.time() - t0, 1)}
    print(f"  TEAL Vanilla PPL: {ppl_teal:.4f} ({results['teal_vanilla_50']['time_s']}s)", flush=True)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("[6/6] RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Condition':<25} {'C4 PPL':>10} {'Δ vs Dense':>12}")
    print("-" * 50)
    print(f"{'Dense':<25} {ppl_dense:>10.4f} {'baseline':>12}")
    delta_kl = (ppl_kl - ppl_dense) / ppl_dense * 100
    print(f"{'KL Predictor 50%':<25} {ppl_kl:>10.4f} {f'+{delta_kl:.2f}%':>12}")
    delta_teal = (ppl_teal - ppl_dense) / ppl_dense * 100
    print(f"{'TEAL Vanilla 50%':<25} {ppl_teal:>10.4f} {f'+{delta_teal:.2f}%':>12}")
    print("-" * 50)
    kl_advantage = (ppl_teal - ppl_kl) / ppl_teal * 100
    print(f"\nKL advantage over TEAL: {kl_advantage:.2f}% lower PPL")
    print(f"KL generalizes to OOD: {'YES' if ppl_kl < ppl_teal else 'NO'}")

    # Save results
    os.makedirs("/root/distill_sparse_swiglu/results", exist_ok=True)
    out_path = "/root/distill_sparse_swiglu/results/c4_ood_ppl.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

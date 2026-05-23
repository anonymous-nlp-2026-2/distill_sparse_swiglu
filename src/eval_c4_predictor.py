"""Evaluate a single predictor checkpoint on C4 validation (OOD PPL)."""
import os, sys, time, argparse
import torch
import torch.nn.functional as F

os.environ["HF_HOME"] = "/root/autodl-tmp/.hf_cache"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["WANDB_MODE"] = "offline"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import get_eval_dataset
from predictor import PredictorWrapper

import importlib.util
from transformers import AutoModelForCausalLM, AutoTokenizer


def _best_attn():
    if importlib.util.find_spec("flash_attn"):
        return "flash_attention_2"
    return "sdpa"


@torch.no_grad()
def compute_ppl_predictor_dynamic(wrapper, dataset, device):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Predictor checkpoint path")
    parser.add_argument("--name", default="predictor", help="Display name")
    parser.add_argument("--max_samples", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    MODEL_PATH = "/root/autodl-tmp/models/llama-3.1-8b"

    print(f"=== {args.name} C4 OOD Eval ===", flush=True)

    print("[1/4] Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map={"": device},
        attn_implementation=_best_attn())
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print("[2/4] Loading C4 validation...", flush=True)
    c4_data = get_eval_dataset("c4", tokenizer, 2048, max_samples=args.max_samples)
    print(f"  {len(c4_data)} sequences loaded", flush=True)

    print("[3/4] Loading predictor...", flush=True)
    wrapper = PredictorWrapper(model, bottleneck_size=128)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    wrapper.predictors.load_state_dict(ckpt["predictors"])
    wrapper.predictors.to(device=device, dtype=torch.bfloat16)
    wrapper.predictors.eval()
    wrapper.gumbel_mask.hard = True

    print("[4/4] Computing PPL...", flush=True)
    t0 = time.time()
    ppl = compute_ppl_predictor_dynamic(wrapper, c4_data, device)
    elapsed = time.time() - t0
    print(f"\nRESULT: {args.name} C4 PPL = {ppl:.4f} (took {elapsed:.1f}s)")


if __name__ == "__main__":
    main()

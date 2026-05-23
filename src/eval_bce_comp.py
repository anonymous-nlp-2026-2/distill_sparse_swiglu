"""Evaluate BCE+Compensation predictor with compensation enabled."""
import os, sys, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import importlib.util

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import get_eval_dataset
from predictor import PredictorWrapper, CompensationNetwork

def _best_attn_impl():
    if importlib.util.find_spec("flash_attn"):
        return "flash_attention_2"
    return "sdpa"

@torch.no_grad()
def evaluate_perplexity(model, examples, device):
    total_loss = 0.0
    total_tokens = 0
    for ex in examples:
        ids = ex["input_ids"].unsqueeze(0).to(device)
        labels = ex["labels"].unsqueeze(0).to(device)
        logits = model(input_ids=ids).logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += labels.numel()
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()

def main():
    device = torch.device("cuda:0")
    model_path = "/root/autodl-tmp/models/llama-3.1-8b"
    ckpt_path = "/root/distill_sparse_swiglu/checkpoints/bce_comp_staged_s42/predictor_bce_comp.pt"

    print(f"Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map={"": device},
        attn_implementation=_best_attn_impl(),
    )
    model.eval()

    wrapper = PredictorWrapper(model, bottleneck_size=128)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    wrapper.predictors.load_state_dict(ckpt["predictors"])
    wrapper.predictors.to(device=device, dtype=torch.bfloat16)
    wrapper.predictors.eval()
    wrapper.gumbel_mask.hard = True

    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    comp_net = CompensationNetwork(num_layers, hidden_size, bottleneck_size=256)
    comp_net.load_state_dict(ckpt["comp_network"])
    comp_net.to(device=device, dtype=torch.bfloat16)
    comp_net.eval()

    wrapper.comp_network = comp_net
    wrapper.use_compensation = True

    print("Loading WikiText-2 ...")
    wt2 = get_eval_dataset("wikitext2", tokenizer, 2048)
    print(f"  {len(wt2)} sequences")

    print("\n[1/1] BCE+Comp Per-Token Hard Mask (WikiText-2) ...")
    ppl = evaluate_perplexity(model, wt2, device)
    print(f"  Perplexity (BCE+comp): {ppl:.2f}")

    # Also compute sparsity
    sample = wt2[0]["input_ids"].unsqueeze(0).to(device)
    wrapper.sparse_mode = True
    _ = model(input_ids=sample)
    masks = wrapper.get_masks()
    total_ones = sum(m.sum().item() for m in masks.values())
    total_elems = sum(m.numel() for m in masks.values())
    sparsity = 1.0 - total_ones / total_elems
    print(f"  Model-level sparsity: {sparsity:.3f}")

if __name__ == "__main__":
    main()

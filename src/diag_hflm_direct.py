"""Direct HFLM loglikelihood test: bypass dataset loading, test the exact code path."""
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

import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.api.instance import Instance

# MMLU-style questions (context, continuation) pairs
QUESTIONS = [
    {
        "context": "The following are multiple choice questions (with answers) about world knowledge.\n\nQuestion: What is the chemical symbol for gold?\nA. Ag\nB. Au\nC. Fe\nD. Cu\nAnswer:",
        "choices": [" A", " B", " C", " D"],
        "answer": 1,  # B (Au)
    },
    {
        "context": "The following are multiple choice questions (with answers) about world knowledge.\n\nQuestion: Which planet is known as the Red Planet?\nA. Venus\nB. Mars\nC. Jupiter\nD. Saturn\nAnswer:",
        "choices": [" A", " B", " C", " D"],
        "answer": 1,  # B (Mars)
    },
    {
        "context": "The following are multiple choice questions (with answers) about world knowledge.\n\nQuestion: What is the powerhouse of the cell?\nA. Nucleus\nB. Ribosome\nC. Mitochondria\nD. Golgi apparatus\nAnswer:",
        "choices": [" A", " B", " C", " D"],
        "answer": 2,  # C
    },
    {
        "context": "The following are multiple choice questions (with answers) about science.\n\nQuestion: What is the derivative of x^2?\nA. x\nB. 2x\nC. x^2\nD. 2x^2\nAnswer:",
        "choices": [" A", " B", " C", " D"],
        "answer": 1,  # B (2x)
    },
    {
        "context": "The following are multiple choice questions (with answers) about science.\n\nQuestion: Which gas makes up most of Earth's atmosphere?\nA. Oxygen\nB. Carbon dioxide\nC. Nitrogen\nD. Argon\nAnswer:",
        "choices": [" A", " B", " C", " D"],
        "answer": 2,  # C (Nitrogen)
    },
]

def test_hflm_loglikelihood(model, tokenizer, tag, questions):
    """Test HFLM loglikelihood on given questions."""
    lm_obj = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=4)
    
    # Build instances
    instances = []
    for qi, q in enumerate(questions):
        for ci, choice in enumerate(q["choices"]):
            inst = Instance(
                request_type="loglikelihood",
                doc={},
                arguments=(q["context"], choice),
                idx=qi * len(q["choices"]) + ci,
            )
            instances.append(inst)
    
    # Run loglikelihood
    results = lm_obj.loglikelihood(instances)
    
    # Parse results
    correct = 0
    for qi, q in enumerate(questions):
        logprobs = []
        for ci in range(len(q["choices"])):
            idx = qi * len(q["choices"]) + ci
            lp, is_greedy = results[idx]
            logprobs.append(lp)
        pred = logprobs.index(max(logprobs))
        is_correct = pred == q["answer"]
        correct += is_correct
        if not is_correct:
            print(f"  Q{qi} WRONG: pred={q['choices'][pred]}, true={q['choices'][q['answer']]}, "
                  f"logprobs={[f'{x:.3f}' for x in logprobs]}")
    
    acc = correct / len(questions)
    print(f"{tag}: {correct}/{len(questions)} = {acc*100:.1f}%")
    return acc

# === Test 1: Dense ===
print("\n=== Dense via HFLM.loglikelihood ===")
dense_acc = test_hflm_loglikelihood(model, tokenizer, "Dense", QUESTIONS)

# === Test 2: Patch and test sparse ===
print("\nPatching model with KL predictor...")
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

print("\n=== Sparse via HFLM.loglikelihood ===")
sparse_acc = test_hflm_loglikelihood(model, tokenizer, "Sparse", QUESTIONS)

# === Test 3: Check batch_size effect ===
print("\n=== Sparse via HFLM.loglikelihood (batch_size=1) ===")
sparse_acc_bs1 = test_hflm_loglikelihood(model, tokenizer, "Sparse-bs1", QUESTIONS)

print(f"\n=== Summary ===")
print(f"Dense: {dense_acc*100:.1f}%")
print(f"Sparse (bs=4): {sparse_acc*100:.1f}%")
print(f"Sparse (bs=1): {sparse_acc_bs1*100:.1f}%")

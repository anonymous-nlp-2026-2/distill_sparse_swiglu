"""Diagnostic: Manual MMLU-style evaluation on sparse model.
Compare dense vs sparse log-likelihoods on 20 hand-crafted MMLU-style questions.
"""
import os, sys, torch, torch.nn as nn
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/.hf_cache")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predictor import SparsityPredictor
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

device = torch.device("cuda:0")
MODEL_PATH = "/root/autodl-tmp/models/llama-3.1-8b"
CKPT_PATH = "/root/distill_sparse_swiglu/checkpoints/mvp_kl_norm_v2_s42/predictor_kl_normalized.pt"

QUESTIONS = [
    {"q": "What is the chemical symbol for gold?", "choices": ["Au", "Ag", "Fe", "Cu"], "answer": 0},
    {"q": "Which planet is known as the Red Planet?", "choices": ["Venus", "Mars", "Jupiter", "Saturn"], "answer": 1},
    {"q": "What is the powerhouse of the cell?", "choices": ["Nucleus", "Ribosome", "Mitochondria", "Golgi apparatus"], "answer": 2},
    {"q": "In which year did World War II end?", "choices": ["1943", "1944", "1945", "1946"], "answer": 2},
    {"q": "What is the derivative of x^2?", "choices": ["x", "2x", "x^2", "2x^2"], "answer": 1},
    {"q": "Which gas makes up most of Earth's atmosphere?", "choices": ["Oxygen", "Carbon dioxide", "Nitrogen", "Argon"], "answer": 2},
    {"q": "Who wrote 'Romeo and Juliet'?", "choices": ["Dickens", "Shakespeare", "Austen", "Twain"], "answer": 1},
    {"q": "What is the SI unit of force?", "choices": ["Watt", "Joule", "Newton", "Pascal"], "answer": 2},
    {"q": "Which organ produces insulin?", "choices": ["Liver", "Kidney", "Pancreas", "Stomach"], "answer": 2},
    {"q": "What is 15% of 200?", "choices": ["20", "25", "30", "35"], "answer": 2},
    {"q": "Which element has atomic number 1?", "choices": ["Helium", "Hydrogen", "Lithium", "Carbon"], "answer": 1},
    {"q": "The Pythagorean theorem relates to which shape?", "choices": ["Circle", "Square", "Right triangle", "Pentagon"], "answer": 2},
    {"q": "What is the currency of Japan?", "choices": ["Yuan", "Won", "Yen", "Baht"], "answer": 2},
    {"q": "DNA stands for?", "choices": ["Deoxyribose nucleic acid", "Deoxyribonucleic acid", "Dinitrogen acid", "Dynamic nucleic acid"], "answer": 1},
    {"q": "Which continent is the Sahara Desert on?", "choices": ["Asia", "Africa", "South America", "Australia"], "answer": 1},
    {"q": "What is the speed of light approximately?", "choices": ["300,000 km/s", "150,000 km/s", "600,000 km/s", "30,000 km/s"], "answer": 0},
    {"q": "Which vitamin is produced by sunlight?", "choices": ["Vitamin A", "Vitamin B", "Vitamin C", "Vitamin D"], "answer": 3},
    {"q": "What is the freezing point of water in Celsius?", "choices": ["-10", "0", "10", "32"], "answer": 1},
    {"q": "Who discovered penicillin?", "choices": ["Pasteur", "Fleming", "Curie", "Darwin"], "answer": 1},
    {"q": "What is the largest mammal?", "choices": ["Elephant", "Blue whale", "Giraffe", "Hippopotamus"], "answer": 1},
]

def compute_choice_logprobs(model, tokenizer, prompt, choices, device):
    """Compute log-probability for each choice, MMLU-style."""
    logprobs = []
    for choice in choices:
        full = prompt + " " + choice
        ids = tokenizer(full, return_tensors="pt").input_ids.to(device)
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
        n_prompt = prompt_ids.shape[1]
        
        with torch.no_grad():
            logits = model(input_ids=ids).logits
        
        lp = 0.0
        for i in range(n_prompt, ids.shape[1]):
            token_lp = F.log_softmax(logits[0, i-1], dim=-1)[ids[0, i]].item()
            lp += token_lp
        logprobs.append(lp)
    return logprobs

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, device_map={"": device}, attn_implementation="sdpa")
model.eval()

# === Dense baseline ===
print("\n=== Dense evaluation ===")
dense_correct = 0
for i, q in enumerate(QUESTIONS):
    prompt = f"Question: {q['q']}\nAnswer:"
    lps = compute_choice_logprobs(model, tokenizer, prompt, q['choices'], device)
    pred = lps.index(max(lps))
    correct = pred == q['answer']
    dense_correct += correct
    if not correct:
        print(f"  Q{i} WRONG: pred={q['choices'][pred]}, true={q['choices'][q['answer']]}, "
              f"lps={[f'{x:.2f}' for x in lps]}")
print(f"Dense accuracy: {dense_correct}/{len(QUESTIONS)} = {dense_correct/len(QUESTIONS)*100:.1f}%")

# === Patch with predictor (eval_benchmark.py style) ===
print("\nLoading predictor...")
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

# === Sparse evaluation ===
print("\n=== Sparse evaluation ===")
sparse_correct = 0
for i, q in enumerate(QUESTIONS):
    prompt = f"Question: {q['q']}\nAnswer:"
    lps = compute_choice_logprobs(model, tokenizer, prompt, q['choices'], device)
    pred = lps.index(max(lps))
    correct = pred == q['answer']
    sparse_correct += correct
    if not correct:
        print(f"  Q{i} WRONG: pred={q['choices'][pred]}, true={q['choices'][q['answer']]}, "
              f"lps={[f'{x:.2f}' for x in lps]}")
print(f"Sparse accuracy: {sparse_correct}/{len(QUESTIONS)} = {sparse_correct/len(QUESTIONS)*100:.1f}%")

print(f"\nSummary: Dense {dense_correct/len(QUESTIONS)*100:.1f}% vs Sparse {sparse_correct/len(QUESTIONS)*100:.1f}%")

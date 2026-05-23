#!/bin/bash
set -o pipefail
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
cd /root/distill_sparse_swiglu
export HF_HOME=/root/autodl-tmp/.hf_cache
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1

echo "===== D029 cuda:1 batch START $(date) ====="

# Condition 1: TEAL 50%
echo ""
echo "===== [1/5] bench_teal50 START $(date) ====="
python -u src/benchmark_eval.py --exp_id bench_teal50 --baseline_mode teal_vanilla --sparsity_target 0.5 --tasks wikitext2,arc_challenge,winogrande,mmlu,gsm8k --gpu 0
echo "===== [1/5] bench_teal50 EXIT_CODE=$? $(date) ====="

# Condition 2: TEAL 30%
echo ""
echo "===== [2/5] bench_teal30 START $(date) ====="
python -u src/benchmark_eval.py --exp_id bench_teal30 --baseline_mode teal_vanilla --sparsity_target 0.3 --tasks wikitext2,arc_challenge,winogrande,mmlu,gsm8k --gpu 0
echo "===== [2/5] bench_teal30 EXIT_CODE=$? $(date) ====="

# Condition 3: KL 70%
echo ""
echo "===== [3/5] bench_kl70_s42 START $(date) ====="
python -u src/benchmark_eval.py --exp_id bench_kl70_s42 --checkpoint checkpoints/kl_sparsity70_s42/predictor_kl_normalized.pt --sparsity_mode per_token --sparsity_target 0.7 --tasks wikitext2,arc_challenge,winogrande,mmlu,gsm8k --gpu 0
echo "===== [3/5] bench_kl70_s42 EXIT_CODE=$? $(date) ====="

# Condition 4: BCE iso-compute 50%
echo ""
echo "===== [4/5] bench_bce_iso50_s42 START $(date) ====="
python -u src/benchmark_eval.py --exp_id bench_bce_iso50_s42 --checkpoint checkpoints/bce_isocompute_v2_s42/predictor_bce.pt --sparsity_mode per_token --sparsity_target 0.5 --tasks wikitext2,arc_challenge,winogrande,mmlu,gsm8k --gpu 0
echo "===== [4/5] bench_bce_iso50_s42 EXIT_CODE=$? $(date) ====="

# Condition 5: Mistral KL 50%
echo ""
echo "===== [5/5] bench_mistral_kl50_s42 START $(date) ====="
python -u src/benchmark_eval.py --exp_id bench_mistral_kl50_s42 --model_name_or_path /root/autodl-tmp/models/mistral-7b --checkpoint checkpoints/mistral_kl_s42/predictor_kl_normalized.pt --sparsity_mode per_token --sparsity_target 0.5 --tasks wikitext2,arc_challenge,winogrande,mmlu,gsm8k --gpu 0
echo "===== [5/5] bench_mistral_kl50_s42 EXIT_CODE=$? $(date) ====="

echo ""
echo "===== D029 cuda:1 batch END $(date) ====="

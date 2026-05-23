#!/bin/bash
export HF_DATASETS_OFFLINE=1
export HF_HOME=/root/autodl-tmp/.hf_cache
export TRANSFORMERS_OFFLINE=1
cd /root/distill_sparse_swiglu

PYTHON=/root/miniconda3/bin/python3.12

echo "=== Mistral 50% KL per_token ===" >> logs/eval_remaining.log
$PYTHON src/benchmark_eval.py --model_name_or_path /root/autodl-tmp/models/mistral-7b/ \
  --checkpoint checkpoints/mistral_kl_s42/predictor_kl_normalized.pt \
  --sparsity_mode per_token --sparsity_target 0.5 --allocation uniform \
  --bottleneck_size 128 --exp_id mistral_kl50_pertoken --tasks wikitext2 --gpu 0 \
  >> logs/eval_remaining.log 2>&1

echo "=== Mistral 50% TEAL baseline ===" >> logs/eval_remaining.log
$PYTHON src/benchmark_eval.py --model_name_or_path /root/autodl-tmp/models/mistral-7b/ \
  --baseline_mode teal_vanilla --sparsity_target 0.5 --allocation uniform \
  --exp_id mistral_teal50_baseline --tasks wikitext2 --gpu 0 \
  >> logs/eval_remaining.log 2>&1

echo "=== Mistral 50% KL teal_global ===" >> logs/eval_remaining.log
$PYTHON src/benchmark_eval.py --model_name_or_path /root/autodl-tmp/models/mistral-7b/ \
  --checkpoint checkpoints/mistral_kl_s42/predictor_kl_normalized.pt \
  --sparsity_mode teal_global --sparsity_target 0.5 --allocation uniform \
  --bottleneck_size 128 --exp_id mistral_kl50_teal_global --tasks wikitext2 --gpu 0 \
  >> logs/eval_remaining.log 2>&1

echo "=== ALL 50% DONE ===" >> logs/eval_remaining.log

#!/bin/bash
set -e
source /root/distill_sparse_swiglu/setup_env.sh
cd /root/distill_sparse_swiglu

export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export HF_DATASETS_OFFLINE=1

echo "========== [$(date)] s42 =========="
python src/evaluate.py \
  --checkpoint checkpoints/mvp_bce_s42/predictor_bce.pt \
  --model_name_or_path /root/autodl-tmp/models/llama-3.1-8b \
  --sparsity_target 0.5 --gpu 0 --calibration_samples 32 --skip_c4

echo ""
echo "========== [$(date)] s123 =========="
python src/evaluate.py \
  --checkpoint checkpoints/mvp_bce_s123/predictor_bce.pt \
  --model_name_or_path /root/autodl-tmp/models/llama-3.1-8b \
  --sparsity_target 0.5 --gpu 0 --calibration_samples 32 --skip_c4

echo ""
echo "========== [$(date)] s456 =========="
python src/evaluate.py \
  --checkpoint checkpoints/mvp_bce_s456/predictor_bce.pt \
  --model_name_or_path /root/autodl-tmp/models/llama-3.1-8b \
  --sparsity_target 0.5 --gpu 0 --calibration_samples 32 --skip_c4

echo ""
echo "========== ALL DONE [$(date)] =========="

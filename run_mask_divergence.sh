#!/bin/bash
set -e
cd /root/distill_sparse_swiglu
export PATH="/root/miniconda3/bin:$PATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/root/autodl-tmp/.hf_cache
export CUDA_VISIBLE_DEVICES=1

echo "=========================================="
echo "Phase 1: Qwen-14B mask divergence analysis"
echo "=========================================="
python3 src/analyze_mask_divergence.py \
  --model_path /root/autodl-tmp/models/qwen2.5-14b \
  --predictor_path checkpoints/qwen14b_kl30_s42/predictor_kl_normalized.pt \
  --model_name qwen14b \
  --sparsity 0.3 \
  --calib_samples 32 \
  --device cuda:0 \
  --output_dir results

echo ""
echo "=========================================="
echo "Phase 2: LLaMA-8B mask divergence analysis"
echo "=========================================="
python3 src/analyze_mask_divergence.py \
  --model_path /root/autodl-tmp/models/llama-3.1-8b \
  --predictor_path checkpoints/kl_sparsity30_s42_v3/predictor_kl_normalized.pt \
  --model_name llama8b \
  --sparsity 0.3 \
  --calib_samples 32 \
  --device cuda:0 \
  --output_dir results

echo ""
echo "=========================================="
echo "Both analyses complete!"
echo "=========================================="

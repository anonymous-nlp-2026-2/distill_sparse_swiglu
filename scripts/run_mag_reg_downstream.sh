#!/bin/bash
source /root/distill_sparse_swiglu/setup_env.sh
cd /root/distill_sparse_swiglu

echo "=== Downstream eval for mag_reg checkpoints ==="
echo "Start: $(date)"

# alpha=0.01
echo "--- alpha=0.01 downstream ---"
CUDA_VISIBLE_DEVICES=1 python src/benchmark_eval.py \
  --checkpoint checkpoints/kl_mag_reg_a001/predictor_kl_mag_reg.pt \
  --tasks arc_challenge,mmlu \
  --sparsity_mode teal_global \
  --allocation uniform \
  --sparsity_target 0.5 \
  --exp_id mag_reg_a001_downstream \
  --gpu 0 \
  --batch_size 8

echo "alpha=0.01 downstream done at $(date)"

# alpha=0.1
echo "--- alpha=0.1 downstream ---"
CUDA_VISIBLE_DEVICES=1 python src/benchmark_eval.py \
  --checkpoint checkpoints/kl_mag_reg_a01/predictor_kl_mag_reg.pt \
  --tasks arc_challenge,mmlu \
  --sparsity_mode teal_global \
  --allocation uniform \
  --sparsity_target 0.5 \
  --exp_id mag_reg_a01_downstream \
  --gpu 0 \
  --batch_size 8

echo "alpha=0.1 downstream done at $(date)"
echo "=== All downstream evals complete ==="

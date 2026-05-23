#!/bin/bash
source /root/distill_sparse_swiglu/setup_env.sh
cd /root/distill_sparse_swiglu

echo "=== Starting mag_reg experiments (alpha=0.01, 0.1) ==="
echo "Start time: $(date)"

# alpha=0.01
echo "--- alpha=0.01 ---"
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python scripts/train_kl_mag_reg.py \
  --alpha 0.01 \
  --output_dir /root/distill_sparse_swiglu/checkpoints/kl_mag_reg_a001 \
  --gpu 0 \
  --num_steps 1000 \
  --batch_size 4 \
  --seq_len 2048 \
  --lr 1e-3 \
  --seed 42 \
  --sparsity_target 0.5 \
  --importance_batches 50

echo "alpha=0.01 done at $(date)"

# alpha=0.1
echo "--- alpha=0.1 ---"
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python scripts/train_kl_mag_reg.py \
  --alpha 0.1 \
  --output_dir /root/distill_sparse_swiglu/checkpoints/kl_mag_reg_a01 \
  --gpu 0 \
  --num_steps 1000 \
  --batch_size 4 \
  --seq_len 2048 \
  --lr 1e-3 \
  --seed 42 \
  --sparsity_target 0.5 \
  --importance_batches 50

echo "alpha=0.1 done at $(date)"
echo "=== All mag_reg experiments complete ==="

#!/bin/bash
# KL-only extra seeds (s0, s1) for 5-seed headline
# 复用 mvp_kl_norm_v2 超参，s0→gpu0, s1→gpu1 并行

source /root/distill_sparse_swiglu/setup_env.sh
cd /root/distill_sparse_swiglu

PYTHONUNBUFFERED=1 WANDB_MODE=disabled python src/train.py \
  --model_name_or_path /root/autodl-tmp/models/llama-3.1-8b \
  --loss_type kl_normalized \
  --num_steps 1000 \
  --batch_size 4 \
  --seq_len 2048 \
  --lr 1e-3 \
  --seed 0 \
  --sparsity_target 0.5 \
  --gpu 0 \
  --log_every 10 \
  --output_dir /root/distill_sparse_swiglu/checkpoints/kl_norm_v2_s0 \
  --wandb_run_name kl_norm_v2_s0 \
  --dataset wikitext-103 &

PYTHONUNBUFFERED=1 WANDB_MODE=disabled python src/train.py \
  --model_name_or_path /root/autodl-tmp/models/llama-3.1-8b \
  --loss_type kl_normalized \
  --num_steps 1000 \
  --batch_size 4 \
  --seq_len 2048 \
  --lr 1e-3 \
  --seed 1 \
  --sparsity_target 0.5 \
  --gpu 1 \
  --log_every 10 \
  --output_dir /root/distill_sparse_swiglu/checkpoints/kl_norm_v2_s1 \
  --wandb_run_name kl_norm_v2_s1 \
  --dataset wikitext-103 &

wait
echo "All KL extra seed runs completed."

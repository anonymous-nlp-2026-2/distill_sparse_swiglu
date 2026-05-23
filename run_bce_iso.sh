#!/bin/bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
cd /root/distill_sparse_swiglu
export HF_HOME=/root/autodl-tmp/.hf_cache
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

echo "=== BCE+Framework Iso-Ablation ==="
echo "Start time: $(date)"

echo ""
echo "=== Run 1/3: lr=1e-4 ==="
CUDA_VISIBLE_DEVICES=1 python -u src/train_bce_gumbel.py \
  --model_name_or_path /root/autodl-tmp/models/llama-3.1-8b \
  --num_steps 1000 --batch_size 4 --seq_len 2048 \
  --lr 1e-4 --seed 42 --sparsity_target 0.5 \
  --output_dir checkpoints/bce_framework_lr1e4 \
  --dataset wikitext-103 --gradient_checkpointing \
  --gpu 0 --log_every 50

echo ""
echo "=== Run 2/3: lr=1e-3 ==="
CUDA_VISIBLE_DEVICES=1 python -u src/train_bce_gumbel.py \
  --model_name_or_path /root/autodl-tmp/models/llama-3.1-8b \
  --num_steps 1000 --batch_size 4 --seq_len 2048 \
  --lr 1e-3 --seed 42 --sparsity_target 0.5 \
  --output_dir checkpoints/bce_framework_lr1e3 \
  --dataset wikitext-103 --gradient_checkpointing \
  --gpu 0 --log_every 50

echo ""
echo "=== Run 3/3: lr=3e-3 ==="
CUDA_VISIBLE_DEVICES=1 python -u src/train_bce_gumbel.py \
  --model_name_or_path /root/autodl-tmp/models/llama-3.1-8b \
  --num_steps 1000 --batch_size 4 --seq_len 2048 \
  --lr 3e-3 --seed 42 --sparsity_target 0.5 \
  --output_dir checkpoints/bce_framework_lr3e3 \
  --dataset wikitext-103 --gradient_checkpointing \
  --gpu 0 --log_every 50

echo ""
echo "=== All 3 BCE+Framework runs complete ==="
echo "End time: $(date)"

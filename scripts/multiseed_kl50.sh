#!/bin/bash
set -e
cd /root/distill_sparse_swiglu
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0

echo "=== KL 50% seed=123 Training ==="
/root/miniconda3/bin/python -u src/train.py \
  --loss_type kl_normalized --sparsity_target 0.5 --num_steps 1000 \
  --batch_size 4 --seq_len 2048 --lr 1e-3 --seed 123 \
  --gradient_checkpointing --dataset wikitext-103 --log_every 100 \
  --output_dir checkpoints/kl_50pct_s123 --wandb_run_name kl_50pct_s123

echo "=== KL 50% seed=123 Eval ==="
/root/miniconda3/bin/python -u src/benchmark_eval.py --exp_id kl_50pct_s123_global \
  --checkpoint checkpoints/kl_50pct_s123/predictor_kl_normalized.pt \
  --sparsity_mode teal_global --sparsity_target 0.5 \
  --tasks wikitext2,arc_challenge,winogrande,mmlu,gsm8k --gpu 0

echo "=== KL 50% seed=456 Training ==="
/root/miniconda3/bin/python -u src/train.py \
  --loss_type kl_normalized --sparsity_target 0.5 --num_steps 1000 \
  --batch_size 4 --seq_len 2048 --lr 1e-3 --seed 456 \
  --gradient_checkpointing --dataset wikitext-103 --log_every 100 \
  --output_dir checkpoints/kl_50pct_s456 --wandb_run_name kl_50pct_s456

echo "=== KL 50% seed=456 Eval ==="
/root/miniconda3/bin/python -u src/benchmark_eval.py --exp_id kl_50pct_s456_global \
  --checkpoint checkpoints/kl_50pct_s456/predictor_kl_normalized.pt \
  --sparsity_mode teal_global --sparsity_target 0.5 \
  --tasks wikitext2,arc_challenge,winogrande,mmlu,gsm8k --gpu 0

echo "=== ALL DONE ==="

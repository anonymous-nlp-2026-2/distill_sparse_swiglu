#!/bin/bash
set -e
cd /root/distill_sparse_swiglu
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Wait for s123 eval to finish (PID 25429)
echo "Waiting for s123 eval (PID 25429) to finish..."
while kill -0 25429 2>/dev/null; do
    sleep 10
done
echo "s123 eval done. Starting s456 training."

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

#!/bin/bash
set -e
cd /root/distill_sparse_swiglu
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Wait for s123 eval (PID from kl30_s123_eval tmux session) to finish
echo "[$(date)] Waiting for s123 eval to finish..."
while pgrep -f "benchmark_eval.*kl_30pct_s123" > /dev/null 2>&1; do
    sleep 10
done
echo "[$(date)] s123 eval finished. Starting s456 training."

echo "=== KL 30% seed=456 Training (lambda_max=5000) ==="
/root/miniconda3/bin/python -u src/train.py \
  --loss_type kl_normalized --sparsity_target 0.3 --num_steps 1000 \
  --batch_size 4 --seq_len 2048 --lr 1e-3 --seed 456 \
  --lambda_max 5000 \
  --gradient_checkpointing --dataset wikitext-103 --log_every 10 \
  --output_dir checkpoints/kl_30pct_s456 --wandb_run_name kl_30pct_s456

echo "=== KL 30% seed=456 Eval ==="
/root/miniconda3/bin/python -u src/benchmark_eval.py --exp_id kl_30pct_s456_global \
  --checkpoint checkpoints/kl_30pct_s456/predictor_kl_normalized.pt \
  --sparsity_mode teal_global --sparsity_target 0.3 \
  --tasks wikitext2,arc_challenge,winogrande,mmlu,gsm8k --gpu 0

echo "=== KL 30% s456 ALL DONE ==="

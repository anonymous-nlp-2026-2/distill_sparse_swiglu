#!/bin/bash
set -e
cd /root/distill_sparse_swiglu
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_CACHE=/root/autodl-tmp/.hf_cache/datasets

# Wait for s123 eval to finish (PID 29795)
echo "Waiting for kl30_s123 eval (PID 29795) to finish..."
while kill -0 29795 2>/dev/null; do
    sleep 30
    echo "  Still waiting... $(date)"
done
echo "kl30_s123 eval finished at $(date)"

# Collect s123 eval results
echo "=== s123 eval results ==="
grep -E "PPL|Results for|acc_norm|acc,|exact_match|word_perplexity|Task.*Version.*Filter|arc_challenge|winogrande|mmlu|gsm8k" ~/runs/distill_sparse_swiglu/kl30_s123_eval.log | grep -v "Overwriting\|cached\|Running\|num_fewshot" || true
echo "========================="

# s456 Training
echo "=== KL 30% seed=456 Training (lambda_max=5000) ==="
/root/miniconda3/bin/python -u src/train.py \
  --loss_type kl_normalized --sparsity_target 0.3 --num_steps 1000 \
  --batch_size 4 --seq_len 2048 --lr 1e-3 --seed 456 \
  --gradient_checkpointing --dataset wikitext-103 --log_every 100 \
  --lambda_max 5000 \
  --output_dir checkpoints/kl_30pct_s456 --wandb_run_name kl_30pct_s456

echo "=== KL 30% seed=456 Training DONE at $(date) ==="

# s456 Eval
echo "=== KL 30% seed=456 Eval ==="
/root/miniconda3/bin/python -u src/benchmark_eval.py --exp_id kl_30pct_s456_global \
  --checkpoint checkpoints/kl_30pct_s456/predictor_kl_normalized.pt \
  --sparsity_mode teal_global --sparsity_target 0.3 \
  --tasks wikitext2,arc_challenge,winogrande,mmlu,gsm8k --gpu 0

echo "=== ALL DONE at $(date) ==="

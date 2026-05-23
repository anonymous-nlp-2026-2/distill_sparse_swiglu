#!/bin/bash
set -e
cd /root/distill_sparse_swiglu
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled
PYTHON=/root/miniconda3/bin/python

echo "========== [$(date)] Run 1/4: KL 50% seed=123 TRAIN =========="
$PYTHON -u src/train.py \
  --loss_type kl_normalized \
  --sparsity_target 0.5 --num_steps 1000 --batch_size 4 --seq_len 2048 \
  --lr 1e-3 --seed 123 --gradient_checkpointing \
  --dataset wikitext-103 --log_every 100 \
  --output_dir checkpoints/kl_50pct_s123 \
  --wandb_run_name kl_50pct_s123

echo "========== [$(date)] Run 1/4: KL 50% seed=123 EVAL =========="
$PYTHON -u src/benchmark_eval.py \
  --exp_id kl_50pct_s123 \
  --checkpoint checkpoints/kl_50pct_s123/predictor_kl_normalized.pt \
  --sparsity_mode teal_global --sparsity_target 0.5 \
  --tasks wikitext2,arc_challenge,winogrande,mmlu,gsm8k --gpu 0

echo "========== [$(date)] Run 2/4: KL 50% seed=456 TRAIN =========="
$PYTHON -u src/train.py \
  --loss_type kl_normalized \
  --sparsity_target 0.5 --num_steps 1000 --batch_size 4 --seq_len 2048 \
  --lr 1e-3 --seed 456 --gradient_checkpointing \
  --dataset wikitext-103 --log_every 100 \
  --output_dir checkpoints/kl_50pct_s456 \
  --wandb_run_name kl_50pct_s456

echo "========== [$(date)] Run 2/4: KL 50% seed=456 EVAL =========="
$PYTHON -u src/benchmark_eval.py \
  --exp_id kl_50pct_s456 \
  --checkpoint checkpoints/kl_50pct_s456/predictor_kl_normalized.pt \
  --sparsity_mode teal_global --sparsity_target 0.5 \
  --tasks wikitext2,arc_challenge,winogrande,mmlu,gsm8k --gpu 0

echo "========== [$(date)] Run 3/4: KL 30% seed=123 TRAIN =========="
$PYTHON -u src/train.py \
  --loss_type kl_normalized \
  --sparsity_target 0.3 --num_steps 1000 --batch_size 4 --seq_len 2048 \
  --lr 1e-3 --seed 123 --gradient_checkpointing \
  --dataset wikitext-103 --log_every 100 \
  --output_dir checkpoints/kl_30pct_s123 \
  --wandb_run_name kl_30pct_s123

echo "========== [$(date)] Run 3/4: KL 30% seed=123 EVAL =========="
$PYTHON -u src/benchmark_eval.py \
  --exp_id kl_30pct_s123 \
  --checkpoint checkpoints/kl_30pct_s123/predictor_kl_normalized.pt \
  --sparsity_mode teal_global --sparsity_target 0.3 \
  --tasks wikitext2,arc_challenge,winogrande,mmlu,gsm8k --gpu 0

echo "========== [$(date)] Run 4/4: KL 30% seed=456 TRAIN =========="
$PYTHON -u src/train.py \
  --loss_type kl_normalized \
  --sparsity_target 0.3 --num_steps 1000 --batch_size 4 --seq_len 2048 \
  --lr 1e-3 --seed 456 --gradient_checkpointing \
  --dataset wikitext-103 --log_every 100 \
  --output_dir checkpoints/kl_30pct_s456 \
  --wandb_run_name kl_30pct_s456

echo "========== [$(date)] Run 4/4: KL 30% seed=456 EVAL =========="
$PYTHON -u src/benchmark_eval.py \
  --exp_id kl_30pct_s456 \
  --checkpoint checkpoints/kl_30pct_s456/predictor_kl_normalized.pt \
  --sparsity_mode teal_global --sparsity_target 0.3 \
  --tasks wikitext2,arc_challenge,winogrande,mmlu,gsm8k --gpu 0

echo "========== [$(date)] ALL 4 RUNS COMPLETE =========="
echo "Results:"
for d in kl_50pct_s123 kl_50pct_s456 kl_30pct_s123 kl_30pct_s456; do
  echo "--- $d ---"
  cat results/benchmark/$d/results.json 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin)
m=d['metrics']
lm=m.get('lm_eval',{})
print(f\"PPL={m['wikitext2_ppl']:.4f}  ARC-c={lm.get('arc_challenge',{}).get('acc_norm','N/A')}  WG={lm.get('winogrande',{}).get('acc','N/A')}  MMLU={lm.get('mmlu',{}).get('acc','N/A')}  GSM8K={lm.get('gsm8k',{}).get('exact_match','N/A')}\")" || echo "  (no results yet)"
done

#!/bin/bash
cd /root/distill_sparse_swiglu
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
export HF_DATASETS_OFFLINE=1
export HF_HOME=/root/autodl-tmp/.hf_cache
export TRANSFORMERS_OFFLINE=1

echo "[$(date)] Waiting for tau_fixed01 training to complete..."
while ! grep -q "Step 1000/1000" logs/tau_fixed01.log 2>/dev/null; do
    sleep 30
done
echo "[$(date)] tau_fixed01 training complete. Starting eval on cuda:0..."
CUDA_VISIBLE_DEVICES=0 python -u src/benchmark_eval.py \
    --checkpoint checkpoints/tau_ablation_fixed01/predictor_kl_normalized.pt \
    --sparsity_mode teal_global --allocation uniform \
    --tasks wikitext2 --batch_size 4 --bottleneck_size 128 --gpu 0 \
    --exp_id tau_fixed01_teal
echo "[$(date)] tau_fixed01 eval complete."

echo "[$(date)] Waiting for steps_curve training to complete..."
while ! grep -q "Step 1000/1000" logs/steps_curve_s42.log 2>/dev/null; do
    sleep 30
done
echo "[$(date)] steps_curve training complete. Starting evals..."

for STEP in 200 400 600 800 1000; do
    CKPT="checkpoints/steps_curve_s42/predictor_kl_normalized_step_${STEP}.pt"
    if [ -f "$CKPT" ]; then
        echo "[$(date)] Evaluating step ${STEP}..."
        CUDA_VISIBLE_DEVICES=0 python -u src/benchmark_eval.py \
            --checkpoint "$CKPT" \
            --sparsity_mode teal_global --allocation uniform \
            --tasks wikitext2 --batch_size 4 --bottleneck_size 128 --gpu 0 \
            --exp_id "steps_curve_step${STEP}"
    else
        echo "[$(date)] WARNING: $CKPT not found!"
    fi
done
echo "[$(date)] All evals complete!"

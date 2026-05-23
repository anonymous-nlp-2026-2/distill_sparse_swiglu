#!/bin/bash
set -e

source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/distill_sparse_swiglu

export WANDB_MODE=offline
export HF_HOME=/root/autodl-tmp/.hf_cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=========================================="
echo "d_b Sensitivity Sweep: 64, 256, 512"
echo "=========================================="

for DB in 64 256 512; do
    echo ""
    echo "===== Training d_b=${DB} ====="
    echo "Start: $(date)"
    
    python -u src/train.py \
        --loss_type kl_normalized \
        --bottleneck_dim ${DB} \
        --num_steps 1000 \
        --sparsity_target 0.5 \
        --lr 1e-3 \
        --seed 42 \
        --gradient_checkpointing \
        --gpu 0 \
        --output_dir checkpoints/db_sweep_${DB} \
        --wandb_run_name db_sweep_${DB}
    
    echo "Training d_b=${DB} done: $(date)"
    echo ""
    echo "===== Eval d_b=${DB} ====="
    
    python -u src/benchmark_eval.py \
        --checkpoint checkpoints/db_sweep_${DB}/predictor_kl_normalized.pt \
        --sparsity_mode per_token \
        --allocation uniform \
        --tasks wikitext2 \
        --batch_size 4 \
        --gpu 0 \
        --bottleneck_size ${DB} \
        --exp_id db_sweep_${DB}
    
    echo "Eval d_b=${DB} done: $(date)"
done

echo ""
echo "===== Eval baseline d_b=128 (confirm) ====="
python -u src/benchmark_eval.py \
    --checkpoint checkpoints/mvp_kl_norm_v2_s42/predictor_kl_normalized.pt \
    --sparsity_mode per_token \
    --allocation uniform \
    --tasks wikitext2 \
    --batch_size 4 \
    --gpu 0 \
    --bottleneck_size 128 \
    --exp_id db_baseline_128

echo ""
echo "=========================================="
echo "ALL DONE: $(date)"
echo "=========================================="

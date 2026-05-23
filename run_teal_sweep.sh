#!/bin/bash
set -e
cd /root/distill_sparse_swiglu
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base

export HF_HOME=/root/autodl-tmp/.hf_cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false

echo "========== START $(date) =========="

for SP in 0.3 0.5 0.7; do
    echo ""
    echo "============================================"
    echo "=== Sparsity target: ${SP} ==="
    echo "============================================"
    python -u src/evaluate.py \
        --baseline_mode teal_vanilla,wina \
        --sparsity_target ${SP} \
        --gpu 0 \
        --seq_len 2048 \
        --model_name_or_path /root/autodl-tmp/models/llama-3.1-8b
done

echo ""
echo "========== DONE $(date) =========="

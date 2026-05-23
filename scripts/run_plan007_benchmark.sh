#!/bin/bash
# Plan 007: Full benchmark evaluation for all sparsity conditions.
# Output: /root/distill_sparse_swiglu/results/plan_007/

set -euo pipefail

cd /root/distill_sparse_swiglu
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

export CUDA_VISIBLE_DEVICES=0,1
export HF_HOME=/root/autodl-tmp/.hf_cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

SCRIPT="python src/eval_benchmark.py"
GPU=${1:-1}  # default GPU 1 (GPU 0 may run training)
OUT="results/plan_007"

echo "=============================================="
echo " Plan 007: Benchmark Evaluation (GPU $GPU)"
echo " $(date)"
echo "=============================================="

# Step 1: Quick validation — dense baseline on WikiText-2 PPL
echo ""
echo "[Step 1] Validation: dense + wikitext2 PPL"
$SCRIPT --condition dense --tasks wikitext2 --gpu $GPU --max_ppl_samples 20 \
    --output_dir ${OUT}/validation

# Step 2: All conditions — PPL only
echo ""
echo "[Step 2] PPL evaluation (all conditions)"
$SCRIPT --condition all --tasks ppl --gpu $GPU --output_dir ${OUT}

# Step 3: lm-eval tasks (requires pre-cached datasets)
# Datasets needed: allenai/ai2_arc, winogrande, cais/mmlu, gsm8k
# If datasets aren't cached, tasks will report errors individually.
echo ""
echo "[Step 3] lm-eval tasks (all conditions)"
$SCRIPT --condition all --tasks arc_challenge,winogrande,mmlu,gsm8k \
    --gpu $GPU --batch_size 4 --skip_ppl --output_dir ${OUT}

echo ""
echo "=============================================="
echo " Done: $(date)"
echo " Results: ${OUT}/"
echo "=============================================="

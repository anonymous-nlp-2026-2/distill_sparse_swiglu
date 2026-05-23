#!/bin/bash
# plan_001 remaining: BCE+comp (5 seeds) + KL+comp (3 seeds)
# Sequential training: freeze predictor → train CompensationNetwork
#
# Dependencies:
#   s42/s123/s456: predictor checkpoints already exist
#   s0/s1: must train BCE predictor first (Phase A), then compensation (Phase B)
#
# Usage:
#   bash scripts/run_plan001_remaining.sh [phase]
#   phase: bce_comp | kl_comp | bce_pred_new | all
#
# GPU allocation: cuda:0 and cuda:1 only (project constraint C001)

set -euo pipefail
source /root/distill_sparse_swiglu/setup_env.sh
cd /root/distill_sparse_swiglu

PHASE="${1:-all}"
COMMON_ARGS="--batch_size 4 --seq_len 2048 --num_steps 1000 --sparsity_target 0.5 --warmup_steps 100 --dataset wikitext-103"

# ============================================================
# Phase A: Train BCE predictors for new seeds s0, s1
# (only needed if checkpoints don't exist yet)
# ============================================================
run_bce_predictor() {
    local SEED=$1
    local GPU=$2
    local OUT="checkpoints/bce_comp_s${SEED}"
    if [ -f "${OUT}/predictor_bce.pt" ]; then
        echo "[SKIP] BCE predictor s${SEED} already exists: ${OUT}/predictor_bce.pt"
        return
    fi
    echo "[Phase A] Training BCE predictor seed=${SEED} gpu=${GPU} -> ${OUT}"
    python src/train.py \
        --loss_type bce \
        --seed ${SEED} \
        --gpu ${GPU} \
        --lr 1e-3 \
        --output_dir ${OUT} \
        --wandb_run_name "bce_pred_s${SEED}" \
        ${COMMON_ARGS}
}

# ============================================================
# Phase B: Train CompensationNetwork (freeze predictor)
# ============================================================
run_compensation() {
    local LOSS_TYPE=$1   # bce or kl
    local SEED=$2
    local GPU=$3
    local PRED_CKPT=$4
    local OUT_DIR=$5

    if [ -f "${OUT_DIR}/compensation.pt" ]; then
        echo "[SKIP] Compensation already exists: ${OUT_DIR}/compensation.pt"
        return
    fi
    echo "[Phase B] Training compensation ${LOSS_TYPE}+comp seed=${SEED} gpu=${GPU}"
    echo "  predictor: ${PRED_CKPT}"
    echo "  output:    ${OUT_DIR}"
    python src/train_compensation.py \
        --predictor_checkpoint ${PRED_CKPT} \
        --seed ${SEED} \
        --gpu ${GPU} \
        --lr 1e-4 \
        --output_dir ${OUT_DIR} \
        --wandb_run_name "${LOSS_TYPE}_comp_s${SEED}" \
        ${COMMON_ARGS}
}

# ============================================================
# BCE+comp: 5 seeds
# ============================================================
run_bce_comp() {
    # Existing predictor checkpoints (s42, s123, s456)
    run_compensation bce 42  0 "checkpoints/mvp_bce_s42/predictor_bce.pt"   "checkpoints/mvp_bce_s42"
    run_compensation bce 123 0 "checkpoints/mvp_bce_s123/predictor_bce.pt"  "checkpoints/mvp_bce_s123"
    run_compensation bce 456 0 "checkpoints/mvp_bce_s456/predictor_bce.pt"  "checkpoints/mvp_bce_s456"

    # New seeds — require Phase A first
    run_bce_predictor 0 0
    run_compensation bce 0 0 "checkpoints/bce_comp_s0/predictor_bce.pt" "checkpoints/bce_comp_s0"

    run_bce_predictor 1 0
    run_compensation bce 1 0 "checkpoints/bce_comp_s1/predictor_bce.pt" "checkpoints/bce_comp_s1"
}

# ============================================================
# KL+comp: 3 seeds
# ============================================================
run_kl_comp() {
    run_compensation kl 42  1 "checkpoints/mvp_kl_norm_v2_s42/predictor_kl_normalized.pt"  "checkpoints/mvp_kl_norm_v2_s42"
    run_compensation kl 123 1 "checkpoints/mvp_kl_norm_v2_s123/predictor_kl_normalized.pt" "checkpoints/mvp_kl_norm_v2_s123"
    run_compensation kl 456 1 "checkpoints/mvp_kl_norm_v2_s456/predictor_kl_normalized.pt" "checkpoints/mvp_kl_norm_v2_s456"
}

# ============================================================
# Dispatch
# ============================================================
case "${PHASE}" in
    bce_pred_new)
        run_bce_predictor 0 0
        run_bce_predictor 1 0
        ;;
    bce_comp)
        run_bce_comp
        ;;
    kl_comp)
        run_kl_comp
        ;;
    all)
        echo "=== BCE+comp (5 seeds) ==="
        run_bce_comp
        echo ""
        echo "=== KL+comp (3 seeds) ==="
        run_kl_comp
        echo ""
        echo "=== All plan_001 remaining runs complete ==="
        ;;
    *)
        echo "Usage: $0 {bce_pred_new|bce_comp|kl_comp|all}"
        exit 1
        ;;
esac

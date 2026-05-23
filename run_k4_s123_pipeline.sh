#!/bin/bash
set -e
cd /root/distill_sparse_swiglu
export HF_HOME=/root/autodl-tmp/.hf_cache
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export WANDB_DISABLED=true

CKPT_DIR="/root/distill_sparse_swiglu/checkpoints/perlayer_kl_k4_s123"
CKPT_FILE="${CKPT_DIR}/predictor_perlayer_kl_k4.pt"

echo "=== Phase 1: Training K=4 s123 ==="
python -u src/train_perlayer_kl_k4.py \
  --num_masked_layers 4 \
  --seed 123 \
  --num_steps 2700 \
  --save_every 500 \
  --gradient_checkpointing \
  --output_dir ${CKPT_DIR} \
  --gpu 0
if [ $? -ne 0 ]; then echo "PHASE 1 FAILED"; exit 1; fi

if [ ! -f "${CKPT_FILE}" ]; then
    echo "PHASE 1 FAILED: checkpoint not found at ${CKPT_FILE}"
    ls -la ${CKPT_DIR}/ 2>/dev/null || echo "Directory does not exist"
    exit 1
fi
echo "Checkpoint verified: $(ls -la ${CKPT_FILE})"

echo "=== Phase 2: PPL Eval ==="
python -u src/benchmark_eval.py \
  --exp_id perlayer_kl_k4_s123_ppl \
  --model_name_or_path /root/autodl-tmp/models/llama-3.1-8b \
  --checkpoint ${CKPT_FILE} \
  --sparsity_mode teal_global --sparsity_target 0.3 --allocation uniform \
  --tasks wikitext2 --seq_len 2048 --gpu 0
if [ $? -ne 0 ]; then echo "PHASE 2 FAILED"; exit 1; fi

echo "=== Phase 3: Downstream 7-task Eval ==="
python -u src/benchmark_eval.py \
  --exp_id perlayer_kl_k4_s123_downstream \
  --model_name_or_path /root/autodl-tmp/models/llama-3.1-8b \
  --checkpoint ${CKPT_FILE} \
  --sparsity_mode teal_global --sparsity_target 0.3 --allocation uniform \
  --tasks arc_challenge,winogrande,mmlu,gsm8k,hellaswag,piqa,boolq --seq_len 2048 --gpu 0
if [ $? -ne 0 ]; then echo "PHASE 3 FAILED"; exit 1; fi

echo "=== Phase 4: Per-layer Sparsity Analysis ==="
python src/analyze_perlayer_sparsity.py \
  --checkpoint ${CKPT_FILE} \
  --calib_samples 32
if [ $? -ne 0 ]; then echo "PHASE 4 FAILED"; exit 1; fi

echo "=== ALL DONE ==="

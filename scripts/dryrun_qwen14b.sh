#!/bin/bash
# Dry-run: Qwen-2.5-14B KL training for 5 steps
# Uses random data (no network needed), tests model loading + predictor dimensions + loss

export CUDA_VISIBLE_DEVICES=0

cd /root/distill_sparse_swiglu

/root/miniconda3/bin/python src/train.py \
    --model_name_or_path /root/autodl-tmp/models/qwen2.5-14b/ \
    --loss_type kl \
    --num_steps 5 \
    --batch_size 1 \
    --seq_len 512 \
    --lr 1e-3 \
    --seed 42 \
    --sparsity_target 0.5 \
    --gpu 0 \
    --output_dir /root/distill_sparse_swiglu/checkpoints/dryrun_qwen14b \
    --wandb_project distill_sparse_swiglu \
    --wandb_run_name dryrun_qwen14b \
    --use_random_data \
    --gradient_checkpointing \
    --log_every 1

echo "Exit code: $?"

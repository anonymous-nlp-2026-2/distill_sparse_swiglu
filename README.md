# Distill Sparse SwiGLU

Code for "KL Distillation for SwiGLU Sparsity Prediction" (EMNLP 2026 submission).

## Setup
```bash
pip install -r requirements.txt
```

## Training
```bash
python src/train_perlayer_kl_k4.py --num_masked_layers 4 --seed 42 --num_steps 1000 --save_every 500 --gradient_checkpointing --output_dir checkpoints/example --gpu 0
```

## Evaluation
```bash
python src/benchmark_eval.py --model_name_or_path <model_path> --checkpoint <checkpoint_path> --sparsity_mode teal_global --sparsity_target 0.3 --allocation uniform --tasks wikitext2,arc_challenge,winogrande,mmlu,gsm8k,hellaswag,piqa --seq_len 2048 --gpu 0
```

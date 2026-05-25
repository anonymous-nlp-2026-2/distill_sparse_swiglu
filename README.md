# Distill Sparse SwiGLU

Code for the EMNLP 2026 submission "Output-Distribution Distillation for Contextual Sparsity Prediction in SwiGLU LLMs" (anonymous submission).

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.10, PyTorch 2.4 / CUDA 12.4. A single GPU with ~24 GB memory is sufficient for the default Llama-3-8B configuration; multi-GPU is not required.

## Repository layout

```
src/                 Core training and evaluation code
  train_perlayer_kl.py        Per-layer KL distillation training (main training entry)
  train_bce_kl_targets.py     BCE supervision against KL-oracle masks (ablation)
  generate_kl_oracle_masks.py Pre-compute KL-oracle masks used by BCE baseline
  eval_constrained_sparsity.py Downstream / PPL evaluation with sparsity constraints
  compute_jaccard.py          Cross-seed / cross-method mask Jaccard similarity
  benchmark_flashinfer.py     End-to-end latency benchmark via FlashInfer kernels
scripts/             Auxiliary analysis scripts (gradient SNR, oracle F1, etc.)
tests/               Unit / regression tests
paper/               LaTeX sources (main.tex is the entry; PDF is included)
figures/             Generated figures referenced by the paper
```

## Training

Per-layer KL distillation (default configuration matching the paper):

```bash
python src/train_perlayer_kl.py \
    --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --sparsity_target 0.30 \
    --bottleneck_dim 128 \
    --num_steps 10000 \
    --batch_size 4 --seq_len 2048 \
    --tau_start 1.0 --tau_end 0.1 \
    --lambda_max 5000 \
    --seed 42 \
    --gradient_checkpointing \
    --output_dir checkpoints/perlayer_kl_30pct_s42 \
    --gpu 0
```

BCE baseline (requires precomputed KL-oracle masks):

```bash
python src/generate_kl_oracle_masks.py --sparsity 0.30 --output_dir artifacts/kl_oracle_masks
python src/train_bce_kl_targets.py --mask_dir artifacts/kl_oracle_masks --seed 42 ...
```

## Evaluation

Downstream tasks + PPL under a target sparsity:

```bash
python src/eval_constrained_sparsity.py \
    --checkpoint checkpoints/perlayer_kl_30pct_s42/final.pt \
    --sparsity_target 0.30 \
    --constraint_type protect_late \
    --tasks wikitext2,arc_challenge,winogrande,mmlu,gsm8k,hellaswag,piqa \
    --seq_len 2048 \
    --gpu 0
```

## Reproducing paper figures

Scripts under `figures/paper/gen_*.py` regenerate the plots from cached experiment outputs. The corresponding raw outputs are not redistributed with this anonymous package; see the paper appendix for the full experimental protocol.

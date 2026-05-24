import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 1.8,
})

COLOR_OURS = '#0072B2'
COLOR_TEAL = '#E69F00'
COLOR_DENSE = '#999999'

# Data: all s42 global allocation (exp_ids below)
# bench_kl30_global=8.9768, kl_sparsity40_s42_global_eval_v2=10.14,
# bench_kl50_global=11.7933, bench_kl70_global=17.91
kl_sparsity = [28.5, 40.0, 50.0, 69.9]
kl_ppl = [8.98, 10.14, 11.79, 17.91]

# bench_teal30=10.97, bench_teal50=23.31, bench_teal70=100.11
teal_sparsity = [28.5, 50.0, 69.9]
teal_ppl = [10.97, 23.31, 100.11]

dense_ppl = 6.12

# KL advantage percentage at matched sparsity levels (28.5%, 50%, 69.9%)
advantages = ['18%', '49%', '82%']
matched_sparsity = [28.5, 50.0, 69.9]
matched_kl_ppl = [8.98, 11.79, 17.91]

fig, ax = plt.subplots(figsize=(5.5, 3.8))

# Dense baseline
ax.axhline(y=dense_ppl, color=COLOR_DENSE, linestyle='--', linewidth=1.2, zorder=1)
ax.text(72, dense_ppl * 1.06, f'Dense ({dense_ppl})', color=COLOR_DENSE,
        fontsize=9, ha='right', va='bottom')

# KL Predictor line
ax.plot(kl_sparsity, kl_ppl, 'o-', color=COLOR_OURS, markersize=7,
        label='KL Predictor (Ours)', zorder=3)

# TEAL baseline line
ax.plot(teal_sparsity, teal_ppl, '^--', color=COLOR_TEAL, markersize=8,
        markeredgewidth=1.5, linewidth=1.2, label='Vanilla TEAL', zorder=3)

# Annotate KL PPL values
for x, y in zip(kl_sparsity, kl_ppl):
    ax.annotate(f'{y:.2f}', (x, y), textcoords='offset points',
                xytext=(0, -10), fontsize=9, color=COLOR_OURS,
                ha='center', va='top')

# Annotate TEAL PPL values
for x, y in zip(teal_sparsity, teal_ppl):
    ax.annotate(f'{y:.1f}', (x, y), textcoords='offset points',
                xytext=(0, 8), fontsize=9, color=COLOR_TEAL,
                ha='center', va='bottom')

# Annotate KL advantage at matched sparsity levels (where both KL and TEAL exist)
for i, (x, adv) in enumerate(zip(matched_sparsity, advantages)):
    mid_y = np.sqrt(matched_kl_ppl[i] * teal_ppl[i])  # geometric mean for log scale
    ax.annotate(f'{adv}\nbetter', (x + 2, mid_y), fontsize=8, color='#555555',
                ha='left', va='center', style='italic')

# Log scale Y axis (handles 6-100 range cleanly)
ax.set_yscale('log')
ax.set_yticks([6, 8, 10, 15, 20, 30, 50, 100])
ax.set_yticklabels(['6', '8', '10', '15', '20', '30', '50', '100'])
ax.set_ylim(5, 130)

ax.set_xlabel('Achieved Model-Level Sparsity (%)')
ax.set_ylabel('WikiText-2 Perplexity (log scale)')
ax.set_xlim(22, 76)
ax.set_xticks([30, 40, 50, 60, 70])

ax.grid(True, alpha=0.3, zorder=0, which='major')
ax.legend(loc='upper left', framealpha=0.9, edgecolor='none')

plt.tight_layout()

outdir = '/home/ubuntu/.agent-ml-research-idea_gen_0509_14/projects/distill_sparse_swiglu/figures/paper'
plt.savefig(f'{outdir}/fig2_sparsity_sweep.pdf')
plt.savefig(f'{outdir}/fig2_sparsity_sweep.png')
plt.close()
print('Done')

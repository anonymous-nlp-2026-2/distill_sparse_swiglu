import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 1.8,
})

COLOR_KL = '#0072B2'
COLOR_TEAL = '#E69F00'

# K-curve data: all 3-seed mean ± std (Director-verified 2026-05-23)
K = [1, 2, 4, 8, 16, 32]
ppl_mean = [9.05, 10.75, 8.89, 9.00, 8.67, 8.81]
ppl_std =  [0.02, 0.15,  0.005, 0.06, 0.02, 0.02]
is_3seed = [True, True, True, True, True, True]

TEAL_PPL = 10.82

fig, ax = plt.subplots(figsize=(3.5, 1.9))

x_pos = np.arange(len(K))

for i, (k, m, s) in enumerate(zip(K, ppl_mean, ppl_std)):
    ax.errorbar(x_pos[i], m, yerr=s if s > 0 else None,
                fmt='o', color=COLOR_KL, markersize=5,
                capsize=2.5, capthick=1.0, elinewidth=1.0,
                markeredgecolor='white', markeredgewidth=0.5,
                zorder=5)

ax.plot(x_pos, ppl_mean, '-', color=COLOR_KL, alpha=0.4, zorder=3)

ax.axhline(y=TEAL_PPL, color=COLOR_TEAL, linestyle='--', linewidth=1.2,
           alpha=0.8, zorder=2)
ax.text(x_pos[0] + 0.15, TEAL_PPL + 0.05, 'TEAL', color=COLOR_TEAL,
        fontsize=7, va='bottom', ha='left')

ax.annotate(f'{ppl_mean[1]:.1f}',
            xy=(x_pos[1], ppl_mean[1]),
            xytext=(x_pos[1] + 0.45, ppl_mean[1] + 0.2),
            fontsize=7, color='#333333',
            arrowprops=dict(arrowstyle='->', color='#333333', lw=0.8))

best_idx = np.argmin(ppl_mean)
ax.annotate(f'{ppl_mean[best_idx]:.1f}',
            xy=(x_pos[best_idx], ppl_mean[best_idx]),
            xytext=(x_pos[best_idx] + 0.45, ppl_mean[best_idx] - 0.25),
            fontsize=7, color='#333333',
            arrowprops=dict(arrowstyle='->', color='#333333', lw=0.8))

ax.set_xticks(x_pos)
ax.set_xticklabels(['32\n(E2E)' if k == 32 else str(k) for k in K])
ax.set_xlabel('Coupling breadth $K$')
ax.set_ylabel('Perplexity (WikiText-2)')

ax.set_ylim(8.2, 11.3)

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color=COLOR_KL, label='3-seed mean$\\pm$std',
           markersize=5, markeredgecolor='white', markeredgewidth=0.4, linestyle=''),
    Line2D([0], [0], color=COLOR_TEAL, linestyle='--', label='TEAL baseline',
           linewidth=1.2),
]
ax.legend(handles=legend_elements, loc='upper right', frameon=True,
          fancybox=False, edgecolor='#cccccc', fontsize=7)

plt.tight_layout()
plt.savefig('figures/paper/k_curve_ppl.pdf')
plt.savefig('figures/paper/k_curve_ppl.png')
print('Saved k_curve_ppl.pdf and k_curve_ppl.png')

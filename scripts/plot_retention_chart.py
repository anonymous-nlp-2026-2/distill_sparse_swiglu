"""Generate per-layer retention chart combining 30%/50%/70% sparsity results."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ARTIFACTS = "/root/distill_sparse_swiglu/artifacts"

data_30 = json.load(open(f"{ARTIFACTS}/global_30pct_retention_analysis.json"))
data_50 = json.load(open(f"{ARTIFACTS}/global_50pct_retention_analysis.json"))
data_70 = json.load(open(f"{ARTIFACTS}/global_70pct_retention_analysis.json"))

layers = np.arange(32)
ret_30 = [p["mean_retention"] for p in data_30["per_layer_retention"]]
ret_50 = [p["mean_retention"] for p in data_50["per_layer_retention"]]
ret_70 = [p["mean_retention"] for p in data_70["per_layer_retention"]]

fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))

colors = {'30': '#2176AE', '50': '#E07A3E', '70': '#C02942'}
baselines = {'30': 0.7, '50': 0.5, '70': 0.3}

ax.plot(layers, ret_30, '-o', color=colors['30'], markersize=3.5, linewidth=1.5,
        label='30% sparsity', zorder=3)
ax.plot(layers, ret_50, '-s', color=colors['50'], markersize=3.5, linewidth=1.5,
        label='50% sparsity', zorder=3)
ax.plot(layers, ret_70, '-^', color=colors['70'], markersize=3.5, linewidth=1.5,
        label='70% sparsity', zorder=3)

for key, baseline in baselines.items():
    ax.axhline(y=baseline, color=colors[key], linestyle='--', linewidth=0.8, alpha=0.5)

for key, ret, baseline in [('30', ret_30, 0.7), ('50', ret_50, 0.5), ('70', ret_70, 0.3)]:
    starved_mask = np.array(ret) < baseline
    for i in range(32):
        if starved_mask[i]:
            ax.axvspan(i - 0.4, i + 0.4, alpha=0.08, color=colors[key], zorder=0)

ax.set_xlabel('Layer Index', fontsize=9)
ax.set_ylabel('Retention Ratio', fontsize=9)
ax.set_xlim(-0.5, 31.5)
ax.set_ylim(0, 1.0)
ax.set_xticks(np.arange(0, 32, 4))
ax.tick_params(labelsize=8)
ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.2, linewidth=0.5)

ax.annotate('uniform\n0.70', xy=(31.5, 0.70), fontsize=6.5, color=colors['30'],
            va='bottom', ha='right', alpha=0.7)
ax.annotate('uniform\n0.50', xy=(31.5, 0.50), fontsize=6.5, color=colors['50'],
            va='bottom', ha='right', alpha=0.7)
ax.annotate('uniform\n0.30', xy=(31.5, 0.30), fontsize=6.5, color=colors['70'],
            va='bottom', ha='right', alpha=0.7)

plt.tight_layout()
plt.savefig(f"{ARTIFACTS}/per_layer_retention_chart.pdf", bbox_inches='tight')
plt.savefig(f"{ARTIFACTS}/per_layer_retention_chart.png", dpi=300, bbox_inches='tight')
print("Saved: per_layer_retention_chart.pdf + .png")

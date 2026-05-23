import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

with open('/root/distill_sparse_swiglu/artifacts/lambda_sparsity_curves.json') as f:
    data = json.load(f)

configs = [
    ('kl_30pct_s42', '30%', 0.3),
    ('kl_40pct_s42', '40%', 0.4),
    ('kl_50pct_s42', '50%', 0.5),
    ('kl_70pct_s42', '70%', 0.7),
]

colors = plt.cm.tab10(np.arange(4))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 4.5), sharex=True,
                                layout='constrained',
                                gridspec_kw={'hspace': 0.05})

for i, (key, label, target) in enumerate(configs):
    d = data[key]
    steps = d['steps']
    lam = d['lambda']
    sp = d['sparsity']

    ax1.plot(steps, lam, color=colors[i], linewidth=1.5, label=f'{label} target')
    ax2.plot(steps, sp, color=colors[i], linewidth=1.5, label=f'{label} actual')
    ax2.axhline(y=target, color=colors[i], linewidth=1.0, linestyle='--', alpha=0.7)

# Panel A: lambda dynamics
ax1.set_yscale('log')
ax1.set_ylabel(r'$\lambda$', fontsize=9)
ax1.tick_params(labelsize=8)
ax1.grid(True, alpha=0.3, which='major')
ax1.legend(fontsize=7, loc='upper left', framealpha=0.9)
ax1.text(0.02, 0.92, '(a)', transform=ax1.transAxes, fontsize=9, fontweight='bold',
         va='top')

# Panel B: sparsity tracking
ax2.set_xlabel('Training steps', fontsize=9)
ax2.set_ylabel('Sparsity ratio', fontsize=9)
ax2.set_ylim(0.15, 0.85)
ax2.tick_params(labelsize=8)
ax2.grid(True, alpha=0.3, which='major')
ax2.legend(fontsize=7, loc='lower right', framealpha=0.9, ncol=1)
ax2.text(0.02, 0.92, '(b)', transform=ax2.transAxes, fontsize=9, fontweight='bold',
         va='top')

out_pdf = '/root/distill_sparse_swiglu/paper/figures/lagrangian_dynamics.pdf'
out_png = '/root/distill_sparse_swiglu/paper/figures/lagrangian_dynamics.png'
fig.savefig(out_pdf, bbox_inches='tight', pad_inches=0.02)
fig.savefig(out_png, bbox_inches='tight', pad_inches=0.02, dpi=300)
plt.close()

print(f'Saved: {out_pdf}')
print(f'Saved: {out_png}')

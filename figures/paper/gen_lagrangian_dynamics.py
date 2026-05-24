import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
})

raw = """1,0.406,0.50
10,0.406,0.01
20,0.406,0.01
30,0.402,0.01
40,0.402,0.01
50,0.398,0.01
60,0.391,0.01
70,0.383,0.01
80,0.375,0.01
90,0.363,0.01
100,0.344,0.01
110,0.328,0.01
120,0.309,0.01
130,0.289,0.01
140,0.266,0.32
150,0.254,327.68
160,0.324,163.84
170,0.320,0.16
180,0.305,0.08
190,0.293,0.08
200,0.277,0.32
210,0.266,327.68
220,0.312,2621.44
230,0.316,81.92
240,0.301,81.92
250,0.285,81.92
260,0.277,1310.72
270,0.305,5000.00
280,0.305,5000.00
290,0.293,5000.00
300,0.277,5000.00
310,0.309,5000.00
320,0.309,5000.00
330,0.293,5000.00
340,0.285,5000.00
350,0.312,5000.00
360,0.309,5000.00
370,0.293,5000.00
380,0.277,5000.00
390,0.316,5000.00
400,0.312,5000.00
410,0.301,5000.00
420,0.285,5000.00
430,0.301,5000.00
440,0.312,5000.00
450,0.309,5000.00
460,0.293,5000.00
470,0.293,5000.00
480,0.320,1250.00
490,0.305,625.00
500,0.293,625.00
510,0.281,625.00
520,0.285,2500.00
530,0.281,2500.00
540,0.328,2500.00
550,0.305,5000.00
560,0.320,9.77
570,0.316,2.44
580,0.312,2.44
590,0.305,2.44
600,0.301,2.44
610,0.297,2.44
620,0.297,2.44
630,0.289,2.44
640,0.289,2.44
650,0.281,2.44
660,0.281,2.44
670,0.277,9.77
680,0.281,1250.00
690,0.281,5000.00
700,0.293,5000.00
710,0.293,5000.00
720,0.289,5000.00
730,0.285,5000.00
740,0.285,5000.00
750,0.285,5000.00
760,0.281,5000.00
770,0.281,5000.00
780,0.285,5000.00
790,0.285,5000.00
800,0.289,5000.00
810,0.285,5000.00
820,0.285,5000.00
830,0.281,5000.00
840,0.281,5000.00
850,0.285,5000.00
860,0.285,5000.00
870,0.285,5000.00
880,0.285,5000.00
890,0.285,5000.00
900,0.285,5000.00
910,0.285,5000.00
920,0.289,5000.00
930,0.285,5000.00
940,0.285,5000.00
950,0.281,5000.00
960,0.277,5000.00
970,0.281,5000.00
980,0.281,5000.00
990,0.277,5000.00
1000,0.281,5000.00"""

steps, sparsities, lambdas = [], [], []
for line in raw.strip().split('\n'):
    s, sp, lam = line.split(',')
    steps.append(int(s))
    sparsities.append(float(sp))
    lambdas.append(float(lam))

steps = np.array(steps)
sparsities = np.array(sparsities)
lambdas = np.array(lambdas)

# Clamp lambda floor for log scale
lambdas_plot = np.clip(lambdas, 1e-2, None)

fig, ax1 = plt.subplots(figsize=(5, 3.5))

color_lambda = '#4472C4'
color_sparsity = '#E05A3A'

ax1.set_xlabel('Training Step')
ax1.set_ylabel(r'$\lambda$ (Lagrange multiplier)', color=color_lambda)
ax1.plot(steps, lambdas_plot, color=color_lambda, linewidth=1.2, label=r'$\lambda$')
ax1.set_yscale('log')
ax1.set_ylim(5e-3, 1e4)
ax1.tick_params(axis='y', labelcolor=color_lambda)

ax2 = ax1.twinx()
ax2.set_ylabel('Actual Sparsity', color=color_sparsity)
ax2.plot(steps, sparsities, color=color_sparsity, linewidth=1.2, label='Sparsity')
ax2.axhline(y=0.3, color=color_sparsity, linestyle='--', linewidth=0.8, alpha=0.7)
ax2.annotate('target = 0.3', xy=(1000, 0.3), xytext=(-60, -14),
             textcoords='offset points', fontsize=7, color=color_sparsity, alpha=0.8)
ax2.set_ylim(0.20, 0.45)
ax2.tick_params(axis='y', labelcolor=color_sparsity)

ax1.set_xlim(0, 1050)

ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

ax1.grid(True, alpha=0.2, color='grey', linewidth=0.5)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.9)

fig.tight_layout()

import pathlib
out_dir = pathlib.Path(__file__).parent
fig.savefig(out_dir / 'lagrangian_dynamics.pdf', bbox_inches='tight', dpi=300)
fig.savefig(out_dir / 'lagrangian_dynamics.png', bbox_inches='tight', dpi=300)
print(f'Saved to {out_dir / "lagrangian_dynamics.pdf"}')
print(f'Saved to {out_dir / "lagrangian_dynamics.png"}')

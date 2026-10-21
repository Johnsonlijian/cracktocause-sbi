"""Figure: calibration and accuracy of the amortized posterior on held-out simulations.

usage: python plot_calibration.py NPE_OUT_DIR CONFIG OUT.png
Top rows: rank of the true value among the posterior samples for every parameter (uniform if the
posterior is calibrated), with the coverage of the central 90% intervals. Bottom row (facade): posterior
median against the true value with 90% intervals for the settlement parameters and the interface
stiffness, for a random subset of held-out maps with cracks (blue) and without cracks (grey).
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'scripts'))
from train_npe import CONFIGS  # noqa: E402

BLUE, GREY, INK, MUTED, GRID = '#2a78d6', '#c3c2b7', '#0b0b0b', '#898781', '#e1e0d9'
LABELS = {'S': '$S$ (mm)', 'x_p': '$x_p$ (mm)', 'log_w': r'$\ln w$', 'log_kn': r'$\ln k_n$',
          'ft': '$f_t$ (MPa)', 'log_Gf': r'$\ln G_f$', 'q': '$q$', 'Ps': '$P_s$ (kN)', 'E': '$E$ (GPa)'}
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.titlesize': 8})

out_dir, config, out = sys.argv[1], sys.argv[2], sys.argv[3]
box = CONFIGS[config]['box']
names = box.names
z = np.load(os.path.join(out_dir, 'heldout.npz'))
rep = json.load(open(os.path.join(out_dir, 'report.json')))
cov = rep.get('heldout', rep)['coverage']['0.9']
th, med, ranks, lo, hi = z['theta'], z['median'], z['ranks'], z['q05'], z['q95']
cracked = z['cracked'].astype(bool) if 'cracked' in z.files else np.ones(len(th), bool)
scatter = ['S', 'x_p', 'log_kn'] if config == 'facade' else ['Ps']
D = len(names)
ncol = 4
nrow_r = int(np.ceil(D / ncol))
fig = plt.figure(figsize=(7.2, 1.55 * nrow_r + 2.5))
gs = fig.add_gridspec(nrow_r + 1, ncol, height_ratios=[1.0] * nrow_r + [1.55], hspace=0.75, wspace=0.35)
for j, n in enumerate(names):
    ax = fig.add_subplot(gs[j // ncol, j % ncol])
    ax.hist(ranks[:, j], bins=20, range=(0, 1), color=BLUE, edgecolor='white', linewidth=0.6)
    ax.axhline(len(ranks) / 20, color=MUTED, lw=0.8, ls='--')
    ax.set_title(f'{LABELS.get(n, n)}: coverage {cov[n]:.2f}', color=INK)
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1])
    ax.set_xlabel('rank', labelpad=1)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
rng = np.random.default_rng(0)
for k, n in enumerate(scatter):
    j = names.index(n)
    ax = fig.add_subplot(gs[nrow_r, k])
    for sel, color, lab in ((~cracked, GREY, 'no cracks'), (cracked, BLUE, 'cracks')):
        idx = np.flatnonzero(sel)
        idx = rng.choice(idx, size=min(120, len(idx)), replace=False) if len(idx) else idx
        ax.vlines(th[idx, j], lo[idx, j], hi[idx, j], color=color, lw=0.5, alpha=0.45)
        ax.plot(th[idx, j], med[idx, j], 'o', ms=2.6, color=color, mec='white', mew=0.3, label=lab)
    ax.plot([box.low[j], box.high[j]], [box.low[j], box.high[j]], color=INK, lw=0.8)
    ax.set_xlim(box.low[j], box.high[j])
    ax.set_ylim(box.low[j], box.high[j])
    ax.set_xlabel(f'true {LABELS.get(n, n)}')
    ax.set_ylabel('posterior median' if k == 0 else '')
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    if k == 0:
        handles, labels = ax.get_legend_handles_labels()
if len(scatter) < ncol:                                  # legend in the free cell of the bottom row
    lax = fig.add_subplot(gs[nrow_r, len(scatter)])
    lax.axis('off')
    lax.legend(handles, ['held-out maps without cracks', 'held-out maps with cracks'] if config == 'facade'
               else ['held-out simulations'] * len(handles), frameon=False, loc='center left', fontsize=7.5,
               markerscale=1.6, handletextpad=0.3)
fig.savefig(out, dpi=600, bbox_inches='tight')
print('saved', out)

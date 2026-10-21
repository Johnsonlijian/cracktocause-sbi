"""Figure: pairwise posterior marginals (corner plot) for one digitized experiment.

usage: python plot_pairs.py NPE_OUT_DIR CONFIG KEY OUT.png [param ...]
Reads post_<KEY>.npy written by scripts/train_npe.py; marks the test values of the experiment. The
default parameters are those of the settlement profile and the interface stiffness (facade) or all (den).
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'scripts'))
from train_npe import CONFIGS  # noqa: E402

LABELS = {'S': '$S$ (mm)', 'x_p': '$x_p$ (mm)', 'log_w': r'$\ln w$', 'log_kn': r'$\ln k_n$', 'ft': '$f_t$ (MPa)',
          'log_Gf': r'$\ln G_f$', 'q': '$q$', 'Ps': '$P_s$ (kN)', 'E': '$E$ (GPa)'}

out_dir, config, key, out = sys.argv[1:5]
cfg = CONFIGS[config]
box = cfg['box']
names = sys.argv[5:] or (['S', 'x_p', 'log_w', 'log_kn'] if config == 'facade' else list(box.names))
cols = [box.names.index(n) for n in names]
s = np.load(os.path.join(out_dir, f'post_{key}.npy'))[:, cols]
true = np.array(dict(cfg['exp'])[key.removesuffix('_w')], float)[cols]
lo, hi = box.low[cols], box.high[cols]
n = len(cols)
fig, axs = plt.subplots(n, n, figsize=(1.9 * n, 1.9 * n))
for i in range(n):
    for j in range(n):
        ax = axs[i, j]
        if j > i:
            ax.axis('off')
            continue
        if i == j:
            ax.hist(s[:, i], bins=40, range=(lo[i], hi[i]), color='C0', alpha=0.8)
            if np.isfinite(true[i]):
                ax.axvline(true[i], color='C3', lw=1.2)
            ax.set_yticks([])
            ax.set_xlim(lo[i], hi[i])
        else:
            ax.hist2d(s[:, j], s[:, i], bins=30, range=[[lo[j], hi[j]], [lo[i], hi[i]]], cmap='Blues')
            if np.isfinite(true[i]) and np.isfinite(true[j]):
                ax.plot(true[j], true[i], marker='+', color='C3', ms=9, mew=1.5)
        if i == n - 1:
            ax.set_xlabel(LABELS.get(names[j], names[j]))
        else:
            ax.set_xticklabels([])
        if j == 0 and i > 0:
            ax.set_ylabel(LABELS.get(names[i], names[i]))
        elif j > 0:
            ax.set_yticklabels([])
fig.tight_layout()
fig.savefig(out, dpi=600)
print('saved', out)

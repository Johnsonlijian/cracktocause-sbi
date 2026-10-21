"""Figure: posterior marginals for the digitized experiments (one row per observation).

usage: python plot_posteriors.py NPE_OUT_DIR CONFIG OUT.png      (CONFIG: facade | den)
Reads report.json and post_<key>.npy written by scripts/train_npe.py.
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

out_dir, config, out = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = CONFIGS[config]
box = cfg['box']
rep = json.load(open(os.path.join(out_dir, 'report.json')))
keys = list(rep.get('experiment', {}).keys())
D = len(box.names)
fig, axs = plt.subplots(len(keys), D, figsize=(2.1 * D, 1.5 * len(keys)), squeeze=False)
labels = {'S': '$S$ (mm)', 'x_p': '$x_p$ (mm)', 'log_w': r'$\ln w$', 'log_kn': r'$\ln k_n$', 'ft': '$f_t$ (MPa)',
          'log_Gf': r'$\ln G_f$', 'q': '$q$', 'Ps': '$P_s$ (kN)', 'E': '$E$ (GPa)'}
for r, k in enumerate(keys):
    s = np.load(os.path.join(out_dir, f'post_{k}.npy'))
    true = dict(cfg['exp'])[k.removesuffix('_w')]          # '_w': map and crack width
    for j, n in enumerate(box.names):
        ax = axs[r, j]
        ax.hist(s[:, j], bins=40, range=(box.low[j], box.high[j]), density=True, color='C0', alpha=0.7)
        if not np.isnan(true[j]):
            ax.axvline(true[j], color='C3', lw=1.2)
        ax.set_yticks([])
        ax.set_xlim(box.low[j], box.high[j])
        if r == len(keys) - 1:
            ax.set_xlabel(labels.get(n, n), fontsize=8)
        else:
            ax.set_xticklabels([])
        ax.tick_params(labelsize=7)
    axs[r, 0].set_ylabel(k, fontsize=8)
fig.tight_layout()
fig.savefig(out, dpi=140)
print('saved', out)

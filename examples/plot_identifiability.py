"""Figure: identifiability — posterior contraction (1 - width90 / prior width) per parameter versus the
settlement magnitude on held-out simulations, for maps with and without cracks; crack map alone (solid)
and crack map with the widest crack opening (dashed, if the estimator was trained with widths).

usage: python plot_identifiability.py NPE_OUT_DIR OUT.png [param ...]
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'scripts'))
from train_npe import CONFIGS  # noqa: E402

INK, MUTED, GRID = '#0b0b0b', '#898781', '#e1e0d9'
COLORS = {'x_p': '#2a78d6', 'log_kn': '#eb6834', 'S': '#1baf7a', 'log_w': '#4a3aa7'}   # fixed slot order
LABELS = {'S': 'settlement $S$', 'x_p': 'profile position $x_p$', 'log_w': r'transition width $\ln w$',
          'log_kn': r'interface stiffness $\ln k_n$'}
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED})

out_dir, out = sys.argv[1], sys.argv[2]
box = CONFIGS['facade']['box']
names = sys.argv[3:] or ['x_p', 'log_kn', 'S', 'log_w']
cols = [box.names.index(n) for n in names]


def contraction(path):
    z = np.load(path)
    c = 1.0 - (z['q95'] - z['q05']) / (0.9 * (box.high - box.low))
    has_w = z['has_width'] if 'has_width' in z.files else np.zeros(len(c), bool)
    return z['theta'], c, z['cracked'], has_w


th, c_map, cracked, _ = contraction(os.path.join(out_dir, 'heldout.npz'))
pw = os.path.join(out_dir, 'heldout_w.npz')
if os.path.exists(pw):
    _, c_w, _, has_w = contraction(pw)
else:
    c_w, has_w = None, np.zeros(len(th), bool)
sel_base = has_w if has_w.any() else np.ones(len(th), bool)     # compare both on the same pairs
S = th[:, 0]
bins = np.linspace(0.5, 12.5, 7)
mid = 0.5 * (bins[1:] + bins[:-1])


def binned(c, sel, j):
    return [np.mean(c[sel & (S >= a) & (S < b), j]) if np.any(sel & (S >= a) & (S < b)) else np.nan
            for a, b in zip(bins[:-1], bins[1:])]


fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
for ax, crk, title in [(axs[0], True, 'maps with cracks'), (axs[1], False, 'maps without cracks')]:
    sel = sel_base & (cracked == crk)
    for n, j in zip(names, cols):
        ax.plot(mid, binned(c_map, sel, j), '-o', ms=3.5, lw=1.6, color=COLORS[n], mec='white', mew=0.5)
        if c_w is not None:
            ax.plot(mid, binned(c_w, sel, j), '--', lw=1.2, color=COLORS[n])
    ax.set_xlabel('settlement $S$ (mm)')
    ax.set_title(f'{title} (n = {int(sel.sum())})', fontsize=8.5, color=INK)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.axhline(0, color=MUTED, lw=0.7)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
axs[0].set_ylabel('posterior contraction (90%)')
handles = [Line2D([], [], color=COLORS[n], lw=1.6, marker='o', ms=3.5, mec='white', label=LABELS[n]) for n in names]
if c_w is not None:
    handles += [Line2D([], [], color=MUTED, lw=1.6, label='crack map'),
                Line2D([], [], color=MUTED, lw=1.2, ls='--', label='crack map + widest opening')]
fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.12))
fig.tight_layout()
fig.savefig(out, dpi=600, bbox_inches='tight')
print('saved', out)

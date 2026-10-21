"""Figure: simulated crack maps for prior draws (rows) at increasing settlement (columns).

usage: python plot_dataset_examples.py DATA_DIR OUT.png [--n 6] [--stages 2 5 8 11]
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
from crackcause import giardina as G  # noqa: E402
from crackcause.settlement import hinge_profile  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('data_dir')
ap.add_argument('out')
ap.add_argument('--n', type=int, default=6)
ap.add_argument('--stages', type=float, nargs='+', default=[2.0, 5.0, 8.0, 11.0])
a = ap.parse_args()
files = sorted(glob.glob(os.path.join(a.data_dir, 'sims', 'sim_*.npz')))
sel = []
for f in files:
    z = np.load(f)
    if len(z['S']) and z['maps'][-1].sum() > 0:
        sel.append(f)
    if len(sel) >= a.n:
        break
nc = len(a.stages) + 1
fig, axs = plt.subplots(len(sel), nc, figsize=(2.6 * nc, 2.2 * len(sel)), squeeze=False)
x = np.linspace(0, G.W, 200)
for r, f in enumerate(sel):
    z = np.load(f)
    th = json.loads(str(z['theta']))
    ax = axs[r, 0]
    ax.plot(x, hinge_profile(x, th['x_p'], th['w']), 'k-')
    ax.set_ylim(-1.1, 0.1)
    ax.set_xlim(0, G.W)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.text(0.98, 0.05, f"x_p={th['x_p']:.0f}\nw={th['w']:.0f}\nk_n={th['kn']:.2f}\nf_t={th['ft']:.3f}\n"
            f"G_f={th['Gf']:.3f}\nq={th['q']:.2f}", transform=ax.transAxes, ha='right', va='bottom', fontsize=6)
    for c, S in enumerate(a.stages, start=1):
        ax = axs[r, c]
        j = int(np.argmin(np.abs(z['S'] - S)))
        ax.imshow(1 - z['maps'][j], cmap='gray', vmin=0, vmax=1, extent=(0, G.W, 0, G.H))
        for x0, x1, y0, y1 in G.OPENINGS:
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=0.4, color='0.5'))
        ax.set_xticks([])
        ax.set_yticks([])
        if r == 0:
            ax.set_title(f'S = {z["S"][j]:.1f} mm', fontsize=9)
axs[0, 0].set_title('profile g/S', fontsize=9)
fig.tight_layout()
fig.savefig(a.out, dpi=130)
print('saved', a.out, len(sel), 'simulations')

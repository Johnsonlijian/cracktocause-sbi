"""Render phase-field crack patterns of FEniCSx facade runs at selected settlements.

usage: python plot_fx_giardina.py RUN.npz OUT.png [S1 S2 ...]   (default S = 2.5 5 7.5 10)
"""
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402

z = np.load(sys.argv[1])
out = sys.argv[2]
want = [float(s) for s in sys.argv[3:]] or [2.5, 5.0, 7.5, 10.0]
coords, cells, S_all, D = z['coords'], z['cells'], z['S'], z['d']
dmg = z['dmg'] > 0.5
tri = mtri.Triangulation(coords[:, 0], coords[:, 1], cells)
picks = [int(np.argmin(np.abs(S_all - s))) for s in want if len(S_all)]
fig, axes = plt.subplots(1, len(picks), figsize=(3.6 * len(picks), 3.4), squeeze=False)
for ax, k in zip(axes[0], picks):
    ax.tripcolor(tri, D[k], shading='gouraud', cmap='Greys', vmin=0.0, vmax=1.0)
    ax.set_aspect('equal')
    ax.set_xlim(0, 1428)
    ax.set_ylim(0, 1186)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'S = {S_all[k]:g} mm', fontsize=10)
log = json.loads(str(z['log']))
fig.suptitle(f'{sys.argv[1].split("/")[-1]}  (steps {len(log)}, unconverged {sum(1 for r in log if not r["ok"])})',
             fontsize=8)
fig.tight_layout()
fig.savefig(out, dpi=170)
print('saved', out)

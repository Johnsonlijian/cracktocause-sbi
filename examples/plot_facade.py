"""Render phase-field crack patterns of facade runs (one row per scenario, last column = final state).

usage: python plot_facade.py <run_dir> <out_png> scen1 [scen2 ...]
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402

run_dir, out_png, scens = sys.argv[1], sys.argv[2], sys.argv[3:]
ncol = 3
fig, axes = plt.subplots(len(scens), ncol, figsize=(4.0 * ncol, 2.9 * len(scens)), squeeze=False)
for r, sc in enumerate(scens):
    z = np.load(os.path.join(run_dir, f'{sc}.npz'))
    nodes, tris, wall = z['nodes'], z['tris'], z['wall']
    tri = mtri.Triangulation(nodes[:, 0] / 1000.0, nodes[:, 1] / 1000.0, tris[wall])
    frames = list(zip(z['snap_lams'], z['snaps']))
    picks = [frames[len(frames) // 3], frames[2 * len(frames) // 3], (z['lam'][-1], z['d'])]
    for c, (lam, d) in enumerate(picks):
        ax = axes[r, c]
        ax.tripcolor(tri, d, shading='gouraud', cmap='Greys', vmin=0.0, vmax=1.0)
        ax.set_aspect('equal')
        ax.set_xlim(0, 10)
        ax.set_ylim(-0.1, 6)
        ax.set_title(f'{sc}: settlement factor {lam:.2f}', fontsize=9)
        ax.set_xticks([0, 5, 10])
        ax.set_yticks([0, 3, 6])
        ax.tick_params(labelsize=7)
fig.tight_layout()
fig.savefig(out_png, dpi=170)
print('saved', out_png)

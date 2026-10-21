"""Grid of final crack patterns: rows = settlement scenarios, columns = interface types.

usage: python plot_batch.py <run_dir> <out_png>
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402

run_dir, out_png = sys.argv[1], sys.argv[2]
scens = ['sagging', 'hogging', 'offset']
ifaces = [('nt', 'no-tension interface'), ('el', 'tension-capable interface')]
fig, axes = plt.subplots(len(scens), len(ifaces), figsize=(10.5, 8.6), squeeze=False)
for r, sc in enumerate(scens):
    for c, (key, label) in enumerate(ifaces):
        ax = axes[r, c]
        stem = os.path.join(run_dir, f'{sc}_{key}')
        if not os.path.exists(stem + '.npz'):
            ax.set_axis_off()
            ax.set_title(f'{sc}, {label}: missing', fontsize=9)
            continue
        z = np.load(stem + '.npz')
        info = json.load(open(stem + '.json'))
        nodes, tris, wall = z['nodes'], z['tris'], z['wall']
        tri = mtri.Triangulation(nodes[:, 0] / 1000.0, nodes[:, 1] / 1000.0, tris[wall])
        ax.tripcolor(tri, z['d'], shading='gouraud', cmap='Greys', vmin=0.0, vmax=1.0)
        p = info['params']
        xs = np.linspace(0.0, 10000.0, 200)
        S = info['mat']['S_max'] * np.exp(-(xs - p['x0']) ** 2 / (2.0 * p['i'] ** 2))
        ax.plot(xs / 1000.0, -0.25 - S / 25.0 * 0.6, color='tab:red', lw=1.2)   # trough (schematic scale)
        ax.text(0.05, -1.05, f"x0={p['x0'] / 1000:g} m, i={p['i'] / 1000:g} m, Smax={info['mat']['S_max']:g} mm",
                fontsize=7, color='tab:red')
        ax.set_aspect('equal')
        ax.set_xlim(0, 10)
        ax.set_ylim(-1.15, 6.05)
        ax.set_xticks([0, 5, 10])
        ax.set_yticks([0, 3, 6])
        ax.tick_params(labelsize=7)
        ax.set_title(f'{sc}, {label} (d max {z["d"].max():.2f}, {info["seconds"] / 60:.1f} min)', fontsize=8.5)
fig.tight_layout()
fig.savefig(out_png, dpi=170)
print('saved', out_png)

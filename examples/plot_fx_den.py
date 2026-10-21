"""Plot a DEN simulation (phase field, crack-line maps, P-delta) against the digitized experiment.

usage: python plot_fx_den.py RESULT.npz OUT.png [--case 4b]
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
from crackcause.observe_fx import load_exp_maps  # noqa: E402
EXP = os.path.join(HERE, '..', 'data_public', 'nooru1992_fig428_crackmaps.npz')

ap = argparse.ArgumentParser()
ap.add_argument('result')
ap.add_argument('out')
ap.add_argument('--case', default='4b')
a = ap.parse_args()
z = np.load(a.result)
coords, cells, dmg = z['coords'], z['cells'], z['dmg']
spec = z['spec'] if 'spec' in z.files else dmg > 0.5
tri = mtri.Triangulation(coords[:, 0], coords[:, 1], cells[spec])
nrec = len(z['delta'])
ex = load_exp_maps(EXP)
fig, axs = plt.subplots(2, max(nrec, 3), figsize=(3.2 * max(nrec, 3), 6.4))
for k in range(nrec):
    ax = axs[0, k]
    ax.tripcolor(tri, z['d'][k], shading='gouraud', cmap='inferno', vmin=0, vmax=1)
    ax.set_title(f'delta = {z["delta"][k] * 1e3:.0f} um')
    ax.set_aspect('equal')
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 200)
    ax.axis('off')
ax = axs[1, 0]
if len(z['maps']):
    sim = z['maps'][-1].astype(float)
    rgb = np.ones((100, 100, 3))
    rgb[ex[f'{a.case}_front'] > 0] = [0.85, 0.1, 0.1]
    rgb[ex[f'{a.case}_rear'] > 0] = [0.1, 0.4, 1.0]
    rgb[sim > 0] = [0, 0, 0]
    ax.imshow(rgb, extent=(0, 200, 0, 200))
    ax.set_title(f'sim (black, delta={z["delta"][-1] * 1e3:.0f} um)\nexp {a.case} front (red) / rear (blue)', fontsize=8)
ax = axs[1, 1]
c = z['curve']
ax.plot(c[:, 0] * 1e3, c[:, 1] / 1e3, 'k-')
ax.set_xlabel('delta (um)')
ax.set_ylabel('P (kN)')
ax.grid(alpha=0.3)
ax = axs[1, 2]
log = json.loads(str(z['log']))
ax.plot([r.get('delta', 0) * 1e3 for r in log if r.get('phase') == 2],
        [r['it'] for r in log if r.get('phase') == 2], 'o-', ms=2)
ax.set_xlabel('delta (um)')
ax.set_ylabel('staggered iterations')
for k in range(3, axs.shape[1]):
    axs[1, k].axis('off')
fig.suptitle(f'{os.path.basename(a.result)} status={z["status"]}')
fig.tight_layout()
fig.savefig(a.out, dpi=110)
print('saved', a.out)

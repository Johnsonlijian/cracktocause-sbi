"""Figure: forward-model setup — façade geometry, loads, base interface and settlement profiles.

usage: python plot_setup.py MESH.npz OUT.png
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
from crackcause import giardina as G  # noqa: E402

from crackcause.settlement import hinge_profile  # noqa: E402

mesh, out = sys.argv[1], sys.argv[2]
z = np.load(mesh)
coords, cells, dmg = z['coords'], z['cells'], z['dmg']
fig = plt.figure(figsize=(12, 5.2))
ax = fig.add_axes([0.04, 0.12, 0.52, 0.8])
tri = mtri.Triangulation(coords[:, 0], coords[:, 1], cells)
ax.tripcolor(tri, facecolors=np.where(dmg > 0.5, 0.85, 0.45), cmap='gray', vmin=0, vmax=1, edgecolors='none')
ax.triplot(tri, lw=0.08, color='0.6')
for x, y, F in G.LOADS:
    ax.annotate('', xy=(x, y), xytext=(x, y + 60 + F / 12.0),
                arrowprops=dict(arrowstyle='->', color='tab:blue', lw=0.9))
xs = np.linspace(0, G.W, 60)
for x in xs[::2]:
    ax.plot([x, x], [-8, -40], color='tab:green', lw=0.8)
ax.text(G.W / 2, -85, 'compression-only springs $k_n$, weak tangential springs $k_t$', ha='center',
        color='tab:green', fontsize=9)
ax.set_aspect('equal')
ax.set_xlim(-30, G.W + 30)
ax.set_ylim(-120, G.H + 160)
ax.axis('off')
ax.set_title('Façade of Giardina et al. (2013): openings, timber lintels (dark), loads (blue)', fontsize=10)
ax2 = fig.add_axes([0.62, 0.15, 0.35, 0.72])
d = np.genfromtxt(os.path.join(HERE, '..', 'data_public', 'giardina2013_fig8_applied_profiles.csv'),
                  delimiter=',', names=True)
x = np.linspace(0, G.W, 400)
for k, (col, S) in enumerate(zip(['v_a_mm', 'v_b_mm', 'v_c_mm', 'v_d_mm'], G.EXP_STAGES_S)):
    ax2.plot(d['x_mm'], d[col], 'o', ms=2, color=f'C{k}', alpha=0.5)
    ax2.plot(x, S * hinge_profile(x, 732.0, 98.0), '-', color=f'C{k}', lw=1.2,
             label=f'S = {S:.2f} mm' + (' (measured: dots, fit: line)' if k == 0 else ''))
rng = np.random.default_rng(3)
for _ in range(6):
    xp, w = rng.uniform(300, 1400), np.exp(rng.uniform(np.log(30), np.log(400)))
    ax2.plot(x, 5.0 * hinge_profile(x, xp, w), color='0.6', lw=0.7, ls='--')
ax2.plot([], [], color='0.6', ls='--', lw=0.7, label='prior draws (S = 5 mm)')
ax2.set_xlabel('x (mm)')
ax2.set_ylabel('ground displacement g(x) (mm)')
ax2.legend(fontsize=8, loc='lower right')
ax2.grid(alpha=0.3)
ax2.set_title('Settlement profile: smoothed hinge ($x_p$, $w$, $S$)', fontsize=10)
fig.savefig(out, dpi=600)
print('saved', out)

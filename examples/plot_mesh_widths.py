"""Figure: discretization sensitivity of the observations — crack lines coloured by their opening, for
reference facade runs with different element size h and regularization length ell, at three stages.

usage: python plot_mesh_widths.py OUT.png RESULT.npz [RESULT.npz ...]
RESULT.npz: outputs of fx_giardina.py (records S, d, u); labels are taken from the file names.
"""
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
from crackcause import giardina as G  # noqa: E402
from crackcause.observe_fx import PointInterpolator, Rasterizer, crack_lines, crack_widths  # noqa: E402

STAGES = (2.26, 4.54, 6.81)
out, results = sys.argv[1], sys.argv[2:]
fig, axs = plt.subplots(len(results), len(STAGES), figsize=(3.6 * len(STAGES), 2.9 * len(results)), squeeze=False)
im = None
for r, f in enumerate(results):
    z = np.load(f)
    c, cells, dmg = z['coords'], z['cells'], z['dmg']
    ras = Rasterizer(c, cells[dmg > 0.5], (0.0, G.W, 0.0, G.H), (119, 143))
    ip = PointInterpolator(c, cells)
    m = re.search(r'_l(\d+)_h(\d+)', os.path.basename(f))
    label = f'h = {m.group(2)} mm, $\\ell$ = {m.group(1)} mm' if m else os.path.basename(f)
    for col, S in enumerate(STAGES):
        ax = axs[r, col]
        k = np.flatnonzero(np.abs(z['S'] - S) < 1e-6)
        if not len(k):
            ax.axis('off')
            continue
        L = crack_lines(ras(z['d'][k[0]]), ras.inside, 0.9)
        W = crack_widths(L, ras, ip, z['u'][k[0]].reshape(-1, 2))
        lab, n = ndimage.label(L[:15] > 0, structure=np.ones((3, 3)))
        top = sorted((float(np.nanmax(W[:15][lab == i])) for i in range(1, n + 1)), reverse=True)
        ax.imshow(np.where(ras.inside, 1.0, 0.85), cmap='Greys_r', vmin=0, vmax=1, extent=(0, G.W, 0, G.H))
        im = ax.imshow(np.ma.masked_where(L == 0, W), cmap='viridis', vmin=0, vmax=9, interpolation='nearest',
                       extent=(0, G.W, 0, G.H))
        tops = ', '.join(f'{t:.1f}' for t in top[:3])
        ax.set_title(f'{label}, S = {S} mm\n$w_{{max}}$ = {np.nanmax(W[L > 0]):.1f} mm; top course: {tops}', fontsize=7.5)
        ax.set_xticks([])
        ax.set_yticks([])
if im is not None:
    fig.colorbar(im, ax=axs, shrink=0.6, label='crack opening (mm)')
fig.savefig(out, dpi=600, bbox_inches='tight')
print('saved', out)

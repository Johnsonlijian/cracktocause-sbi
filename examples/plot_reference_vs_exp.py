"""Figure: experimental crack maps (Giardina et al. 2013, Fig. 5) against reference simulations at the
experimental parameters (homogeneous strength and random-field realizations), at the four stages.

usage: python plot_reference_vs_exp.py OUT.png RESULT.npz [RESULT.npz ...]
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
from crackcause import giardina as G  # noqa: E402
from crackcause.observe_fx import Rasterizer, crack_lines, load_exp_maps  # noqa: E402

out, results = sys.argv[1], sys.argv[2:]
ex = load_exp_maps(os.path.join(HERE, '..', 'data_public', 'giardina2013_fig5_exp_crackmaps.npz'))
keys = ['S2.5', 'S5', 'S7.5', 'S10']
rows = 1 + len(results)
fig, axs = plt.subplots(rows, 4, figsize=(11, 2.25 * rows), squeeze=False)


def draw(ax, img, color, inside=None):
    rgb = np.ones((119, 143, 3))
    if inside is not None:
        rgb[~inside] = 0.88
    for x0, x1, y0, y1 in G.OPENINGS:
        r0, r1 = int((G.H - y1) / 10), int(np.ceil((G.H - y0) / 10))
        c0, c1 = int(x0 / 10), int(np.ceil(x1 / 10))
        rgb[r0:r1, c0:c1] = 0.88
    rgb[ndimage.binary_dilation(img > 0)] = color
    ax.imshow(rgb, extent=(0, G.W, 0, G.H))
    ax.set_xticks([])
    ax.set_yticks([])


for c, (k, S) in enumerate(zip(keys, G.EXP_STAGES_S)):
    draw(axs[0, c], ex[k], [0.8, 0.1, 0.1])
    axs[0, c].set_title(f'S = {S:.2f} mm', fontsize=9)
axs[0, 0].set_ylabel('experiment', fontsize=8)
for r, f in enumerate(results, start=1):
    z = np.load(f)
    ras = Rasterizer(z['coords'], z['cells'][z['dmg'] > 0.5], (0.0, G.W, 0.0, G.H), (119, 143))
    for c, S in enumerate(G.EXP_STAGES_S):
        j = np.flatnonzero(np.abs(z['S'] - S) < 1e-6)
        if len(j) == 0:
            axs[r, c].axis('off')
            continue
        m = crack_lines(ras(z['d'][j[0]]), ras.inside)
        draw(axs[r, c], m, [0.1, 0.1, 0.1], ras.inside)
    name = os.path.basename(f).replace('giardina_l20_h10_kn0.7_linear', '').replace('.npz', '').strip('_')
    axs[r, 0].set_ylabel(name or 'homogeneous', fontsize=8)
fig.tight_layout()
fig.savefig(out, dpi=600)
print('saved', out)

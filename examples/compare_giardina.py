"""Compare a reference facade simulation with the digitized crack maps of Giardina et al. (2013), Fig. 5.

usage: python compare_giardina.py RESULT.npz OUT.png [--threshold 0.9] [--tol 3]
Crack maps of the simulation are produced with the observation operator at the experimental stages
(S = 2.26, 4.54, 6.81, 9.10 mm at x = 30 mm); agreement is measured by precision and recall of crack
pixels within a tolerance of `tol` pixels (10 mm each) and by the symmetric mean distance (mm).
"""
import argparse
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

EXP = os.path.join(HERE, '..', 'data_public', 'giardina2013_fig5_exp_crackmaps.npz')


def agreement(sim, exp, tol):
    if sim.sum() == 0 or exp.sum() == 0:
        return dict(precision=np.nan, recall=np.nan, mean_dist_mm=np.nan)
    d_to_exp = ndimage.distance_transform_edt(exp == 0)
    d_to_sim = ndimage.distance_transform_edt(sim == 0)
    prec = float((d_to_exp[sim > 0] <= tol).mean())
    rec = float((d_to_sim[exp > 0] <= tol).mean())
    md = 10.0 * 0.5 * (d_to_exp[sim > 0].mean() + d_to_sim[exp > 0].mean())
    return dict(precision=prec, recall=rec, mean_dist_mm=float(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('result')
    ap.add_argument('out')
    ap.add_argument('--threshold', type=float, default=0.9)
    ap.add_argument('--tol', type=float, default=3.0)
    a = ap.parse_args()
    z = np.load(a.result)
    coords, cells, dmg = z['coords'], z['cells'], z['dmg']
    ras = Rasterizer(coords, cells[dmg > 0.5], (0.0, G.W, 0.0, G.H), (119, 143))
    ex = load_exp_maps(EXP)
    keys = ['S2.5', 'S5', 'S7.5', 'S10']
    S_rec = z['S']
    fig, axs = plt.subplots(2, 4, figsize=(16, 7))
    rows = []
    for k, (key, S_true) in enumerate(zip(keys, G.EXP_STAGES_S)):
        j = int(np.argmin(np.abs(S_rec - S_true)))
        dimg = ras(z['d'][j])
        sim = crack_lines(dimg, ras.inside, threshold=a.threshold)
        e = ex[key]
        m = agreement(sim, e, a.tol)
        rows.append((key, S_true, float(S_rec[j]), m))
        ax = axs[0, k]
        ax.imshow(np.where(ras.inside, dimg, np.nan), cmap='inferno', vmin=0, vmax=1, extent=(0, G.W, 0, G.H))
        ax.set_title(f'phase field, S = {S_rec[j]:.2f} mm')
        ax = axs[1, k]
        rgb = np.ones((119, 143, 3))
        rgb[~ras.inside] = 0.85
        rgb[ndimage.binary_dilation(e > 0)] = [0.85, 0.1, 0.1]
        rgb[sim > 0] = [0.0, 0.0, 0.0]
        ax.imshow(rgb, extent=(0, G.W, 0, G.H))
        ax.set_title(f'{key}: exp (red) vs sim (black)\nP={m["precision"]:.2f} R={m["recall"]:.2f} '
                     f'd={m["mean_dist_mm"]:.0f} mm', fontsize=9)
    for ax in axs.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(os.path.basename(a.result))
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    for key, St, Sr, m in rows:
        print(f'{key}: S_exp={St:.2f} S_sim={Sr:.2f} precision={m["precision"]:.2f} recall={m["recall"]:.2f} '
              f'mean distance={m["mean_dist_mm"]:.0f} mm')


if __name__ == '__main__':
    main()

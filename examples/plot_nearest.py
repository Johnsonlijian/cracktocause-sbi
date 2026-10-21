"""Figure and misspecification check in image space: the simulated crack maps closest to each experimental
map (symmetric chamfer distance), and the percentile of that distance among held-out simulations.

usage: python plot_nearest.py DATA_DIR OUT.png [--config facade|den] [--keys S2.5 S10] [--n 3]
The chamfer distance between two line maps is the mean distance (in pixels) from the lines of one map
to the nearest line of the other, averaged over both directions. The reference distribution uses one
random cracked map from each of up to 200 simulations, with neighbours from other simulations only.
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
from scipy import ndimage  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
from crackcause.observe_fx import load_exp_maps  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('data_dir')
ap.add_argument('out')
ap.add_argument('--config', choices=['facade', 'den'], default='facade')
ap.add_argument('--keys', nargs='+', default=None)
ap.add_argument('--n', type=int, default=3)
a = ap.parse_args()

facade = a.config == 'facade'
keys = a.keys or (['S2.5', 'S10'] if facade else ['4a_front', '4b_front', '4c_front'])
ex = load_exp_maps(os.path.join(HERE, '..', 'data_public', 'giardina2013_fig5_exp_crackmaps.npz' if facade
                                else 'nooru1992_fig428_crackmaps.npz'))
TH, M, SID = [], [], []
for f in sorted(glob.glob(os.path.join(a.data_dir, 'sims', 'sim_*.npz'))):
    z = np.load(f)
    if len(z['maps']) == 0:
        continue
    th = json.loads(str(z['theta']))
    pairs = zip(z['S'], z['maps']) if facade else [(0.0, z['maps'][0])]      # DEN: final map, d >= 0.5
    for s, m in pairs:
        if m.sum() > 0:
            TH.append((s, th['x_p'], th['w'], th['kn']) if facade else (th['Ps'], th['ft'], th['Gf'], th['E']))
            M.append(m > 0), SID.append(f)
TH, M, SID = np.array(TH), np.array(M), np.array(SID)
DT = np.array([ndimage.distance_transform_edt(~m) for m in M], np.float32)


def chamfer(e):
    de = ndimage.distance_transform_edt(~e)
    return 0.5 * (np.array([de[m].mean() for m in M]) + DT[:, e].mean(1))


rng = np.random.default_rng(0)
ref = []
for s in rng.choice(np.unique(SID), size=min(200, len(np.unique(SID))), replace=False):
    i = rng.choice(np.flatnonzero(SID == s))
    d = chamfer(M[i])
    d[SID == s] = np.inf
    ref.append(d.min())
ref = np.array(ref)
fig, axs = plt.subplots(len(keys), a.n + 1, figsize=(2.6 * (a.n + 1), 2.3 * len(keys)), squeeze=False)
for r, k in enumerate(keys):
    e = ex[k] > 0
    d = chamfer(e)
    order = []
    for i in np.argsort(d):                                # nearest maps from different simulations
        if SID[i] not in [SID[j] for j in order]:
            order.append(i)
        if len(order) == a.n:
            break
    pct = (ref < d[order[0]]).mean() * 100
    print(f'{k}: nearest chamfer {d[order[0]]:.2f} px (held-out reference median {np.median(ref):.2f}, '
          f'q90 {np.quantile(ref, 0.9):.2f}) -> percentile {pct:.0f}')
    axs[r, 0].imshow(e, cmap='Reds', vmin=0, vmax=1.2, interpolation='nearest')
    axs[r, 0].set_title(f'experiment {k}', fontsize=8)
    for c, i in enumerate(order, start=1):
        axs[r, c].imshow(M[i], cmap='Greys', vmin=0, vmax=1.2, interpolation='nearest')
        if facade:
            s, xp, w, kn = TH[i]
            ttl = f'd = {d[i]:.1f} px: S = {s:.1f}, $x_p$ = {xp:.0f},\nw = {w:.0f}, $k_n$ = {kn:.2f}'
        else:
            ps, ft, gf, E = TH[i]
            ttl = f'd = {d[i]:.1f} px: $P_s$ = {ps:.1f} kN,\n$f_t$ = {ft:.2f}, $G_f$ = {gf:.3f}, E = {E:.0f}'
        axs[r, c].set_title(ttl, fontsize=7)
    for ax in axs[r]:
        ax.set_xticks([]), ax.set_yticks([])
fig.tight_layout()
fig.savefig(a.out, dpi=600)
print('saved', a.out)

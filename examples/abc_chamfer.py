"""Baseline for the facade experiment: rejection ABC with the chamfer distance between crack maps
(optionally plus a widest-crack-width discrepancy), using the simulated maps of the campaign.

usage: python abc_chamfer.py DATA_DIR OUT.json [--accept 0.01] [--lam 3.0]
Accepts the fraction `accept` of simulated cracked maps closest to each experimental map and reports
quantiles of the accepted parameters. With widths (simulations that stored them), the distance is
chamfer [px] + lam * |ln(w_sim + 0.05) - ln(w_obs + 0.05)|.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
sys.path.insert(0, os.path.join(HERE, '..', 'scripts'))
from crackcause.observe_fx import load_exp_maps  # noqa: E402
from train_npe import CONFIGS, W0, exp_max_widths, max_width  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('data_dir')
ap.add_argument('out')
ap.add_argument('--accept', type=float, default=0.01)
ap.add_argument('--lam', type=float, default=3.0)
a = ap.parse_args()
cfg = CONFIGS['facade']
ex = load_exp_maps(os.path.join(HERE, '..', cfg['exp_file']))
wexp = exp_max_widths(os.path.join(HERE, '..', cfg['exp_widths']), cfg['exp_A'])
wexp = {k: (v if k in cfg['exp_width_keys'] else None) for k, v in wexp.items()}

TH, M, WM = [], [], []
for f in sorted(glob.glob(os.path.join(a.data_dir, 'sims', 'sim_*.npz'))):
    z = np.load(f)
    th = json.loads(str(z['theta']))
    has_w = 'widths' in z.files and len(z['widths']) == len(z['maps'])
    for j, (s, m) in enumerate(zip(z['S'], z['maps'])):
        if m.sum() > 0:
            TH.append([s] + cfg['theta_of'](th)), M.append(m > 0)
            WM.append(max_width(z['widths'][j], m) if has_w else np.nan)
TH, M, WM = np.array(TH), np.array(M), np.array(WM)
DT = np.array([ndimage.distance_transform_edt(~m) for m in M], np.float32)
names = cfg['box'].names
rep = dict(n_cracked_maps=int(len(M)), n_with_width=int(np.isfinite(WM).sum()), accept=a.accept, lam=a.lam)
for key, true in cfg['exp']:
    e = ex[key] > 0
    de = ndimage.distance_transform_edt(~e)
    ch = 0.5 * (np.array([de[m].mean() for m in M]) + DT[:, e].mean(1))
    variants = {'map': (ch, np.ones(len(M), bool))}
    if wexp.get(key) is not None and np.isfinite(WM).any():
        ok = np.isfinite(WM)
        variants['map+w'] = (ch + a.lam * np.abs(np.log(np.nan_to_num(WM) + W0) - np.log(wexp[key] + W0)), ok)
    for var, (dist, ok) in variants.items():
        idx = np.flatnonzero(ok)
        k = max(5, int(a.accept * len(idx)))
        acc = idx[np.argsort(dist[idx])[:k]]
        rep[f'{key}_{var}'] = {n: dict(q05=float(np.quantile(TH[acc, j], 0.05)), median=float(np.median(TH[acc, j])),
                                       q95=float(np.quantile(TH[acc, j], 0.95)), true=float(true[j]))
                               for j, n in enumerate(names)}
        rep[f'{key}_{var}']['n_accepted'] = int(k)
        rep[f'{key}_{var}']['max_distance'] = float(dist[acc].max())
        print(key, var, f'k={k}', {n: tuple(round(rep[f"{key}_{var}"][n][q], 2) for q in ('q05', 'median', 'q95'))
                                   for n in ('S', 'x_p', 'log_w', 'log_kn')})
json.dump(rep, open(a.out, 'w'), indent=1)

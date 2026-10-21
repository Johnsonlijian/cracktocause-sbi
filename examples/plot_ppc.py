"""Posterior predictive check for the facade experiment: crack maps re-simulated at posterior samples
(fx_ppc_facade.py), compared with the observed map.

usage: python plot_ppc.py PPC_DIR KEY OUT.png OUT.json [N_SHOW]
KEY: experimental stage of the posterior (e.g. S10). For every predictive run the map recorded at the
sampled settlement is used. Reports the symmetric chamfer distance (mm) of each predictive map to the
observed map and, as reference, the distances between predictive maps; and the widest opening of the
predictive maps against the measured one.
"""
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
sys.path.insert(0, os.path.join(HERE, '..', 'scripts'))
from crackcause.observe_fx import load_exp_maps  # noqa: E402
from train_npe import CONFIGS, exp_max_widths  # noqa: E402

ppc_dir, key, out_png, out_json = sys.argv[1:5]
cfg = CONFIGS['facade']
obs = load_exp_maps(os.path.join(HERE, '..', cfg['exp_file']))[key] > 0
w_obs = exp_max_widths(os.path.join(HERE, '..', cfg['exp_widths']), cfg['exp_A']).get(key)
runs = []
for f in sorted(glob.glob(os.path.join(ppc_dir, 'ppc_*.npz'))):
    z = np.load(f)
    S = z['S']
    if not len(S):
        continue
    j = int(np.argmin(np.abs(S - float(z['S_sample']))))
    runs.append(dict(theta=json.loads(str(z['theta'])), S=float(S[j]), S_sample=float(z['S_sample']),
                     map=z['maps'][j] > 0, wmax=float(z['wmax'][j]), status=str(z['status'])))


def chamfer(a, b):
    if not a.any() or not b.any():
        return np.nan
    da, db = ndimage.distance_transform_edt(~a), ndimage.distance_transform_edt(~b)
    return 10.0 * 0.5 * (da[b].mean() + db[a].mean())                       # mm (10 mm pixels)


d_obs = np.array([chamfer(obs, r['map']) for r in runs])
d_pair = np.array([chamfer(a['map'], b['map']) for i, a in enumerate(runs) for b in runs[i + 1:]])
rep = dict(key=key, n=len(runs), chamfer_obs_mm=[float(x) for x in d_obs],
           chamfer_obs_median=float(np.nanmedian(d_obs)), chamfer_between_median=float(np.nanmedian(d_pair)),
           wmax_pred=[r['wmax'] for r in runs], wmax_obs=w_obs, S=[r['S'] for r in runs],
           status=[r['status'] for r in runs])
json.dump(rep, open(out_json, 'w'), indent=1)
print(json.dumps({k: v for k, v in rep.items() if not isinstance(v, list)}, indent=1))
show = int(sys.argv[5]) if len(sys.argv) > 5 else 11          # predictive maps drawn (statistics use all)
n = min(len(runs), show) + 1
cols = min(n, 4)
rows = int(np.ceil(n / cols))
fig, axs = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.8 * rows), squeeze=False)
axs[0, 0].imshow(obs, cmap='Reds', vmin=0, vmax=1.2, interpolation='nearest')
axs[0, 0].set_title(f'experiment {key}' + (f'\n$w_{{max}}$ = {w_obs:.1f} mm' if w_obs else ''), fontsize=7.5)
for k, r in enumerate(runs[:n - 1], start=1):
    ax = axs[k // cols, k % cols]
    ax.imshow(r['map'], cmap='Greys', vmin=0, vmax=1.2, interpolation='nearest')
    th = r['theta']
    ax.set_title(f"S = {r['S']:.1f}, $x_p$ = {th['x_p']:.0f}, $k_n$ = {th['kn']:.2f}\n"
                 f"$w_{{max}}$ = {r['wmax']:.1f} mm, d = {d_obs[k - 1]:.0f} mm", fontsize=7)
for ax in axs.flat:
    ax.set_xticks([])
    ax.set_yticks([])
for k in range(n, rows * cols):
    axs[k // cols, k % cols].axis('off')
fig.tight_layout()
fig.savefig(out_png, dpi=600)
print('saved', out_png)

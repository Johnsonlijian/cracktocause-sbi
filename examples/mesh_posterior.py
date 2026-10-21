"""Does the discretization of the simulator change the inference? Paired comparison on the facade.

usage: python mesh_posterior.py DATA_H10 DATA_H5 OUT.json [--ensemble 5] [--n-pair 300]
Simulations 0..n_pair-1 were run on both meshes with identical parameters, stages and strength fields.
An ensemble is trained on the production-mesh simulations with index >= n_pair only, and evaluated on the
crack maps of the paired simulations from both meshes. Reports, per mesh: coverage of the 90% intervals,
contraction for maps with cracks, and the agreement of the paired maps themselves (both cracked, mean
distance between crack lines, absolute difference of the posterior medians).
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
from crackcause.npe import EnsemblePosterior, sample, train_streaming  # noqa: E402
from train_npe import CONFIGS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('data_h10')
ap.add_argument('data_h5')
ap.add_argument('out')
ap.add_argument('--ensemble', type=int, default=5)
ap.add_argument('--n-pair', type=int, default=300)
a = ap.parse_args()
cfg = CONFIGS['facade']
box = cfg['box']


def load(data_dir, keep):
    th, mp, gr, idx_s = [], [], [], []
    for f in sorted(glob.glob(os.path.join(data_dir, 'sims', 'sim_*.npz'))):
        i = int(os.path.basename(f)[4:10])
        if not keep(i):
            continue
        z = np.load(f)
        if len(z['maps']) == 0:
            continue
        base = cfg['theta_of'](json.loads(str(z['theta'])))
        for s, m in zip(z['S'], z['maps']):
            th.append([s] + base), mp.append(m), gr.append(i), idx_s.append((i, round(float(s), 6)))
    return np.array(th), np.array(mp, np.uint8), np.array(gr), idx_s


h5_ids = {int(os.path.basename(f)[4:10]) for f in glob.glob(os.path.join(a.data_h5, 'sims', 'sim_*.npz'))}
th_tr, mp_tr, gr_tr, _ = load(a.data_h10, lambda i: i >= a.n_pair)
th_10, mp_10, gr_10, key_10 = load(a.data_h10, lambda i: i < a.n_pair and i in h5_ids)
th_5, mp_5, gr_5, key_5 = load(a.data_h5, lambda i: i < a.n_pair)
# keep the stages recorded on both meshes (a run may stop early on one of them)
common = sorted(set(key_10) & set(key_5))
i10 = [key_10.index(k) for k in common]
i5 = [key_5.index(k) for k in common]
th_10, mp_10 = th_10[i10], mp_10[i10]
th_5, mp_5 = th_5[i5], mp_5[i5]
print(f'training: {len(np.unique(gr_tr))} simulations ({len(th_tr)} maps); paired test: {len(set(k[0] for k in common))} '
      f'simulations, {len(common)} maps', flush=True)
posts = []
for k in range(a.ensemble):
    p, _, _ = train_streaming(th_tr, mp_tr, gr_tr, box, cfg['shape'], epochs=300, patience=20, seed=k,
                              log=lambda m: None, **cfg['train_kw'])
    posts.append(p)
post = EnsemblePosterior(posts) if len(posts) > 1 else posts[0]
rep = dict(n_train_sims=int(len(np.unique(gr_tr))), n_pair_sims=int(len(set(k[0] for k in common))),
           n_pair_maps=int(len(common)))
meds = {}
for tag, th, mp in (('h10', th_10, mp_10), ('h5', th_5, mp_5)):
    smp = sample(post, box, mp, n=1000)
    lo, hi, med = np.quantile(smp, 0.05, 1), np.quantile(smp, 0.95, 1), np.median(smp, 1)
    meds[tag] = med
    crk = mp.reshape(len(mp), -1).sum(1) > 0
    cov = (th >= lo) & (th <= hi)
    con = 1.0 - (hi - lo) / (0.9 * (box.high - box.low))
    rep[tag] = dict(cracked_fraction=float(crk.mean()), coverage90=dict(zip(box.names, map(float, cov.mean(0)))),
                    coverage90_cracked=dict(zip(box.names, map(float, cov[crk].mean(0)))),
                    contraction_cracked=dict(zip(box.names, map(float, con[crk].mean(0)))))
# agreement of the paired maps
both = (mp_10.reshape(len(mp_10), -1).sum(1) > 0) & (mp_5.reshape(len(mp_5), -1).sum(1) > 0)
either = (mp_10.reshape(len(mp_10), -1).sum(1) > 0) | (mp_5.reshape(len(mp_5), -1).sum(1) > 0)
dist = []
for m10, m5 in zip(mp_10[both], mp_5[both]):
    d10, d5 = ndimage.distance_transform_edt(m10 == 0), ndimage.distance_transform_edt(m5 == 0)
    dist.append(10.0 * 0.5 * (d10[m5 > 0].mean() + d5[m10 > 0].mean()))
dist = np.array(dist)
jx = box.names.index('x_p')
rep['paired_maps'] = dict(cracked_on_one_mesh_only=float(((~both) & either).sum() / max(either.sum(), 1)),
                          chamfer_mm_median=float(np.median(dist)) if len(dist) else None,
                          chamfer_mm_q90=float(np.quantile(dist, 0.9)) if len(dist) else None,
                          frac_within_50mm=float(np.mean(dist < 50.0)) if len(dist) else None,
                          median_abs_diff_xp_median_mm=float(np.median(np.abs(meds['h10'][both, jx] - meds['h5'][both, jx]))),
                          median_abs_diff_S_median_mm=float(np.median(np.abs(meds['h10'][both, 0] - meds['h5'][both, 0]))))
json.dump(rep, open(a.out, 'w'), indent=1)
print(json.dumps(rep, indent=1))

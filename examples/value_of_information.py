"""What limits the information of a crack map about the settlement magnitude? Model-free analysis.

usage: python value_of_information.py DATA_DIR OUT.json [--sets map,map+kn,...]
1. Same-simulation bound: with every parameter and the strength field known, the maps of one simulation are
   compared with each other; the settlements whose maps are indistinguishable from a given map (symmetric
   chamfer distance and relative change of the crack length below two tolerance levels) form the interval of
   S that the map leaves open. This is an upper bound for the information of a map on S.
2. Known parameters: quantile gradient boosting of S (5%, 50%, 95%) on the crack-map features of the
   estimator (column and row bands, total length, coarse grid; log scale) alone and together with the true
   values of subsets of the other parameters (ln k_n with a measurement error of sd 0.3), with five-fold
   cross-validation grouped by simulation. The strength field stays unknown. Quantile regression intervals
   under-cover, so they are calibrated by conformalized quantile regression (Romano et al. 2019): every
   interval is widened by the 90% quantile of the out-of-fold scores max(q05 - S, S - q95).
Both on maps with cracks; contraction = 1 - mean width of the 90% interval / 90% of the prior range.
The maps, features and same-simulation results are cached in OUT.cache.npz (delete it after new data).
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

ap = argparse.ArgumentParser()
ap.add_argument('data_dir')
ap.add_argument('out')
ap.add_argument('--sets', default='map,map+kn,map+xp+w,map+kn+xp+w,map+ft+Gf+q,map+all')
a = ap.parse_args()
S_LO, S_HI, STEP = 0.5, 12.5, 0.5
PRIOR90 = 0.9 * (S_HI - S_LO)
files = sorted(glob.glob(os.path.join(a.data_dir, 'sims', 'sim_*.npz')))


def feats(m, bx=5, by=10, cy=40):
    ny, nx = m.shape
    m = m.astype(float)
    cols = np.add.reduceat(m.sum(0), np.arange(0, nx, bx))
    rows = np.add.reduceat(m.sum(1), np.arange(0, ny, by))
    coarse = np.add.reduceat(np.add.reduceat(m, np.arange(0, ny, cy), 0), np.arange(0, nx, bx), 1).ravel()
    return np.log1p(np.r_[cols, rows, m.sum(), coarse])


def chamfer_matrix(maps):
    """Symmetric mean distance (pixels) between the crack lines of every pair of maps."""
    n = len(maps)
    dt = [ndimage.distance_transform_edt(m == 0) for m in maps]
    on = [m > 0 for m in maps]
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if on[i].any() and on[j].any():
                C[i, j] = C[j, i] = 0.5 * (dt[i][on[j]].mean() + dt[j][on[i]].mean())
            elif on[i].any() or on[j].any():
                C[i, j] = C[j, i] = np.inf
    return C


TOLS = ((0.5, 0.10), (1.0, 0.20))                # (chamfer in pixels, relative change of the crack length)
X, TH, S, G, DS = [], [], [], [], []
same_sim = {t: dict(width=[], dS=[]) for t in TOLS}
cache = a.out.replace('.json', '') + '.cache.npz'
if os.path.exists(cache):
    files = []                                    # everything below the loop comes from the cache
    cz = np.load(cache)
    X, TH, S, G, DS = (list(cz[n]) for n in ('X', 'TH', 'S', 'G', 'DS'))
    for i, t in enumerate(TOLS):
        same_sim[t] = dict(width=list(cz[f'w{i}']), dS=list(cz[f'd{i}']))
for k, f in enumerate(files):
    z = np.load(f)
    s_k, maps = z['S'], z['maps']
    if len(s_k) == 0:
        continue
    th = json.loads(str(z['theta']))
    t = [th['x_p'], np.log(th['w']), np.log(th['kn']), th['ft'], np.log(th['Gf']), th['q']]
    px = maps.reshape(len(maps), -1).sum(1)
    on = np.flatnonzero(px >= 3)
    if len(on) == 0:
        continue
    sc = s_k[on[0]]
    C = chamfer_matrix(maps)
    for j in on:
        X.append(feats(maps[j])), TH.append(t), S.append(s_k[j]), G.append(k), DS.append(s_k[j] - sc)
        for tol_c, tol_n in TOLS:
            same = [abs(int(px[j]) - int(px[i])) <= tol_n * max(px[j], px[i], 1) and C[j, i] <= tol_c
                    for i in range(len(s_k))]
            lo = j
            while lo > 0 and same[lo - 1]:
                lo -= 1
            hi = j
            while hi < len(s_k) - 1 and same[hi + 1]:
                hi += 1
            s_lo = s_k[lo] - STEP / 2 if lo > 0 else S_LO
            s_hi = s_k[hi] + STEP / 2 if hi < len(s_k) - 1 else S_HI
            same_sim[(tol_c, tol_n)]['width'].append(s_hi - s_lo)
            same_sim[(tol_c, tol_n)]['dS'].append(s_k[j] - sc)
X, TH, S, G, DS = map(np.array, (X, TH, S, G, DS))
if not os.path.exists(cache):
    np.savez_compressed(cache, X=X, TH=TH, S=S, G=G, DS=DS,
                        **{f'w{i}': np.array(same_sim[t]['width']) for i, t in enumerate(TOLS)},
                        **{f'd{i}': np.array(same_sim[t]['dS']) for i, t in enumerate(TOLS)})
bins = [0, 1, 2, 4, 8, 12]
rep = dict(n_maps=int(len(S)), n_sims=int(len(np.unique(G))), bins_dS=bins, same_simulation={}, known_parameters={})
for (tol_c, tol_n), v in same_sim.items():
    con, d = 1 - np.array(v['width']) / PRIOR90, np.array(v['dS'])
    rep['same_simulation'][f'chamfer{tol_c}_length{tol_n}'] = dict(
        contraction90=float(con.mean()),
        contraction90_by_dS=[float(con[(d >= b0) & (d < b1)].mean()) for b0, b1 in zip(bins[:-1], bins[1:])])
cols = dict(xp=0, w=1, kn=2, ft=3, Gf=4, q=5)
rng = np.random.default_rng(0)
for name in a.sets.split(','):
    keys = list(cols) if name == 'map+all' else [p for p in name.split('+')[1:]]
    T = TH[:, [cols[p] for p in keys]].copy() if keys else np.zeros((len(S), 0))
    if 'kn' in keys and name != 'map+all':
        T[:, keys.index('kn')] += 0.3 * rng.standard_normal(len(S))     # measured with sd 0.3 in ln k_n
    F = np.column_stack([X, T])
    q = {p: np.zeros(len(S)) for p in (0.05, 0.5, 0.95)}
    for tr, te in GroupKFold(5).split(F, S, G):
        for p in q:
            r = HistGradientBoostingRegressor(loss='quantile', quantile=p, max_iter=200, learning_rate=0.08,
                                              min_samples_leaf=40, random_state=0).fit(F[tr], S[tr])
            q[p][te] = r.predict(F[te])
    q05, q95 = q[0.05], np.maximum(q[0.95], q[0.05])
    score = np.maximum(q05 - S, S - q95)              # conformalized quantile regression
    c = np.sort(score)[min(int(np.ceil((len(S) + 1) * 0.9)), len(S)) - 1]
    raw_lo, raw_hi = np.clip(q05, S_LO, S_HI), np.clip(q95, S_LO, S_HI)
    lo, hi = np.clip(q05 - c, S_LO, S_HI), np.clip(q95 + c, S_LO, S_HI)
    con = 1 - (hi - lo) / PRIOR90
    rep['known_parameters'][name] = dict(
        coverage90=float(np.mean((S >= lo) & (S <= hi))), contraction90=float(con.mean()),
        contraction90_by_dS=[float(con[(DS >= b0) & (DS < b1)].mean()) for b0, b1 in zip(bins[:-1], bins[1:])],
        upper_at_prior_edge=float(np.mean(hi > S_HI - 1.0)), conformal_widening=float(c),
        raw_coverage90=float(np.mean((S >= raw_lo) & (S <= raw_hi))),
        raw_contraction90=float(1 - np.mean(raw_hi - raw_lo) / PRIOR90),
        r2_median=float(1 - np.mean((S - q[0.5]) ** 2) / np.var(S)))
    print(name, json.dumps(rep['known_parameters'][name]), flush=True)
json.dump(rep, open(a.out, 'w'), indent=1)
print(json.dumps(rep['same_simulation'], indent=1))

"""Are the mechanisms robust to mesh refinement? Paired comparison of the facade campaign on two meshes.

usage: python mesh_mechanisms.py DATA_H10 DATA_H5 OUT.json [--n-pair 150]
Simulations 0..n_pair-1 were run on both meshes with identical parameters, stages and strength fields. For
every pair, at the records present on both meshes: onset settlement S_c (first record with at least three
crack pixels), hinge at the last common record (crackcause.hinges, component and band criteria), and the
mean distance between the paired crack maps.
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
from crackcause.hinges import band_hinge, hinge_of, main_crack  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('data_h10')
ap.add_argument('data_h5')
ap.add_argument('out')
ap.add_argument('--n-pair', type=int, default=150)
a = ap.parse_args()


def chamfer_mm(p, q):
    dp, dq = ndimage.distance_transform_edt(p == 0), ndimage.distance_transform_edt(q == 0)
    return 10.0 * 0.5 * (dp[q > 0].mean() + dq[p > 0].mean())


pairs = []
for i in range(a.n_pair):
    f10 = os.path.join(a.data_h10, 'sims', f'sim_{i:06d}.npz')
    f5 = os.path.join(a.data_h5, 'sims', f'sim_{i:06d}.npz')
    if not (os.path.exists(f10) and os.path.exists(f5)):
        continue
    z10, z5 = np.load(f10), np.load(f5)
    k10 = {round(float(s), 6): j for j, s in enumerate(z10['S'])}
    k5 = {round(float(s), 6): j for j, s in enumerate(z5['S'])}
    common = sorted(set(k10) & set(k5))
    if not common:
        continue
    m10 = np.array([z10['maps'][k10[s]] for s in common])
    m5 = np.array([z5['maps'][k5[s]] for s in common])
    p10, p5 = m10.reshape(len(common), -1).sum(1), m5.reshape(len(common), -1).sum(1)
    on10, on5 = np.flatnonzero(p10 >= 3), np.flatnonzero(p5 >= 3)
    s10, s5 = band_hinge(m10[-1])[1], band_hinge(m5[-1])[1]
    line10 = max(s10, key=s10.get) if max(s10.values()) >= 0.1 else 'none'   # line the crack heads for
    line5 = max(s5, key=s5.get) if max(s5.values()) >= 0.1 else 'none'
    r = dict(i=i, n_common=len(common), S_last=common[-1], status_h5=str(z5['status']), line_h10=line10,
             line_h5=line5, span_h10=s10, span_h5=s5,
             Sc_h10=float(common[on10[0]]) if len(on10) else None, Sc_h5=float(common[on5[0]]) if len(on5) else None,
             hinge_h10=hinge_of(*main_crack(m10[-1])), hinge_h5=hinge_of(*main_crack(m5[-1])),
             band_h10=band_hinge(m10[-1])[0], band_h5=band_hinge(m5[-1])[0],
             chamfer_last_mm=float(chamfer_mm(m10[-1], m5[-1])) if p10[-1] and p5[-1] else None)
    pairs.append(r)
n = len(pairs)
both = [r for r in pairs if r['Sc_h10'] is not None and r['Sc_h5'] is not None]
one = [r for r in pairs if (r['Sc_h10'] is None) != (r['Sc_h5'] is None)]
dln = np.array([np.log(r['Sc_h5'] / r['Sc_h10']) for r in both])
step = np.array([abs(r['Sc_h5'] - r['Sc_h10']) for r in both])
mech = [r for r in pairs if r['band_h10'] != 'none' or r['band_h5'] != 'none']
table = {}
for r in mech:
    key = f"{r['band_h10']}->{r['band_h5']}"
    table[key] = table.get(key, 0) + 1
same = sum(r['band_h10'] == r['band_h5'] for r in mech)
ch = np.array([r['chamfer_last_mm'] for r in both if r['chamfer_last_mm'] is not None])
lines_both = [r for r in pairs if r['line_h10'] != 'none' and r['line_h5'] != 'none']
diff = [r for r in lines_both if r['line_h10'] != r['line_h5']]
for r in diff:                                         # pivots and hinges of the pairs whose line differs
    r['x_p'] = json.loads(str(np.load(os.path.join(a.data_h10, 'sims', f"sim_{r['i']:06d}.npz"))['theta']))['x_p']
lines_one = [r for r in pairs if (r['line_h10'] == 'none') != (r['line_h5'] == 'none')]
rep = dict(n_pairs=n, n_failed_h5=int(sum(r['status_h5'] == 'failed' for r in pairs)),
           n_cracking_both=len(both), n_cracking_one_mesh=len(one),
           onset_ln_ratio_median=float(np.median(dln)) if len(dln) else None,
           onset_ln_ratio_q10_q90=[float(v) for v in np.quantile(dln, [0.1, 0.9])] if len(dln) else None,
           onset_within_one_record=float(np.mean(step <= 0.51)) if len(step) else None,
           n_with_hinge=len(mech), hinge_same=int(same), hinge_table=table,
           n_line_both=len(lines_both), line_same=int(sum(r['line_h10'] == r['line_h5'] for r in lines_both)),
           n_line_one_mesh=len(lines_one),
           line_diff=[dict(i=r['i'], x_p=float(r['x_p']), h10=r['line_h10'], h5=r['line_h5']) for r in diff], n_hinge_both=int(sum(r['band_h10'] != 'none' and r['band_h5'] != 'none'
                                                                for r in pairs)),
           hinge_same_when_both=int(sum(r['band_h10'] == r['band_h5'] != 'none' for r in pairs)),
           chamfer_last_mm_median=float(np.median(ch)) if len(ch) else None,
           chamfer_last_within_50mm=float(np.mean(ch < 50.0)) if len(ch) else None,
           pairs=pairs)
json.dump(rep, open(a.out, 'w'), indent=1)
print(json.dumps({k: v for k, v in rep.items() if k != 'pairs'}, indent=1))

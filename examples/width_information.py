"""How much does the opening of the widest crack tell about the settlement magnitude? Explained variance of
S among cracked facade maps of simulations that stored crack widths.

usage: python width_information.py DATA_DIR OUT.json
Reports R^2 of linear regressions of S on log(w_max + 0.05 mm), on the log crack length, on both, and on
log w_max together with the true values of all other parameters (an upper bound for what the width can
add when everything else is known), and the spread of w_max within settlement bins.
"""
import glob
import json
import os
import sys
import zipfile

import numpy as np

rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], 'sims', 'sim_*.npz'))):
    if 'widths.npy' not in zipfile.ZipFile(f).namelist():
        continue
    z = np.load(f)
    th = json.loads(str(z['theta']))
    for s, m, w in zip(z['S'], z['maps'], z['widths']):
        if m.sum() == 0:
            continue
        v = np.nan_to_num(np.asarray(w, float))[m > 0]
        rows.append([s, np.log(v.max() + 0.05), np.log(m.sum()), th['x_p'], np.log(th['w']), np.log(th['kn']),
                     th['ft'], np.log(th['Gf']), th['q']])
a = np.array(rows)
S = a[:, 0]


def r2(cols):
    X = np.column_stack([np.ones(len(S)), a[:, cols]])
    e = S - X @ np.linalg.lstsq(X, S, rcond=None)[0]
    return float(1.0 - e.var() / S.var())


rep = dict(n_maps=int(len(a)), r2_width=r2([1]), r2_length=r2([2]), r2_both=r2([1, 2]),
           r2_width_all_true=r2([1, 3, 4, 5, 6, 7, 8]), corr_width=float(np.corrcoef(a[:, 1], S)[0, 1]))
bins = {}
for lo, hi in ((0, 3), (3, 6), (6, 9), (9, 13)):
    m = (S >= lo) & (S < hi)
    q = np.exp(np.quantile(a[m, 1], [0.1, 0.5, 0.9])) - 0.05
    bins[f'{lo}-{hi}'] = dict(n=int(m.sum()), q10=float(q[0]), median=float(q[1]), q90=float(q[2]))
rep['wmax_by_S'] = bins
json.dump(rep, open(sys.argv[2], 'w'), indent=1)
print(json.dumps(rep, indent=1))

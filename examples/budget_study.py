"""Simulation-budget study: calibration and contraction of the facade estimator trained on subsets of the
training simulations (train_npe.py --n-train-sims N), evaluated on the same held-out simulations.

usage: python budget_study.py OUT.json NPE_DIR:N [NPE_DIR:N ...]
N is the number of training simulations of each estimator (for the full run, the number used).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'scripts'))
from train_npe import CONFIGS  # noqa: E402

box = CONFIGS['facade']['box']
names = box.names
rows = []
for arg in sys.argv[2:]:
    d, n = arg.rsplit(':', 1)
    z = np.load(os.path.join(d, 'heldout.npz'))
    c = 1.0 - (z['q95'] - z['q05']) / (0.9 * (box.high - box.low))
    crk = z['cracked'].astype(bool)
    cov = (z['theta'] >= z['q05']) & (z['theta'] <= z['q95'])
    rows.append(dict(n_train_sims=int(n), coverage90_mean=float(cov.mean()), coverage90_min=float(cov.mean(0).min()),
                     **{f'contraction_cracked_{p}': float(c[crk, names.index(p)].mean())
                        for p in ('S', 'x_p', 'log_kn', 'log_w')}))
json.dump(rows, open(sys.argv[1], 'w'), indent=1)
for r in rows:
    print(r)

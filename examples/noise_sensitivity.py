"""Sensitivity of the facade posterior to the noise regularization of the settlement magnitude.

usage: python noise_sensitivity.py DATA_DIR OUT.json
Two single estimators with the production settings (scripts/train_npe.py, seed 0, same held-out
simulations), one with noise of 0.3 standard deviations on all parameters (production) and one without
noise on S. Reports coverage and contraction of the 90% intervals of S and x_p on held-out maps with and
without cracks, and the range of the coverage over all parameters.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
sys.path.insert(0, os.path.join(HERE, '..', 'scripts'))
from crackcause.npe import sample, train_streaming  # noqa: E402
from train_npe import CONFIGS, load  # noqa: E402

data_dir, out = sys.argv[1:3]
cfg = CONFIGS['facade']
box = cfg['box']
theta, maps, groups, extra = load(data_dir, cfg)
sims = np.unique(groups)
rng = np.random.default_rng(0)
test = np.isin(groups, rng.choice(sims, size=max(1, int(0.1 * len(sims))), replace=False))
idx = np.flatnonzero(test)
crk = maps[idx].reshape(len(idx), -1).sum(1) > 0
pw = 0.9 * (box.high - box.low)
rep = {}
for variant, zn in (('noise_all', 0.3), ('no_noise_on_S', [0.0] + [0.3] * (len(box.names) - 1))):
    kw = dict(cfg['train_kw'], z_noise=zn)
    p, _, hist = train_streaming(theta[~test], maps[~test], groups[~test], box, cfg['shape'], epochs=300,
                                 patience=20, seed=0, log=lambda m: None,
                                 extra=None if extra is None else extra[~test], **kw)
    ex = None if extra is None else np.zeros((len(idx), 2), np.float32)
    smp = sample(p, box, maps[idx], n=1000, extra=ex)
    lo, hi = np.quantile(smp, 0.05, 1), np.quantile(smp, 0.95, 1)
    cov = (theta[idx] >= lo) & (theta[idx] <= hi)
    con = 1 - (hi - lo) / pw
    rep[variant] = dict(val_loss=float(min(h[1] for h in hist)), epochs=len(hist), **{
        lab: dict(coverage_S=float(cov[m, 0].mean()), contraction_S=float(con[m, 0].mean()),
                  coverage_xp=float(cov[m, 1].mean()), contraction_xp=float(con[m, 1].mean()),
                  coverage_all=[float(cov[m].mean(0).min()), float(cov[m].mean(0).max())])
        for lab, m in (('cracked', crk), ('uncracked', ~crk))})
    print(variant, json.dumps(rep[variant]), flush=True)
json.dump(rep, open(out, 'w'), indent=1)

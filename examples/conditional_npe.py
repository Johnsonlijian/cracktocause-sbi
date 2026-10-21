"""How much does a crack map tell about S once the interface stiffness is measured? Conditional estimator.

usage: python conditional_npe.py DATA_DIR kn N_ENSEMBLE OUT.json
kn: ln k_n is observed with a standard deviation of 0.3 (as from a soil investigation) and enters the
    estimator as an additional input; during training it is perturbed by that noise and hidden in 30% of the
    pairs, so that the same estimator gives the posterior with and without the measurement.
Same held-out simulations as scripts/train_npe.py (seed 0, 10%). Reports coverage and contraction of the
90% intervals for held-out maps with and without cracks, and writes the interval bounds next to OUT.json.
(An "oracle" variant that receives all other parameters exactly was dropped: its early stopping is driven by
the six known dimensions and it stopped before learning S; examples/value_of_information.py answers that
question without a neural estimator.)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
sys.path.insert(0, os.path.join(HERE, '..', 'scripts'))
from crackcause.npe import EnsemblePosterior, sample, train_streaming  # noqa: E402
from train_npe import CONFIGS, load  # noqa: E402

data_dir, variant, n_ens, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
cfg = CONFIGS['facade']
box = cfg['box']
theta, maps, groups, _ = load(data_dir, cfg)
sims = np.unique(groups)
rng = np.random.default_rng(0)
test_sims = rng.choice(sims, size=max(1, int(0.1 * len(sims))), replace=False)
test = np.isin(groups, test_sims)
names = box.names
J = {n: names.index(n) for n in names}
g = np.random.default_rng(1)
if variant != 'kn':
    raise SystemExit(f'unknown variant {variant}')
extra = np.column_stack([theta[:, J['log_kn']], np.ones(len(theta))]).astype(np.float32)
kw = dict(extra_noise=0.3, p_mask=0.3)
posts = []
for k in range(n_ens):
    p, _, _ = train_streaming(theta[~test], maps[~test], groups[~test], box, cfg['shape'], epochs=300, patience=20,
                              seed=k, log=lambda m: None, extra=extra[~test], **cfg['train_kw'], **kw)
    posts.append(p)
    print('member', k, 'trained', flush=True)
post = EnsemblePosterior(posts) if n_ens > 1 else posts[0]
idx = np.flatnonzero(test)
crk = maps[idx].reshape(len(idx), -1).sum(1) > 0
e = extra[idx].copy()
e[:, 0] += 0.3 * g.standard_normal(len(idx))
modes = {'map+kn': e, 'map': np.zeros_like(e)}
rep = {}
pw = 0.9 * (box.high - box.low)
for mode, ex in modes.items():
    smp = sample(post, box, maps[idx], n=1000, extra=ex)
    lo, hi = np.quantile(smp, 0.05, 1), np.quantile(smp, 0.95, 1)
    cov = (theta[idx] >= lo) & (theta[idx] <= hi)
    con = 1 - (hi - lo) / pw
    rep[mode] = {lab: dict(n=int(m.sum()), coverage90=dict(zip(names, map(float, cov[m].mean(0)))),
                           contraction90=dict(zip(names, map(float, con[m].mean(0)))))
                 for lab, m in (('cracked', crk), ('uncracked', ~crk))}
    np.savez_compressed(out.replace('.json', f'_{mode}.npz'), theta=theta[idx], q05=lo, q95=hi,
                        median=np.median(smp, 1), cracked=crk, sim=groups[idx])
rep['n_ensemble'] = n_ens
json.dump(rep, open(out, 'w'), indent=1)
print(json.dumps(rep, indent=1))

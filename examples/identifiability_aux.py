"""What does a crack map tell about the settlement, alone and combined with an independent estimate of a
nuisance parameter?

usage: python identifiability_aux.py NPE_OUT_DIR OUT.json [--sd-lnkn 0.3]
The amortized posterior was trained with the broad prior; an independent measurement m of ln k_n with
standard deviation sd (e.g. from a soil investigation) enters by importance weights
w = N(ln k_n; m, sd^2) on the posterior samples (the prior of ln k_n is uniform, so no prior ratio).
For held-out simulations the measurement is simulated as m = true + sd * N(0, 1), so coverage stays
interpretable. Reports coverage and contraction of the 90% intervals of S and x_p for maps with and
without cracks, with the effective sample size of the weights, and the reweighted posteriors of the
facade experiment with ln k_n measured around the calibrated 0.7 N/mm^3.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'scripts'))
from train_npe import CONFIGS  # noqa: E402


def wquantile(x, w, q):
    """Weighted quantiles of samples x (n,) with weights w (n,)."""
    o = np.argsort(x)
    c = np.cumsum(w[o])
    c = c / c[-1]
    return np.interp(q, c, x[o])


def summarize(smp, theta, cracked, names, box, weights=None):
    out = {}
    pw = 0.9 * (box.high - box.low)
    for name in ('S', 'x_p'):
        j = names.index(name)
        res = {}
        for lab, m in (('cracked', cracked), ('uncracked', ~cracked)):
            idx = np.flatnonzero(m)
            if len(idx) == 0:
                continue
            lo, hi, ess = [], [], []
            for i in idx:
                w = np.ones(smp.shape[1]) if weights is None else weights[i]
                a, b = wquantile(smp[i, :, j], w, [0.05, 0.95])
                lo.append(a), hi.append(b)
                ess.append(w.sum() ** 2 / (w ** 2).sum())
            lo, hi = np.array(lo), np.array(hi)
            t = theta[idx, j]
            res[lab] = dict(n=int(len(idx)), coverage90=float(((t >= lo) & (t <= hi)).mean()),
                            contraction90=float(1 - (hi - lo).mean() / pw[j]),
                            width90=float((hi - lo).mean()), ess_median=float(np.median(ess)))
        out[name] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npe_dir')
    ap.add_argument('out')
    ap.add_argument('--sd-lnkn', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    cfg = CONFIGS['facade']
    box, names = cfg['box'], cfg['box'].names
    z = np.load(os.path.join(a.npe_dir, 'heldout.npz'))
    smp = np.load(os.path.join(a.npe_dir, 'heldout_samples.npy')).astype(float)
    theta, cracked = z['theta'], z['cracked'].astype(bool)
    jk = names.index('log_kn')
    rng = np.random.default_rng(a.seed)
    m = theta[:, jk] + a.sd_lnkn * rng.standard_normal(len(theta))
    w = np.exp(-0.5 * ((smp[:, :, jk] - m[:, None]) / a.sd_lnkn) ** 2)
    rep = dict(sd_lnkn=a.sd_lnkn, map_only=summarize(smp, theta, cracked, names, box),
               map_and_kn=summarize(smp, theta, cracked, names, box, weights=w))
    # experiment: ln k_n measured around the calibrated value
    rep['experiment'] = {}
    for key, true in cfg['exp']:
        f = os.path.join(a.npe_dir, f'post_{key}.npy')
        if not os.path.exists(f):
            continue
        s = np.load(f)
        we = np.exp(-0.5 * ((s[:, jk] - true[jk]) / a.sd_lnkn) ** 2)
        r = {}
        for name in ('S', 'x_p'):
            j = names.index(name)
            q = np.quantile(s[:, j], [0.05, 0.5, 0.95])
            qw = wquantile(s[:, j], we, [0.05, 0.5, 0.95])
            r[name] = dict(true=true[j], map_only=[float(v) for v in q], map_and_kn=[float(v) for v in qw])
        r['ess'] = float(we.sum() ** 2 / (we ** 2).sum())
        rep['experiment'][key] = r
    json.dump(rep, open(a.out, 'w'), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()

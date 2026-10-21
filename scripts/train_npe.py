"""Train and evaluate the amortized neural posterior for crack-line maps (sbi, GPU).

usage: python train_npe.py CONFIG DATA_DIR OUT_DIR [--epochs 300] [--holdout 0.1] [--n-test 5000]
CONFIG: facade | den

Simulations are split by simulation (all stages of a simulation go to the same split). Writes the
trained estimator, held-out calibration (SBC ranks, coverage of central credible intervals, posterior
contraction, error of the posterior median) and the posteriors for the digitized experiments.

Facade: besides the crack map, the observation may contain the opening of the widest crack (scalar
input [log(w_max + 0.05 mm), 1]; [0, 0] when not measured). Simulations stored without width maps
enter with the width unmeasured. Held-out calibration and the experiments are evaluated with the map
only ('map') and, where a width is available, with map and width ('map+w').
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
from crackcause.npe import (Box, EnsemblePosterior, calibration, embed, ood_score, sample, train,  # noqa: E402
                            train_streaming)
from crackcause.observe_fx import load_exp_maps  # noqa: E402

LN = np.log
W0 = 0.05                    # mm, offset of the log width (hairline cracks)
# regularized streaming NPE (see crackcause.npe.train_streaming): small flow and summary MLP, parameter
# noise of 0.3 standard deviations, inspection-noise augmentation of 70% of the batches
REG = dict(z_noise=0.3, hidden=48, transforms=4, weight_decay=1e-3, p_clean=0.3)
CONFIGS = {
    'facade': dict(
        box=Box([0.5, 300.0, LN(30.0), LN(0.01), 0.06, LN(0.005), 0.6],
                [12.5, 1400.0, LN(400.0), LN(2.0), 0.16, LN(0.03), 1.4],
                ['S', 'x_p', 'log_w', 'log_kn', 'ft', 'log_Gf', 'q']),
        shape=(119, 143), stage_key='S', widths=True,
        train_kw=dict(REG, emb_kw=dict(hidden=64, out=32, dropout=0.3),
                      aug_kw=dict(shift=2, drop=0.15, block=4, spur=3e-4, p_dilate=0.0)),
        theta_of=lambda th: [th['x_p'], LN(th['w']), LN(th['kn']), th['ft'], LN(th['Gf']), th['q']],
        exp_file='data_public/giardina2013_fig5_exp_crackmaps.npz',
        exp_widths='data_public/giardina2013_fig7_crack_widths.csv',
        # S = settlement at x = 30 mm of the applied profiles; hinge fit of the profile (x_p, w);
        # interface stiffness and material values of Giardina et al. (2013); nominal stage A (mm)
        exp=[('S2.5', [2.26, 732.0, LN(98.0), LN(0.7), 0.1, LN(0.01), 1.0]),
             ('S5', [4.54, 732.0, LN(98.0), LN(0.7), 0.1, LN(0.01), 1.0]),
             ('S7.5', [6.81, 732.0, LN(98.0), LN(0.7), 0.1, LN(0.01), 1.0]),
             ('S10', [9.10, 732.0, LN(98.0), LN(0.7), 0.1, LN(0.01), 1.0])],
        exp_A={'S2.5': 2.5, 'S5': 5.0, 'S7.5': 7.5, 'S10': 10.0},
        # the widest measured opening is w_max only at 10 mm: crack 1, the widest then, was measured
        # from 9.5 mm onwards, so the maxima at 5 and 7.5 mm are lower bounds and are not used
        exp_width_keys=('S10',)),
    'den': dict(
        box=Box([0.0, 1.8, LN(0.06), 25.0], [16.0, 3.6, LN(0.16), 38.0], ['Ps', 'ft', 'log_Gf', 'E']),
        shape=(100, 100), stage_key=None, map_index=0,           # final state, threshold d >= 0.5
        train_kw=dict(REG, emb_kw=dict(bx=5, by=5, cy=10, hidden=64, out=32, dropout=0.3),
                      aug_kw=dict(shift=2, drop=0.15, block=3, spur=3e-4, p_dilate=0.0)),
        theta_of=lambda th: [th['Ps'], th['ft'], LN(th['Gf']), th['E']],
        exp_file='data_public/nooru1992_fig428_crackmaps.npz',
        exp=[(f'{c}_{face}', [ps, np.nan, np.nan, np.nan]) for c, ps in (('4a', 5.0), ('4b', 10.0), ('4c', 27.5))
             for face in ('front', 'rear')]),
}


def width_input(wmax):
    """Scalar observation [log(w_max + W0), 1] for a measured widest opening (mm), [0, 0] if unmeasured."""
    return [0.0, 0.0] if wmax is None or not np.isfinite(wmax) else [float(LN(wmax + W0)), 1.0]


def max_width(widths, lines):
    v = np.asarray(widths, np.float32)[np.asarray(lines) > 0]
    v = v[np.isfinite(v)]
    return float(v.max()) if len(v) else 0.0


def exp_max_widths(path, stages):
    """Widest measured crack opening at the nominal stages A (mm) from the digitized width curves; None
    where no crack was measured at that stage (curves are used within 0.3 mm of their first point)."""
    curves = {}
    for r in csv.DictReader(open(path)):
        curves.setdefault(int(r['crack']), []).append((float(r['A_mm']), float(r['width_mm'])))
    out = {}
    for key, A in stages.items():
        vals = []
        for pts in curves.values():
            a, w = np.array(pts).T
            if a.min() - 0.3 <= A <= a.max():
                vals.append(float(np.interp(A, a, w)))
        out[key] = max(vals) if vals else None
    return out


def load(data_dir, cfg):
    """(theta, maps, groups, extra): extra is None unless the config observes crack widths."""
    thetas, maps, groups, extra = [], [], [], []
    for k, f in enumerate(sorted(glob.glob(os.path.join(data_dir, 'sims', 'sim_*.npz')))):
        z = np.load(f)
        if len(z['maps']) == 0:
            continue
        base = cfg['theta_of'](json.loads(str(z['theta'])))
        if cfg['stage_key']:
            has_w = cfg.get('widths') and 'widths' in z.files and len(z['widths']) == len(z['maps'])
            wid = z['widths'] if has_w else None
            for j, (s, m) in enumerate(zip(z[cfg['stage_key']], z['maps'])):
                thetas.append([s] + base)
                maps.append(m)
                groups.append(k)
                extra.append(width_input(max_width(wid[j], m) if has_w else None))
        else:
            thetas.append(base)
            maps.append(z['maps'][cfg.get('map_index', -1)])
            groups.append(k)
    extra = np.array(extra, np.float32) if cfg.get('widths') else None
    return np.array(thetas), np.array(maps, np.uint8), np.array(groups), extra


def evaluate(post, box, theta, maps, extra, n=1000):
    """Posterior samples and calibration summaries for a set of held-out pairs."""
    smp = sample(post, box, maps, n=n, extra=extra)
    ranks, cov = calibration(smp, theta)
    med = np.median(smp, 1)
    width = np.quantile(smp, 0.95, 1) - np.quantile(smp, 0.05, 1)
    prior_width = 0.9 * (box.high - box.low)
    rep = dict(coverage={str(k): dict(zip(box.names, map(float, v))) for k, v in cov.items()},
               median_abs_err=dict(zip(box.names, map(float, np.abs(med - theta).mean(0)))),
               contraction_90=dict(zip(box.names, map(float, 1 - width.mean(0) / prior_width))))
    return smp, ranks, med, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config', choices=list(CONFIGS))
    ap.add_argument('data_dir')
    ap.add_argument('out_dir')
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--holdout', type=float, default=0.1)
    ap.add_argument('--n-test', type=int, default=5000)          # all held-out pairs
    ap.add_argument('--n-aug', type=int, default=4)
    ap.add_argument('--samples', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--trainer', choices=['streaming', 'sbi'], default='streaming')
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--embedding', choices=['features', 'cnn'], default='features')
    ap.add_argument('--ensemble', type=int, default=1, help='number of independently trained estimators')
    ap.add_argument('--n-train-sims', type=int, default=0,
                    help='use only this many training simulations (same held-out set; budget study)')
    ap.add_argument('--no-experiment', action='store_true', help='skip the experimental posteriors')
    a = ap.parse_args()
    cfg = CONFIGS[a.config]
    box, shape = cfg['box'], cfg['shape']
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    theta, maps, groups, extra = load(a.data_dir, cfg)
    sims = np.unique(groups)
    rng = np.random.default_rng(a.seed)
    test_sims = rng.choice(sims, size=max(1, int(a.holdout * len(sims))), replace=False)
    test = np.isin(groups, test_sims)
    if a.n_train_sims:                                 # budget study: subsample the training simulations
        train_sims = np.setdiff1d(sims, test_sims)
        keep = rng.choice(train_sims, size=min(a.n_train_sims, len(train_sims)), replace=False)
        use = test | np.isin(groups, keep)
        theta, maps, groups = theta[use], maps[use], groups[use]
        extra = None if extra is None else extra[use]
        test = test[use]
    n_w = 0 if extra is None else len(np.unique(groups[extra[:, 1] > 0.5]))
    print(f'{len(theta)} pairs from {len(sims)} simulations ({n_w} with crack widths; {test.sum()} pairs held out); '
          f'load {time.time() - t0:.0f}s', flush=True)
    if a.trainer == 'sbi':
        post, est, _ = train(theta[~test], maps[~test], box, shape, epochs=a.epochs, n_aug=a.n_aug, seed=a.seed)
    else:
        kw = dict(cfg.get('train_kw', {})) if a.embedding == 'features' else {}
        posts, ests = [], []
        for k in range(a.ensemble):                       # members differ in initialization, split, batches
            p_k, e_k, hist = train_streaming(theta[~test], maps[~test], groups[~test], box, shape, epochs=a.epochs,
                                             patience=a.patience, seed=a.seed + k, log=lambda m: print(m, flush=True),
                                             embedding=a.embedding,
                                             extra=None if extra is None else extra[~test], **kw)
            sfx = '' if k == 0 else f'_{k}'
            np.save(os.path.join(a.out_dir, f'loss_history{sfx}.npy'), np.array(hist))
            torch.save(e_k.state_dict(), os.path.join(a.out_dir, f'estimator_state{sfx}.pt'))
            posts.append(p_k), ests.append(e_k)
        post = posts[0] if a.ensemble == 1 else EnsemblePosterior(posts)
        est = ests[0]                                     # summary space of the misspecification score
    if a.trainer == 'sbi':
        torch.save(est.state_dict(), os.path.join(a.out_dir, 'estimator_state.pt'))
    print(f'trained in {time.time() - t0:.0f}s', flush=True)

    idx = rng.choice(np.flatnonzero(test), size=min(a.n_test, int(test.sum())), replace=False)
    cracked = maps[idx].reshape(len(idx), -1).sum(1) > 0
    rep = dict(n_pairs=int(len(theta)), n_sims=int(len(sims)), n_sims_with_widths=int(n_w), n_test=int(len(idx)),
               ensemble=int(a.ensemble))
    modes = {'map': None if extra is None else np.zeros((len(idx), 2), np.float32)}
    if extra is not None and (extra[idx, 1] > 0.5).any():
        modes['map+w'] = extra[idx]
    for mode, ex_in in modes.items():
        smp, ranks, med, r = evaluate(post, box, theta[idx], maps[idx], ex_in)
        tag = '' if mode == 'map' else '_w'
        rep['heldout' + tag] = r
        np.savez_compressed(os.path.join(a.out_dir, f'heldout{tag}.npz'), theta=theta[idx], median=med, ranks=ranks,
                            q05=np.quantile(smp, 0.05, 1), q95=np.quantile(smp, 0.95, 1), cracked=cracked,
                            sim=groups[idx], has_width=np.zeros(len(idx), bool) if ex_in is None else ex_in[:, 1] > 0.5)
        np.save(os.path.join(a.out_dir, f'heldout_samples{tag}.npy'), smp.astype(np.float32))   # (n_test, 1000, D)
    rep['seconds'] = time.time() - t0

    if cfg['exp_file'] and os.path.exists(os.path.join(ROOT, cfg['exp_file'])) and not a.no_experiment:
        ex = load_exp_maps(os.path.join(ROOT, cfg['exp_file']))     # thinned like the simulated maps
        keys = [k for k, _ in cfg['exp'] if k in ex]
        emaps = np.array([ex[k] for k in keys])
        wexp = (exp_max_widths(os.path.join(ROOT, cfg['exp_widths']), cfg['exp_A'])
                if extra is not None and cfg.get('exp_widths') else {})
        wexp = {k: (v if k in cfg.get('exp_width_keys', wexp) else None) for k, v in wexp.items()}
        rep['experiment_widths'] = wexp
        # misspecification check: distance of the observed maps to the simulated ones in summary space;
        # reference: cracked maps only (uncracked maps share one summary and have zero k-NN distance)
        crk = maps.reshape(len(maps), -1).sum(1) > 0
        pool = np.flatnonzero(~test & crk)
        tr_idx = rng.choice(pool, size=min(20000, len(pool)), replace=False)
        te_idx = np.flatnonzero(test & crk)
        variants = {'map': None if extra is None else np.zeros((len(keys), 2), np.float32)}
        if any(wexp.get(k) is not None for k in keys):
            variants['map+w'] = np.array([width_input(wexp.get(k)) for k in keys], np.float32)
        rep['experiment'], rep['ood_percentile'] = {}, {}
        for var, ex_in in variants.items():
            ref_extra = None if extra is None else (extra if var == 'map+w' else np.zeros_like(extra))
            f_tr = embed(est, maps[tr_idx], extra=None if ref_extra is None else ref_extra[tr_idx])
            f_te = embed(est, maps[te_idx], extra=None if ref_extra is None else ref_extra[te_idx])
            pct, _, _ = ood_score(f_tr, f_te, embed(est, emaps, extra=ex_in))
            smp_e = sample(post, box, emaps, n=a.samples, extra=ex_in)
            tag = '' if var == 'map' else '_w'
            for j, (k, s) in enumerate(zip(keys, smp_e)):
                if var == 'map+w' and ex_in[j, 1] < 0.5:
                    continue                                 # no width measured at this stage
                np.save(os.path.join(a.out_dir, f'post_{k}{tag}.npy'), s)
                true = dict(cfg['exp'])[k]
                rep['ood_percentile'][k + tag] = float(pct[j])
                rep['experiment'][k + tag] = {
                    n: dict(q05=float(np.quantile(s[:, i], 0.05)), median=float(np.median(s[:, i])),
                            q95=float(np.quantile(s[:, i], 0.95)), true=None if np.isnan(true[i]) else float(true[i]))
                    for i, n in enumerate(box.names)}
    json.dump(rep, open(os.path.join(a.out_dir, 'report.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != 'experiment'}, indent=1))
    for k, v in rep.get('experiment', {}).items():
        print(k, {n: (round(d['q05'], 3), round(d['median'], 3), round(d['q95'], 3), d['true']) for n, d in v.items()})


if __name__ == '__main__':
    main()

"""SBI data-generation worker for the Nooru-Mohamed DEN test, load-path 4 (WSL; one process per core).

usage: python fx_den_worker.py OUT_DIR --worker K --n-workers N --n-sims M [--ell 5 --h 2.5]
Simulation i draws (Ps, ft, Gf, E) from the prior with seed (base + i), applies the lateral shear at
zero axial force and then the axial elongation at constant Ps until delta_max or until equilibrium is
lost (the end of the test), and stores the phase field and crack-line maps (thresholds 0.5 and 0.9)
of the final state, plus the P-delta curve, in OUT_DIR/sims/sim_XXXXXX.npz.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
sys.path.insert(0, HERE)
from fx_den import DENModel  # noqa: E402
from crackcause import fx_den as D  # noqa: E402
from crackcause import campaign  # noqa: E402
from crackcause.randomfield import gaussian_field, lognormal_property  # noqa: E402

FT_COV, FT_CORR = 0.15, 10.0                       # lognormal strength field: CoV, correlation length (mm)


def sample_prior(rng):
    return dict(Ps=float(rng.uniform(0.0, 16.0)),                       # lateral shear (kN)
                ft=float(rng.uniform(1.8, 3.6)),                        # tensile strength (MPa)
                Gf=float(np.exp(rng.uniform(np.log(0.06), np.log(0.16)))),  # fracture energy (N/mm)
                E=float(rng.uniform(25.0, 38.0)))                       # Young's modulus (GPa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out_dir')
    ap.add_argument('--worker', type=int, required=True)
    ap.add_argument('--n-workers', type=int, required=True)
    ap.add_argument('--n-sims', type=int, required=True)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--seed', type=int, default=19920521)
    ap.add_argument('--ell', type=float, default=5.0)
    ap.add_argument('--h', type=float, default=2.5)
    ap.add_argument('--dmax', type=float, default=0.2)
    ap.add_argument('--no-claim', dest='claim', action='store_false')
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out_dir, 'sims'), exist_ok=True)
    t0 = time.time()
    den = DENModel(a.ell, a.h, eta_split=3e-3, eta_contact=1e-3, atol_u=5.0)
    if a.worker == 0:
        np.savez_compressed(os.path.join(a.out_dir, 'mesh.npz'), coords=den.coords, cells=den.cells,
                            dmg=den.cd['dmg'], spec=den.spec)
    print(f'worker {a.worker}: setup {time.time() - t0:.1f}s, {len(den.cells)} cells', flush=True)
    # indices are claimed through lock files (crackcause.campaign): workers fill all gaps in index
    # order; --worker/--n-workers only name the log (--no-claim restores the fixed residue split)
    order = (campaign.pending(a.out_dir, a.start, a.n_sims) if a.claim
             else range(a.start + a.worker, a.start + a.n_sims, a.n_workers))
    for i in order:
        path = os.path.join(a.out_dir, 'sims', f'sim_{i:06d}.npz')
        if os.path.exists(path):
            continue
        lock = campaign.claim(path) if a.claim else None
        if a.claim and lock is None:
            continue
        rng = np.random.default_rng(a.seed + i)
        th = sample_prior(rng)
        ts = time.time()
        zf = gaussian_field(0.0, D.L, 0.0, D.L, FT_CORR, 1.0, rng)(den.mid[:, [1, 0]])
        den.set_params(th['Ps'] * 1e3, ft=lognormal_property(th['ft'], FT_COV, zf), Gf=th['Gf'], E=th['E'] * 1e3)
        try:
            res = den.run(dmax=a.dmax, record_at=(), verbose=False, u_inner_it=8)
            status = res['status']
        except Exception as exc:                      # keep the worker alive
            res, status = dict(log=[], curve=[], records=[]), f'error: {exc!r}'
        d_fin = den.model.d.x.array.copy()
        ok = status in ('ok', 'failed') and len(res['curve']) > 0
        maps = np.array([den.crack_map(0.5), den.crack_map(0.9)], np.uint8) if ok else np.zeros((0, 100, 100), np.uint8)
        campaign.save_atomic(path, theta=json.dumps(th), maps=maps, thresholds=np.array([0.5, 0.9]),
                            d=d_fin.astype(np.float16), curve=np.array(res['curve']), log=json.dumps(res['log']),
                            status=status, seconds=time.time() - ts)
        campaign.release(lock)
        dl = res['curve'][-1][0] if len(res['curve']) else 0.0
        print(f'sim {i}: {status}, delta_end={dl * 1e3:.0f}um, {(time.time() - ts) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()

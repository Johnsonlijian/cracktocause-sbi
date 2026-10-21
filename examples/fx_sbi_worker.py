"""SBI data-generation worker for the facade (runs in WSL; one process per core).

usage: python fx_sbi_worker.py OUT_DIR --worker K --n-workers N --n-sims M [--ell 20 --h 10]
Simulation i (i = K, K+N, ...) draws theta_i from the prior with seed (base + i), runs the
settlement history to S_max and stores, per recorded stage, the crack-line map (observation
operator), the crack opening along the lines (crack_widths, float16 map), the nodal phase field
(float16) and the nodal displacements (float32), for re-applying other observation operators, in
OUT_DIR/sims/sim_XXXXXX.npz. Recorded stages are S = o + 0.5 k with a random offset o ~ U(0, 0.5)
per simulation, so that the settlement magnitude is covered continuously across the dataset.
The mesh and compiled forms are built once per worker and reused.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
from crackcause import giardina as G  # noqa: E402
from crackcause.fx_facade import base_facets, build_mesh, cell_data  # noqa: E402
from crackcause.fx_pfczm import FxPFCZM  # noqa: E402
from crackcause.observe_fx import PointInterpolator, Rasterizer, crack_lines, crack_widths  # noqa: E402
from crackcause import campaign  # noqa: E402
from crackcause.randomfield import gaussian_field, lognormal_property  # noqa: E402
from crackcause.settlement import hinge_profile  # noqa: E402

STAGE_STEP, N_STAGES = 0.5, 24                     # S = o + 0.5 k, k = 1..24, o ~ U(0, 0.5)
# simulator version stored with every result: '2026-10-12b' = step growth < 12 iterations, widths and
# displacements stored, residual of the accepted state logged per step ('res')
CODE_VERSION = '2026-10-12b'
FT_COV, FT_CORR = 0.15, 50.0                       # lognormal strength field: CoV, correlation length (mm)
IMG = (119, 143)                                   # 10 mm pixels over the facade


def sample_prior(rng):
    return dict(
        x_p=float(rng.uniform(300.0, 1400.0)),             # pivot of the settlement profile (mm)
        w=float(np.exp(rng.uniform(np.log(30.0), np.log(400.0)))),   # transition width (mm)
        kn=float(np.exp(rng.uniform(np.log(0.01), np.log(2.0)))),    # interface normal stiffness (N/mm^3)
        ft=float(rng.uniform(0.06, 0.16)),                  # masonry tensile strength (MPa)
        Gf=float(np.exp(rng.uniform(np.log(0.005), np.log(0.03)))),  # fracture energy (N/mm)
        q=float(rng.uniform(0.6, 1.4)),                      # scale of the vertical building loads
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out_dir')
    ap.add_argument('--worker', type=int, required=True)
    ap.add_argument('--n-workers', type=int, required=True)
    ap.add_argument('--n-sims', type=int, required=True)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--seed', type=int, default=20261011)
    ap.add_argument('--ell', type=float, default=20.0)
    ap.add_argument('--h', type=float, default=10.0)
    ap.add_argument('--relax', type=float, default=1.0)
    ap.add_argument('--smooth', type=float, default=1e-2)
    ap.add_argument('--eta-contact', type=float, default=5e-2)
    ap.add_argument('--prox', type=float, default=1e-4)
    ap.add_argument('--atol-u', type=float, default=6.0, help='absolute residual tolerance (N), 5e-4 of the load')
    ap.add_argument('--inner', type=int, default=12, help='cap on inner u Newton steps (0 = exact)')
    ap.add_argument('--no-claim', dest='claim', action='store_false')
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out_dir, 'sims'), exist_ok=True)
    t0 = time.time()
    msh = build_mesh(a.h)
    cd, mid = cell_data(msh)
    model = FxPFCZM(msh, cd, a.ell, G.T, softening='linear', base_facets=base_facets(msh),
                    kn=G.INTERFACE['kn'], kt=G.INTERFACE['kt'], eta_split=a.smooth, eta_contact=a.eta_contact, prox=a.prox, atol_u=a.atol_u)
    coords = model.Q.tabulate_dof_coordinates()[:, :2]
    cells = model.Q.dofmap.list.copy()
    ras = Rasterizer(coords, cells[cd['dmg'] > 0.5], (0.0, G.W, 0.0, G.H), IMG)
    interp = PointInterpolator(coords, cells)                  # whole wall, lintels included
    # displacement blocks must follow the node order of the scalar space (checked once)
    xv = model.V.tabulate_dof_coordinates()[:, :2]
    assert xv.shape == coords.shape and np.allclose(xv, coords), 'V and Q dof orders differ'
    if a.worker == 0:
        np.savez_compressed(os.path.join(a.out_dir, 'mesh.npz'), coords=coords, cells=cells, dmg=cd['dmg'])
    print(f'worker {a.worker}: setup {time.time() - t0:.1f}s, {len(cells)} cells', flush=True)
    masonry = cd['dmg'] > 0.5
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
        offset = float(rng.uniform(0.0, STAGE_STEP))
        stages = list(np.round(offset + STAGE_STEP * np.arange(1, N_STAGES + 1), 6))
        ts = time.time()
        z = gaussian_field(0.0, G.W, 0.0, G.H, FT_CORR, 5.0, rng)(mid[:, [1, 0]])
        ft = np.where(masonry, lognormal_property(th['ft'], FT_COV, z), 1e3)
        Gf = np.full_like(ft, th['Gf'])
        model.set_materials(ft=ft, Gf=Gf, body_scale=th['q'], kn=th['kn'])
        model.reset()
        phi = hinge_profile(coords[:, 0], th['x_p'], th['w'])
        maps, dsn, wid, us = [], [], [], []

        def record(S, m):
            img = ras(m.d.x.array)
            lines = crack_lines(img, ras.inside, threshold=0.9)
            u = m.u.x.array.reshape(-1, 2)
            maps.append(lines)
            wid.append(crack_widths(lines, ras, interp, u).astype(np.float16))
            dsn.append(m.d.x.array.astype(np.float16))
            us.append(u.astype(np.float32))

        try:
            records, log = model.run_settlement(phi, stages[-1], record_at=stages, relax=a.relax,
                                                callback=record, verbose=False, keep_work=True,
                                                u_inner_it=a.inner or None)
            status = ('failed' if any(r.get('failed', False) for r in log)
                      else 'unconverged' if any(r.get('unconverged', False) for r in log) else 'ok')
        except Exception as exc:                       # keep the worker alive
            records, log, status = [], [], f'error: {exc!r}'
        S_rec = np.array([r['S'] for r in records])
        campaign.save_atomic(path, theta=json.dumps(th), S=S_rec, offset=offset, ft_cov=FT_COV, ft_corr=FT_CORR,
                            maps=np.array(maps, np.uint8) if maps else np.zeros((0,) + IMG, np.uint8),
                            d=np.array(dsn) if dsn else np.zeros((0, len(coords)), np.float16),
                            widths=np.array(wid) if wid else np.zeros((0,) + IMG, np.float16),
                            u=np.array(us) if us else np.zeros((0, len(coords), 2), np.float32),
                            log=json.dumps(log), status=status, seconds=time.time() - ts, version=CODE_VERSION)
        campaign.release(lock)
        print(f'sim {i}: {status}, {len(S_rec)} stages, S_last={S_rec[-1] if len(S_rec) else 0:.2f}, '
              f'{(time.time() - ts) / 60:.1f} min, unconverged {sum(1 for r in log if not r.get("ok", True))}',
              flush=True)


if __name__ == '__main__':
    main()

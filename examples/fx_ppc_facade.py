"""Posterior predictive check for the façade: re-simulate posterior samples, record crack maps at the
experimental stages (WSL; one process per worker).

usage: python fx_ppc_facade.py SAMPLES.npy OUT_DIR --worker K --n-workers N [--n 20]
SAMPLES.npy: posterior samples (M, 7) in the order S, x_p, log_w, log_kn, ft, log_Gf, q (from
scripts/train_npe.py). Every sample is run through the experimental stages S = 2.26, 4.54, 6.81,
9.10 mm and its own settlement magnitude; crack maps and the widest crack opening are recorded at
all of them (the predictive distribution for an observation uses the record at the sampled S).
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
from crackcause.randomfield import gaussian_field, lognormal_property  # noqa: E402
from crackcause.settlement import hinge_profile  # noqa: E402

FT_COV, FT_CORR = 0.15, 50.0                       # same strength field as the training simulations

ap = argparse.ArgumentParser()
ap.add_argument('samples')
ap.add_argument('out_dir')
ap.add_argument('--worker', type=int, required=True)
ap.add_argument('--n-workers', type=int, required=True)
ap.add_argument('--n', type=int, default=20)
ap.add_argument('--seed', type=int, default=7)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
smp = np.load(a.samples)
rng = np.random.default_rng(a.seed)
pick = smp[rng.choice(len(smp), size=a.n, replace=False)]
msh = build_mesh(10.0)
cd, mid = cell_data(msh)
model = FxPFCZM(msh, cd, 20.0, G.T, softening='linear', base_facets=base_facets(msh), kn=G.INTERFACE['kn'],
                kt=G.INTERFACE['kt'], eta_split=1e-2, eta_contact=5e-2, prox=1e-4, atol_u=6.0)
coords = model.Q.tabulate_dof_coordinates()[:, :2]
cells = model.Q.dofmap.list.copy()
masonry = cd['dmg'] > 0.5
ras = Rasterizer(coords, cells[masonry], (0.0, G.W, 0.0, G.H), (119, 143))
interp = PointInterpolator(coords, cells)
for i in range(a.worker, a.n, a.n_workers):
    path = os.path.join(a.out_dir, f'ppc_{i:03d}.npz')
    if os.path.exists(path):
        continue
    S_s, x_p, lw, lkn, ft, lgf, q = pick[i]
    stages = sorted(set(round(float(s), 3) for s in G.EXP_STAGES_S) | {round(float(S_s), 3)})
    th = dict(x_p=float(x_p), w=float(np.exp(lw)), kn=float(np.exp(lkn)), ft=float(ft), Gf=float(np.exp(lgf)), q=float(q))
    ts = time.time()
    zf = gaussian_field(0.0, G.W, 0.0, G.H, FT_CORR, 5.0, np.random.default_rng(a.seed * 1000 + i))(mid[:, [1, 0]])
    model.set_materials(ft=np.where(masonry, lognormal_property(th['ft'], FT_COV, zf), 1e3),
                        Gf=np.full(len(masonry), th['Gf']), body_scale=th['q'], kn=th['kn'])
    model.reset()
    phi = hinge_profile(coords[:, 0], th['x_p'], th['w'])
    maps, wmax = [], []

    def record(S, m):
        lines = crack_lines(ras(m.d.x.array), ras.inside)
        w = crack_widths(lines, ras, interp, m.u.x.array.reshape(-1, 2))[lines > 0]
        maps.append(lines)
        wmax.append(float(np.nanmax(w)) if np.isfinite(w).any() else 0.0)

    records, log = model.run_settlement(phi, stages[-1], record_at=stages, verbose=False, keep_work=True, u_inner_it=12,
                                        callback=record)
    np.savez_compressed(path, theta=json.dumps(th), S_sample=float(S_s), S=np.array([r['S'] for r in records]),
                        maps=np.array(maps, np.uint8), wmax=np.array(wmax),
                        status='failed' if any(r.get('failed', False) for r in log) else 'ok', seconds=time.time() - ts)
    print(f'ppc {i}: {len(records)} stages, {(time.time() - ts) / 60:.1f} min', flush=True)

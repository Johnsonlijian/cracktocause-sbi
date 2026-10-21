"""Diagnose the displacement Newton solve during unstable crack growth on the reference facade.

usage: python diag_newton.py OUT_DIR [--s-hard 2.25] [--iters 40]
Runs the reference settlement history up to the last converged step before S_hard, then performs
staggered iterations at S_hard with a monitored displacement solve (residual history, line-search
step lengths) and records where the residual is concentrated (contact springs vs. damaged elements).
"""
import argparse
import os
import sys
import time

import numpy as np
from petsc4py import PETSc

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
from crackcause import giardina as G  # noqa: E402
from crackcause.fx_facade import base_facets, build_mesh, cell_data  # noqa: E402
from crackcause.fx_pfczm import FxPFCZM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('out_dir')
ap.add_argument('--s-hard', type=float, default=2.25)
ap.add_argument('--iters', type=int, default=40)
ap.add_argument('--smooth', type=float, default=1e-2)
ap.add_argument('--eta-contact', type=float, default=5e-2)
ap.add_argument('--prox', type=float, default=1e-4)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
msh = build_mesh(10.0)
cd, _ = cell_data(msh)
model = FxPFCZM(msh, cd, 20.0, G.T, softening='linear', base_facets=base_facets(msh), kn=G.INTERFACE['kn'],
                kt=G.INTERFACE['kt'], eta_split=a.smooth, eta_contact=a.eta_contact, prox=a.prox, atol_u=1.2)
xs, ps = G.settlement_shape()
phi = np.interp(model.Q.tabulate_dof_coordinates()[:, 0], xs, ps)
t0 = time.time()
records, log = model.run_settlement(phi, 1.875, record_at=[1.875], keep_work=True, u_inner_it=12, verbose=True)
print(f'reached S=1.875 in {time.time() - t0:.0f}s', flush=True)
model.g.x.array[:] = a.s_hard * phi
snes = model.problem_u.solver
hist = []
snes.setMonitor(lambda s, it, fn: hist.append((it, fn)))
opts = PETSc.Options()
opts['u_snes_linesearch_monitor'] = None
snes.setFromOptions()
V = model.V
xv = V.tabulate_dof_coordinates()[:, :2]
base = np.flatnonzero(np.isclose(xv[:, 1], 0.0))
for it in range(1, a.iters + 1):
    x = model.d.x.array.copy()
    model.u_prox.x.array[:] = model.u.x.array
    hist.clear()
    ru, nit = model.solve_u(max_it=30, accept_max_it=True)
    # residual distribution of the current iterate
    b = model._Fu_form
    from dolfinx import fem, la
    r = fem.assemble_vector(b)
    r.scatter_reverse(la.InsertMode.add)
    ra = r.array.reshape(-1, 2)
    rn = np.linalg.norm(ra, axis=1)
    top = np.argsort(rn)[-5:][::-1]
    model._update_history()
    model.solve_d()
    err = np.abs(model.d.x.array - x).max()
    print(f'it {it}: reason {ru} newton {nit}, |F| ' + ' '.join(f'{fn:.2e}' for _, fn in hist[:12])
          + (' ...' if len(hist) > 12 else '') + f'; err {err:.2e}; dmax {model.d.x.array.max():.3f}', flush=True)
    print('     largest nodal residuals (x, y, |r|, base?): ' + '; '.join(
        f'({xv[k, 0]:.0f},{xv[k, 1]:.0f}) {rn[k]:.1f} {"B" if k in base else ""}' for k in top), flush=True)

"""Benchmark of staggered-solver variants on the Giardina facade (WSL).

usage: python bench_staggered.py OUT_DIR --variant NAME [--smax 4] [--ell 20 --h 10]
variants: plain (restart halving), keep (keep work), aa3/aa5 (Anderson + keep work),
          relax1.5 (over-relaxation + keep work), smooth* (smoothed split/contact, inexact inner u)
Writes OUT_DIR/bench_<variant>.npz with the phase field at S = 1, 2, 3, 4 and the step log.
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

SMOOTH = dict(eta_split=1e-3, eta_contact=1e-3)
VARIANTS = {'plain': ({}, dict()), 'keep': ({}, dict(keep_work=True)),
            'aa3': ({}, dict(anderson=3, keep_work=True)), 'aa5': ({}, dict(anderson=5, keep_work=True)),
            'relax1.5': ({}, dict(relax=1.5, keep_work=True)),
            'smooth': (SMOOTH, dict(keep_work=True)),
            'smooth_in6': (SMOOTH, dict(keep_work=True, u_inner_it=6)),
            'smooth_in6_r15': (SMOOTH, dict(keep_work=True, u_inner_it=6, relax=1.5)),
            'c01': (dict(eta_split=1e-3, eta_contact=1e-2), dict(keep_work=True)),
            'c01_in6': (dict(eta_split=1e-3, eta_contact=1e-2), dict(keep_work=True, u_inner_it=6)),
            'c01_tr': (dict(eta_split=1e-3, eta_contact=1e-2, u_snes_type='newtontr'), dict(keep_work=True)),
            'c01p': (dict(eta_split=1e-3, eta_contact=1e-2, prox=1e-4), dict(keep_work=True)),
            'c01p_in6': (dict(eta_split=1e-3, eta_contact=1e-2, prox=1e-4), dict(keep_work=True, u_inner_it=6)),
            'c01pa_in6': (dict(eta_split=1e-3, eta_contact=1e-2, prox=1e-4, atol_u=1.2), dict(keep_work=True, u_inner_it=6))}

ap = argparse.ArgumentParser()
ap.add_argument('out_dir')
ap.add_argument('--variant', required=True, choices=list(VARIANTS))
ap.add_argument('--smax', type=float, default=4.0)
ap.add_argument('--ell', type=float, default=20.0)
ap.add_argument('--h', type=float, default=10.0)
ap.add_argument('--debug', action='store_true')
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
msh = build_mesh(a.h)
cd, _ = cell_data(msh)
ctor, run = VARIANTS[a.variant]
model = FxPFCZM(msh, cd, a.ell, G.T, softening='linear', base_facets=base_facets(msh),
                kn=G.INTERFACE['kn'], kt=G.INTERFACE['kt'], **ctor)
model.debug = a.debug
coords = model.Q.tabulate_dof_coordinates()[:, :2]
xs, ps = G.settlement_shape()
phi = np.interp(coords[:, 0], xs, ps)
rec_S = [r for r in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0) if r <= a.smax + 1e-9]
t0 = time.time()
records, log = model.run_settlement(phi, a.smax, record_at=rec_S, **run)
sec = time.time() - t0
its = sum(r['it'] for r in log)
print(f'variant {a.variant}: {sec:.0f}s, {its} staggered iterations, {len(log)} steps', flush=True)
np.savez_compressed(os.path.join(a.out_dir, f'bench_{a.variant}.npz'), S=np.array([r['S'] for r in records]),
                    d=np.array([r['d'] for r in records]), coords=coords, log=json.dumps(log), seconds=sec)

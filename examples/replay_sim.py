"""Replay one simulation of the façade SBI campaign exactly (same prior draw, stage offset and strength
field as fx_sbi_worker.py) in debug mode, to diagnose failures.

usage: python replay_sim.py INDEX [--seed 20261011] [--smax S]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
sys.path.insert(0, HERE)
from crackcause import giardina as G  # noqa: E402
from crackcause.fx_facade import base_facets, build_mesh, cell_data  # noqa: E402
from crackcause.fx_pfczm import FxPFCZM  # noqa: E402
from crackcause.randomfield import gaussian_field, lognormal_property  # noqa: E402
from crackcause.settlement import hinge_profile  # noqa: E402
from fx_sbi_worker import FT_CORR, FT_COV, N_STAGES, STAGE_STEP, sample_prior  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('index', type=int)
ap.add_argument('--seed', type=int, default=20261011)
ap.add_argument('--smax', type=float, default=None)
a = ap.parse_args()
msh = build_mesh(10.0)
cd, mid = cell_data(msh)
model = FxPFCZM(msh, cd, 20.0, G.T, softening='linear', base_facets=base_facets(msh), kn=G.INTERFACE['kn'],
                kt=G.INTERFACE['kt'], eta_split=1e-2, eta_contact=5e-2, prox=1e-4, atol_u=6.0)
model.debug = True
coords = model.Q.tabulate_dof_coordinates()[:, :2]
masonry = cd['dmg'] > 0.5
rng = np.random.default_rng(a.seed + a.index)
th = sample_prior(rng)
offset = float(rng.uniform(0.0, STAGE_STEP))
stages = list(np.round(offset + STAGE_STEP * np.arange(1, N_STAGES + 1), 6))
z = gaussian_field(0.0, G.W, 0.0, G.H, FT_CORR, 5.0, rng)(mid[:, [1, 0]])
ft = np.where(masonry, lognormal_property(th['ft'], FT_COV, z), 1e3)
model.set_materials(ft=ft, Gf=np.full_like(ft, th['Gf']), body_scale=th['q'], kn=th['kn'])
model.reset()
print('theta', th, 'offset', offset, flush=True)
phi = hinge_profile(coords[:, 0], th['x_p'], th['w'])
smax = a.smax or stages[-1]
records, log = model.run_settlement(phi, smax, record_at=[s for s in stages if s <= smax], keep_work=True,
                                    u_inner_it=12, verbose=True)
print('done', len(records), 'records; failed:', any(r.get('failed', False) for r in log), flush=True)

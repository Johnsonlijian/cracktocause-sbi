"""Forward-model demo: two-storey masonry facade on a soft interface under a Gaussian settlement trough.

usage: python facade_settlement_demo.py SCENARIO OUT_DIR [--h 50] [--seed 1] [--no-gravity]
                                        [--smax 25] [--split spectral|none] [--interface notension|elastic]
scenarios: sagging | hogging | offset   (see SCENARIOS below)

Units: mm, N, MPa. Homogenized isotropic masonry, PF-CZM (linear softening), lognormal
random tensile-strength field. Settlement (vertical + horizontal, Peck / O'Reilly-New type)
is prescribed at the bottom of a non-damageable interface layer, which is either linear
elastic or a no-tension material that lets the facade lift off (gap) in hogging zones.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
from crackcause.mesh import nodes_where, structured_tri_mesh  # noqa: E402
from crackcause.pfczm import PFCZM  # noqa: E402
from crackcause.randomfield import gaussian_field, lognormal_property  # noqa: E402

SCENARIOS = {                       # trough centre x0, width parameter i, depth z0 (mm)
    'sagging': dict(x0=5000.0, i=4000.0, z0=15000.0),
    'hogging': dict(x0=-3000.0, i=5000.0, z0=15000.0),
    'offset': dict(x0=9000.0, i=3000.0, z0=12000.0),
}
LX, LY, T_WALL, T_INT = 10000.0, 6000.0, 200.0, 100.0
OPENINGS = [                        # (x_lo, x_hi, y_lo, y_hi): door + windows
    (1500.0, 2500.0, 0.0, 2200.0),
    (4400.0, 5600.0, 900.0, 2100.0), (7400.0, 8600.0, 900.0, 2100.0),
    (1400.0, 2600.0, 3700.0, 4900.0), (4400.0, 5600.0, 3700.0, 4900.0), (7400.0, 8600.0, 3700.0, 4900.0),
]
MAT = dict(E=3000.0, nu=0.2, ft=0.15, ft_cov=0.15, ft_corr=600.0, Gf=0.01, ell=150.0,
           E_int=10.0, rho_g=1.8e-5, S_max=25.0)


def build(h, seed, split, interface):
    xs = np.linspace(0.0, LX, int(round(LX / h)) + 1)
    ys = np.concatenate([np.linspace(-T_INT, 0.0, 3)[:-1], np.linspace(0.0, LY, int(round(LY / h)) + 1)])

    def keep(xc, yc):
        k = np.ones_like(xc, bool)
        for xl, xh, yl, yh in OPENINGS:
            k &= ~((xc > xl) & (xc < xh) & (yc > yl) & (yc < yh))
        return k

    nodes, tris = structured_tri_mesh(xs, ys, keep_cell=keep, diag='alternate')
    cen = nodes[tris].mean(axis=1)
    wall = cen[:, 1] > 0.0
    rng = np.random.default_rng(seed)
    z = gaussian_field(0.0, LX, -T_INT, LY, MAT['ft_corr'], h, rng)(cen[:, ::-1])
    ft = np.where(wall, lognormal_property(MAT['ft'], MAT['ft_cov'], z), 1e6)
    E = np.where(wall, MAT['E'], MAT['E_int'])
    model = PFCZM(nodes, tris, E, MAT['nu'], ft, MAT['Gf'], MAT['ell'], thickness=T_WALL,
                  plane='stress', softening='linear', damageable=wall, split=split,
                  no_tension=(~wall) if interface == 'notension' else None,
                  k_res=1e-6)
    return model, nodes, tris, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('scenario', choices=sorted(SCENARIOS))
    ap.add_argument('out_dir')
    ap.add_argument('--h', type=float, default=50.0)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--no-gravity', action='store_true')
    ap.add_argument('--smax', type=float, default=MAT['S_max'])
    ap.add_argument('--split', default='spectral', choices=['spectral', 'none'])
    ap.add_argument('--interface', default='notension', choices=['notension', 'elastic'])
    ap.add_argument('--e-int', type=float, default=MAT['E_int'], help='interface Young modulus (MPa)')
    ap.add_argument('--floor-load', type=float, default=0.0,
                    help='line load per floor (N/mm) at y = 3000 and y = 6000 (floors + roof)')
    ap.add_argument('--tag', default=None, help='output file stem (default: scenario name)')
    ap.add_argument('--relax', type=float, default=1.6, help='over-relaxation of the staggered scheme')
    ap.add_argument('--newton', type=int, default=2, help='Newton updates of u per staggered iteration')
    ap.add_argument('--tol-u', type=float, default=1e-6, help='relative equilibrium tolerance')
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--tol', type=float, default=1e-3)
    ap.add_argument('--max-stag', type=int, default=300)
    a = ap.parse_args()
    MAT['S_max'] = a.smax
    MAT['E_int'] = a.e_int
    MAT['floor_load'] = a.floor_load
    os.makedirs(a.out_dir, exist_ok=True)
    p = SCENARIOS[a.scenario]
    t0 = time.time()
    model, nodes, tris, wall = build(a.h, a.seed, a.split, a.interface)
    base = nodes_where(nodes, lambda x, y: np.isclose(y, -T_INT))
    model.set_dirichlet(np.concatenate([2 * base, 2 * base + 1]))
    xb = nodes[base, 0]
    S = MAT['S_max'] * np.exp(-(xb - p['x0']) ** 2 / (2.0 * p['i'] ** 2))
    ux, uy = -(xb - p['x0']) / p['z0'] * S, -S
    order = np.argsort(np.concatenate([2 * base, 2 * base + 1]))
    vals = np.concatenate([ux, uy])[order]           # aligned with model.fixed (sorted)

    f_g = np.zeros(model.ndof)                       # self-weight of the wall as nodal forces
    w_e = MAT['rho_g'] * T_WALL * model.A * wall
    np.add.at(f_g, 2 * tris + 1, -np.repeat(w_e[:, None] / 3.0, 3, axis=1))
    if a.floor_load > 0.0:                           # floor/roof line loads, lumped to nodes on each level
        for y_level in (3000.0, LY):
            lv = nodes_where(nodes, lambda x, y: np.isclose(y, y_level))
            xs_lv = np.sort(nodes[lv, 0])
            trib = np.gradient(xs_lv) if xs_lv.size > 1 else np.ones(1)
            f_g[2 * lv[np.argsort(nodes[lv, 0])] + 1] -= a.floor_load * trib

    if not a.no_gravity:                             # first step: self-weight only (lam = 0)
        lams = np.concatenate([[0.0], np.linspace(0.0, 1.0, a.steps + 1)[1:]])
    else:
        lams = np.linspace(0.0, 1.0, a.steps + 1)[1:]
    snap_every = max(1, a.steps // 10)
    res = model.run(lams, lambda lam: lam * vals,
                    f_ext_fun=None if a.no_gravity else (lambda lam: f_g),
                    tol=a.tol, max_stag=a.max_stag, relax=a.relax,
                    newton_per_stag=a.newton, tol_u=a.tol_u,
                    snapshot_at=range(snap_every - 1, len(lams), snap_every), verbose=True)
    tag = a.tag or a.scenario
    np.savez_compressed(os.path.join(a.out_dir, f'{tag}.npz'), nodes=nodes, tris=tris, wall=wall,
                        d=res['d'], u=res['u'], snap_lams=np.array([s[0] for s in res['snaps']]),
                        snaps=np.array([s[1] for s in res['snaps']]), **{k: v for k, v in res['hist'].items()})
    info = dict(scenario=a.scenario, params=p, mat=MAT, h=a.h, seed=a.seed, gravity=not a.no_gravity,
                split=a.split, interface=a.interface, nodes=int(model.N), elements=int(model.M),
                seconds=time.time() - t0, total_stag=int(res['hist']['n_stag'].sum()),
                max_stag_hits=int((res['hist']['n_stag'] >= a.max_stag).sum()))
    with open(os.path.join(a.out_dir, f'{tag}.json'), 'w') as fh:
        json.dump(info, fh, indent=1)
    print(json.dumps(info))


if __name__ == '__main__':
    main()

"""Verification of the production PF-CZM solver on a bar in uniaxial tension (WSL / FEniCSx).

usage: python fx_verify_bar.py OUT_DIR [--ells 5 10 20]
Strip 200 x 10 mm (plane stress, t = 1), left end fixed, right end pulled. The strength is reduced by
5% over a central zone of width ell, so that the band localizes there before diffuse damage can
spread along the uniformly stressed bar (with a 1% imperfection the bar stretches at ~f_t with
diffuse damage, which inflates the dissipated energy by 17-32%). For each length scale the traction-separation
response sigma(w), w = U - sigma L / E, is compared with the linear softening law, the dissipated
energy with G_f, and the band-centre phase field d* with the closed forms sigma = f_t (1 - d*) and
w = d* w_c.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
from crackcause.fx_pfczm import FxPFCZM  # noqa: E402

L, H = 200.0, 10.0
E, NU, FT, GF = 3000.0, 0.2, 0.1, 0.01
WEAK = 0.95


def run(ell, h):
    from dolfinx import fem, mesh as dmesh
    from dolfinx.mesh import locate_entities_boundary
    from mpi4py import MPI
    from petsc4py import PETSc
    msh = dmesh.create_rectangle(MPI.COMM_WORLD, [np.array([0.0, 0.0]), np.array([L, H])],
                                 [int(round(L / h)), max(2, int(round(H / h)))], dmesh.CellType.triangle,
                                 diagonal=dmesh.DiagonalType.crossed)
    n = msh.topology.index_map(2).size_local
    mid = dmesh.compute_midpoints(msh, 2, np.arange(n, dtype=np.int32))
    weak = np.abs(mid[:, 0] - L / 2) < 0.5 * ell
    cd = dict(E=np.full(n, E), nu=np.full(n, NU), ft=np.where(weak, WEAK * FT, FT), Gf=np.full(n, GF),
              rho_g=np.zeros(n), dmg=np.ones(n), fx=np.zeros(n), fy=np.zeros(n))
    U = fem.Constant(msh, PETSc.ScalarType(0.0))
    store = {}

    def bcs(V):
        fdim = 1
        left = locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
        right = locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], L))
        dl = fem.locate_dofs_topological(V.sub(0), fdim, left)
        dr = fem.locate_dofs_topological(V.sub(0), fdim, right)
        V1, _ = V.sub(1).collapse()
        dc = fem.locate_dofs_geometrical((V.sub(1), V1), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
        store['right_x'] = dr
        return [fem.dirichletbc(PETSc.ScalarType(0.0), dl, V.sub(0)), fem.dirichletbc(U, dr, V.sub(0)),
                fem.dirichletbc(fem.Function(V1), dc, V.sub(1))]

    model = FxPFCZM(msh, cd, ell, 1.0, softening='linear', bcs_u=bcs, eta_split=1e-2 * FT, atol_u=1e-8)
    model.c_prox.value = 0.0
    xq = model.Q.tabulate_dof_coordinates()[:, :2]
    centre = np.abs(xq[:, 0] - L / 2) < 0.6 * h
    out = []
    wc = 2 * GF / (WEAK * FT)
    Us = np.concatenate([np.linspace(0, 1.2 * FT * L / E, 13)[1:], FT * L / E + np.linspace(0.005, 1.05 * wc, 60)])
    for Uv in Us:
        U.value = Uv
        ok, it = model.staggered(1e-4, 500)
        model.commit()
        P = -model.reaction(store['right_x']) / H
        P = abs(P)
        out.append(dict(U=float(Uv), sigma=float(P), w=float(Uv - P * L / E), dstar=float(model.d.x.array[centre].max()),
                        ok=bool(ok), it=int(it)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out_dir')
    ap.add_argument('--ells', type=float, nargs='+', default=[5.0, 10.0, 20.0])
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    res = {}
    for ell in a.ells:
        r = run(ell, ell / 5.0)
        json.dump(r, open(os.path.join(a.out_dir, f'bar_steps_l{ell:g}.json'), 'w'))
        s = np.array([x['sigma'] for x in r])
        w = np.array([x['w'] for x in r])
        ds = np.array([x['dstar'] for x in r])
        Uv = np.concatenate([[0.0], [x['U'] for x in r]])
        Gd = float(np.trapezoid(np.concatenate([[0.0], s]), Uv) - 0.5 * s[-1] ** 2 * L / E)   # energy per area
        ftw, wcw = WEAK * FT, 2 * GF / (WEAK * FT)
        post = w > 0.01 * wcw
        res[ell] = dict(peak=float(s.max() / ftw), energy=float(Gd / GF),
                        sigma_vs_dstar=float(np.abs(s[post] / ftw - (1 - ds[post])).max()),
                        w_vs_dstar=float(np.abs(w[post] / wcw - ds[post]).max()), steps=r)
        print(f'ell={ell}: peak/ft_weak={res[ell]["peak"]:.3f} G/Gf={res[ell]["energy"]:.3f} '
              f'max|sigma/ft-(1-d*)|={res[ell]["sigma_vs_dstar"]:.3f} max|w/wc-d*|={res[ell]["w_vs_dstar"]:.3f}',
              flush=True)
    json.dump(res, open(os.path.join(a.out_dir, 'verify_bar_fx.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()

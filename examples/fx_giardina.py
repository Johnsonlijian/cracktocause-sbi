"""Reference simulation of the Giardina et al. masonry facade test with FEniCSx PF-CZM.

usage (WSL env):  python fx_giardina.py OUT_DIR [--ell 20] [--h 5] [--smax 11.5] [--softening linear]
                  [--cov 0.15 --seed 3] [--profile hinge --xp 900 --w 98] [--geometry solid|piers60]
Writes OUT_DIR/giardina_l{ell}_h{h}_kn{kn}_{softening}[...].npz with node coordinates, cells, phase field and
displacements at the record settlements and the step log (the tag names every non-default option).
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
from crackcause.settlement import hinge_profile  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out_dir')
    ap.add_argument('--ell', type=float, default=20.0)
    ap.add_argument('--h', type=float, default=None)
    ap.add_argument('--smax', type=float, default=11.5)
    ap.add_argument('--softening', default='linear')
    ap.add_argument('--kn', type=float, default=G.INTERFACE['kn'])
    ap.add_argument('--relax', type=float, default=1.0)
    ap.add_argument('--tol', type=float, default=3e-3)
    ap.add_argument('--smooth', type=float, default=1e-2,
                    help='regularization of the spectral split (MPa) and contact (mm); 0 = sharp (1e-6/1e-5)')
    ap.add_argument('--eta-contact', type=float, default=5e-2, help='contact ramp length (mm)')
    ap.add_argument('--snes', default='newtonls', help='SNES type of the displacement solve')
    ap.add_argument('--prox', type=float, default=1e-4, help='proximal coefficient of the u-solve (N/mm^4)')
    ap.add_argument('--atol-u', type=float, default=6.0, help='absolute residual tolerance (N), 5e-4 of the load')
    ap.add_argument('--inner', type=int, default=12, help='cap on inner u Newton steps (0 = exact)')
    ap.add_argument('--cov', type=float, default=0.0, help='CoV of the lognormal strength field (0 = homogeneous)')
    ap.add_argument('--corr', type=float, default=50.0, help='correlation length of the strength field (mm)')
    ap.add_argument('--seed', type=int, default=0, help='seed of the strength field realization')
    ap.add_argument('--profile', choices=['measured', 'measured_nouplift', 'hinge'], default='measured',
                    help='settlement shape: measured (Fig. 8), measured with the uplift removed, or the hinge fit')
    ap.add_argument('--xp', type=float, default=732.0, help='pivot of the hinge fit (mm)')
    ap.add_argument('--w', type=float, default=98.0, help='transition width of the hinge fit (mm)')
    ap.add_argument('--geometry', default='test',
                    help="facade as tested, or a variant for interventions: 'solid', 'piers<d>' (giardina.geometry)")
    ap.add_argument('--debug', action='store_true')
    a = ap.parse_args()
    h = a.h or a.ell / 4.0
    os.makedirs(a.out_dir, exist_ok=True)
    tag = (f'giardina_l{a.ell:g}_h{h:g}_kn{a.kn:g}_{a.softening}' + (f'_cov{a.cov:g}_seed{a.seed}' if a.cov > 0 else '')
           + ('' if a.profile == 'measured' else f'_{a.profile}')
           + (f'_xp{a.xp:g}_w{a.w:g}' if a.profile == 'hinge' and (a.xp, a.w) != (732.0, 98.0) else '')
           + ('' if a.geometry == 'test' else f'_{a.geometry}'))
    t0 = time.time()
    openings, lintels = G.geometry(a.geometry)
    msh = build_mesh(h, openings=openings, lintels=lintels)
    print(f'mesh built {time.time() - t0:.1f}s', flush=True)
    cd, mid = cell_data(msh, lintels=lintels)
    if a.cov > 0:
        from crackcause.randomfield import gaussian_field, lognormal_property
        zf = gaussian_field(0.0, G.W, 0.0, G.H, a.corr, 5.0, np.random.default_rng(a.seed))(mid[:, [1, 0]])
        cd['ft'] = np.where(cd['dmg'] > 0.5, lognormal_property(G.MASONRY['ft'], a.cov, zf), cd['ft'])
    bf = base_facets(msh)
    reg = dict(eta_split=a.smooth, eta_contact=a.eta_contact, u_snes_type=a.snes, prox=a.prox, atol_u=a.atol_u) if a.smooth > 0 else {}
    model = FxPFCZM(msh, cd, a.ell, G.T, softening=a.softening, base_facets=bf, kn=a.kn, kt=G.INTERFACE['kt'],
                    **reg)
    model.debug = a.debug
    print(f'model (forms compiled) {time.time() - t0:.1f}s', flush=True)
    xq = model.Q.tabulate_dof_coordinates()[:, 0]
    if a.profile == 'hinge':                          # the two-parameter shape of the simulation campaign
        phi_nodes = hinge_profile(xq, a.xp, a.w)
    else:
        xs, phi = G.settlement_shape()
        if a.profile == 'measured_nouplift':          # the measured shape without its slight uplift
            phi = np.minimum(phi, 0.0)
        phi_nodes = np.interp(xq, xs, phi)           # ground displacement = S * phi(x)
    n_cells = msh.topology.index_map(2).size_local
    print(f'mesh: {n_cells} cells, {model.Q.dofmap.index_map.size_local} nodes, base facets {len(bf)}; '
          f'setup {time.time() - t0:.1f}s', flush=True)
    # experimental stages of Fig. 5 (nominal 2.5/5/7.5/10 mm) in terms of S = settlement at x = 30 mm
    # of the applied profiles (Fig. 8): 2.26, 4.54, 6.81, 9.10 mm
    rec_S = sorted([0.5, 1.0, 1.5, 2.0, 2.5, 3.75, 5.0, 6.25, 7.5, 8.75, 10.0, 11.5] + list(G.EXP_STAGES_S))
    rec_S = [s for s in rec_S if s <= a.smax]
    coords = model.Q.tabulate_dof_coordinates()[:, :2]
    cells = model.Q.dofmap.list.copy()
    part = dict(S=[], d=[], u=[])

    def save(records, log):
        np.savez_compressed(os.path.join(a.out_dir, tag + '.npz'), coords=coords, cells=cells,
                            dmg=cd['dmg'], S=np.array([r['S'] for r in records]),
                            d=np.array([r['d'] for r in records]),
                            u=np.array([r['u'] for r in records]), log=json.dumps(log))

    def on_record(S, m):                              # save partial results after every record
        part['S'].append(S)
        part['d'].append(m.d.x.array.copy())
        part['u'].append(m.u.x.array.copy())
        save([dict(S=s_, d=d_, u=u_) for s_, d_, u_ in zip(part['S'], part['d'], part['u'])], [])

    records, log = model.run_settlement(phi_nodes, a.smax, record_at=rec_S, relax=a.relax, tol=a.tol,
                                        keep_work=True, u_inner_it=a.inner or None, callback=on_record)
    save(records, log)
    info = dict(tag=tag, ell=a.ell, h=h, kn=a.kn, softening=a.softening, smooth=a.smooth, geometry=a.geometry,
                profile=a.profile, xp=a.xp, w=a.w, cov=a.cov, seed=a.seed,
                eta_contact=a.eta_contact, snes=a.snes, prox=a.prox, inner=a.inner,
                cells=int(n_cells), failed=any(r.get('failed', False) for r in log),
                seconds=time.time() - t0, steps=len(log), unconverged=int(sum(1 for r in log if not r.get('ok', True))))
    json.dump(info, open(os.path.join(a.out_dir, tag + '.json'), 'w'), indent=1)
    print(json.dumps(info), flush=True)


if __name__ == '__main__':
    main()

"""Nooru-Mohamed DEN test, load-path 4 (lateral shear Ps at P = 0, then axial elongation at
constant Ps), PF-CZM in FEniCSx (WSL).

usage: python fx_den.py OUT_DIR --ps 10 [--ell 5 --h 1.25 --ft 3.0 --Gf 0.11 --E 32800 --dmax 0.2]
Saves OUT_DIR/den_ps{Ps}_l{ell}_h{h}.npz: mesh, phase field at the record points, crack-line maps
(2 mm pixels), P-delta and Ps-delta_s curves, step log.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
from crackcause import fx_den as D  # noqa: E402
from crackcause.fx_pfczm import FxPFCZM  # noqa: E402
from crackcause.observe_fx import Rasterizer, crack_lines  # noqa: E402

IMG = (100, 100)                                   # 2 mm pixels over the 200 x 200 specimen
RECORD = (0.0, 0.02, 0.05, 0.1, 0.2)               # record points in axial elongation (mm)


class DENModel:
    """Facade-independent wrapper: builds mesh, BCs and the PF-CZM model once; reusable."""

    def __init__(self, ell=5.0, h=1.25, softening='linear', **solver_kw):
        from dolfinx import fem
        from petsc4py import PETSc
        self.msh = D.build_mesh(h)
        self.cd, self.mid = D.cell_data(self.msh, Ps=0.0)
        self.Vc = fem.Constant(self.msh, PETSc.ScalarType(0.0))
        factory, self.store = D.boundary_conditions(self.msh, self.Vc)
        self.model = FxPFCZM(self.msh, self.cd, ell, D.T, softening=softening, bcs_u=factory, **solver_kw)
        self.top_y = self.store['top_y']
        m = self.model
        self.coords = m.Q.tabulate_dof_coordinates()[:, :2]
        self.cells = m.Q.dofmap.list.copy()
        xv = m.V.tabulate_dof_coordinates()[:, :2]
        self.side_x = 2 * np.flatnonzero(xv[:, 0] < -1e-9)          # ux dofs of the platen leg
        self.spec = self.cd['E'] < 1e6                             # specimen cells (not platen)
        self.ras = Rasterizer(self.coords, self.cells[self.spec], (0.0, D.L, 0.0, D.L), IMG)

    def set_params(self, Ps, ft=D.MORTAR['ft'], Gf=D.MORTAR['Gf'], E=D.MORTAR['E']):
        cd, _ = D.cell_data(self.msh, Ps=Ps, ft=ft, Gf=Gf, E=E)
        m = self.model
        n = len(cd['E'])
        m.set_materials(ft=cd['ft'], Gf=cd['Gf'], E=cd['E'])
        b = np.zeros((n, 2))
        b[:, 0] = cd['fx']
        m.b.x.array[:b.size] = b.ravel()
        m.b.x.scatter_forward()
        if hasattr(m, '_b0'):
            del m._b0

    def crack_map(self, threshold=0.9):
        return crack_lines(self.ras(self.model.d.x.array), self.ras.inside, threshold=threshold)

    def run(self, n_shear=10, dmax=0.2, dd0=0.004, dd_min=0.0005, dd_max=0.01, record_at=RECORD,
            tol=1e-3, max_it=80, max_it_hard=400, u_inner_it=None, verbose=True, callback=None):
        m, Vc, top_y = self.model, self.Vc, self.top_y
        m.reset()
        t0 = time.time()
        log, curve, records = [], [], []
        # vertical stiffness of the undamaged specimen for the P = 0 control
        m.load_factor.value = 0.0
        Vc.value = 1e-3
        m.solve_u()
        K = m.reaction(top_y) / 1e-3
        Vc.value = 0.0
        m.u.x.array[:] = 0.0

        def u_P0(max_it=50, accept_max_it=False):
            ru, nit = 0, 0
            for _ in range(12):
                ru, nit = m.solve_u(max_it, accept_max_it)
                if ru <= 0:
                    return ru, nit
                P = m.reaction(top_y)
                if abs(P) < 1.0:
                    break
                Vc.value = float(Vc.value) - P / K
            return ru, nit

        def state():
            return dict(P=m.reaction(top_y), V=float(Vc.value), ds=float(m.u.x.array[self.side_x].mean()),
                        dmax=float(m.d.x.array.max()))

        # phase 1: lateral compressive shear at zero axial force
        m.u_hook = u_P0
        for k in range(1, n_shear + 1):
            m.load_factor.value = k / n_shear
            ok, it = m.staggered(tol, max_it_hard, u_inner_it=u_inner_it)
            if not ok:
                m.u_hook = None
                return dict(status='shear_failed', log=log, curve=curve, records=records)
            m.commit()
            s = state()
            log.append(dict(phase=1, lf=k / n_shear, ok=ok, it=it, sec=time.time() - t0, **s))
            curve.append((0.0, s['P'], s['ds']))
            if verbose:
                print(f'shear {k}/{n_shear}: it={it} P={s["P"]:.1f} V={s["V"]:.5f} ds={s["ds"]:.5f} '
                      f'dmax={s["dmax"]:.3f} t={time.time() - t0:.0f}s', flush=True)
        m.u_hook = None
        V1 = float(Vc.value)
        rec = sorted(record_at)

        def record(delta):
            records.append(dict(delta=delta, d=m.d.x.array.copy()))
            if callback is not None:
                callback(delta, self)

        if rec and rec[0] <= 0.0:
            record(0.0)
        # phase 2: axial elongation at constant Ps
        delta, dd = 0.0, dd0
        while delta < dmax - 1e-12:
            dt = min(delta + dd, dmax)
            nxt = [r for r in rec if r > delta + 1e-12]
            if nxt and dt > nxt[0] + 1e-12:
                dt = nxt[0]
            snap = m.snapshot()
            Vc.value = V1 + dt
            ok, it = m.staggered(tol, max_it_hard, u_inner_it=u_inner_it)
            if not ok and dd > dd_min:
                m.restore(snap)
                Vc.value = V1 + delta
                dd = max(dd / 2.0, dd_min)
                continue
            if not ok:                                   # loss of equilibrium under constant Ps: end of test
                m.restore(snap)
                Vc.value = V1 + delta
                log.append(dict(phase=2, delta=dt, ok=False, failed=True, it=it, sec=time.time() - t0))
                if verbose:
                    print(f'delta={dt * 1e3:.1f}um: no convergence at the minimum increment; stopping', flush=True)
                if not records or records[-1]['delta'] < delta:
                    record(delta)                            # final state = last converged step
                return dict(status='failed', log=log, curve=curve, records=records)
            m.commit()
            delta = dt
            s = state()
            log.append(dict(phase=2, delta=delta, ok=ok, it=it, dd=dd, sec=time.time() - t0, **s))
            curve.append((delta, s['P'], s['ds']))
            if verbose:
                print(f'delta={delta * 1e3:.1f}um dd={dd * 1e3:.2f} ok={ok} it={it} P={s["P"]:.1f} '
                      f'ds={s["ds"] * 1e3:.1f}um dmax={s["dmax"]:.3f} t={time.time() - t0:.0f}s', flush=True)
            if any(abs(delta - r) < 1e-12 for r in rec):
                record(delta)
            if ok and it < 8:
                dd = min(dd * 1.5, dd_max)
            elif it > max_it:
                dd = max(dd / 2.0, dd_min)
        if not records or records[-1]['delta'] < delta:
            record(delta)
        return dict(status='ok', log=log, curve=curve, records=records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out_dir')
    ap.add_argument('--ps', type=float, required=True, help='lateral shear force (kN), > 0 compressive')
    ap.add_argument('--ell', type=float, default=5.0)
    ap.add_argument('--h', type=float, default=1.25)
    ap.add_argument('--ft', type=float, default=D.MORTAR['ft'])
    ap.add_argument('--Gf', type=float, default=D.MORTAR['Gf'])
    ap.add_argument('--E', type=float, default=D.MORTAR['E'])
    ap.add_argument('--dmax', type=float, default=0.2)
    ap.add_argument('--inner', type=int, default=0, help='cap on inner u Newton steps (0 = exact)')
    ap.add_argument('--debug', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    den = DENModel(a.ell, a.h, eta_split=1e-3 * a.ft, eta_contact=1e-3)
    den.model.debug = a.debug
    n_cells = den.msh.topology.index_map(2).size_local
    print(f'mesh {n_cells} cells, {len(den.coords)} nodes; setup {time.time() - t0:.1f}s', flush=True)
    den.set_params(a.ps * 1e3, ft=a.ft, Gf=a.Gf, E=a.E)
    maps = []
    res = den.run(dmax=a.dmax, u_inner_it=a.inner or None,
                  callback=lambda delta, obj: maps.append(obj.crack_map()))
    tag = f'den_ps{a.ps:g}_l{a.ell:g}_h{a.h:g}'
    np.savez_compressed(os.path.join(a.out_dir, tag + '.npz'), coords=den.coords, cells=den.cells,
                        dmg=den.cd['dmg'], spec=den.spec, delta=np.array([r['delta'] for r in res['records']]),
                        d=np.array([r['d'] for r in res['records']]), maps=np.array(maps, np.uint8),
                        curve=np.array(res['curve']), log=json.dumps(res['log']), status=res['status'])
    print(f'{tag}: {res["status"]}, total {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()

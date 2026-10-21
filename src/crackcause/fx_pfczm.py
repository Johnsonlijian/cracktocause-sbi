"""PF-CZM (Wu 2017) in FEniCSx (dolfinx >= 0.10) with spectral stress split, compression-only
base springs and alternate minimization with adaptive load stepping.

Runs in the WSL environment (see scripts/setup_wsl_fenicsx.sh). Units: mm, N, MPa.

Per-cell material data (DG0): E, nu, ft, Gf, rho_g, damageable flag, plate load density.
Phase field P1 with bounds [d_n, 1] (PETSc SNES vinewtonrsls); displacement P1 (SNES newtonls).
"""
import time

import numpy as np
import ufl
from dolfinx import fem, la, mesh as dmesh
from dolfinx.fem.petsc import NonlinearProblem
from petsc4py import PETSc

SOFTENING = {'linear': (2.0, -0.5, 0.0),
             'exponential': (2.5, 2.0 ** (5.0 / 3.0) - 3.0, 0.0),
             'cornelissen': (2.0, 1.3868, 0.6567)}
C0 = np.pi


def _ramp(x, eta):
    return 0.5 * (x + ufl.sqrt(x * x + eta * eta))


LU = {'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'}


class FxPFCZM:
    """Phase-field cohesive fracture of a 2D plane-stress body (thickness t).

    cell_data: dict of per-cell numpy arrays (local cells): E, nu, ft, Gf, rho_g, dmg (0/1),
               fx, fy (extra body-force density, e.g. load plates, N/mm^3).
    base_facets: facet indices carrying compression-only normal springs (kn, N/mm^3)
                 and weak tangential springs (kt) against a prescribed ground displacement.
    """

    def __init__(self, msh, cell_data, ell, thickness, softening='linear', base_facets=None,
                 kn=0.0, kt=0.0, k_res=1e-6, eta_split=1e-6, eta_contact=1e-5, quad_degree=2, bcs_u=None,
                 u_snes_type='newtonls', prox=0.0, atol_u=1e-6):
        self.msh, self.t, self.ell = msh, float(thickness), float(ell)
        tdim = msh.topology.dim
        self.V = fem.functionspace(msh, ('Lagrange', 1, (2,)))
        self.Q = fem.functionspace(msh, ('Lagrange', 1))
        self.D0 = fem.functionspace(msh, ('DG', 0))
        self.D0v = fem.functionspace(msh, ('DG', 0, (2,)))
        p, a2, a3 = SOFTENING[softening]

        def dg0(arr):
            f = fem.Function(self.D0)
            f.x.array[:len(arr)] = arr
            f.x.scatter_forward()
            return f

        cd = cell_data
        self.E, self.nu, self.ft, self.Gf = dg0(cd['E']), dg0(cd['nu']), dg0(cd['ft']), dg0(cd['Gf'])
        self.dmg = dg0(cd['dmg'])
        a1_arr = 4.0 * cd['E'] * cd['Gf'] / (np.pi * self.ell * cd['ft'] ** 2)
        self.Y0_arr = cd['ft'] ** 2 / (2.0 * cd['E'])
        self.a1 = dg0(a1_arr)
        self.b = fem.Function(self.D0v)                      # body force density (gravity + plates)
        bvec = np.zeros((len(cd['E']), 2))
        bvec[:, 0] = cd.get('fx', 0.0)
        bvec[:, 1] = -cd['rho_g'] + cd.get('fy', 0.0)
        self.b.x.array[:bvec.size] = bvec.ravel()
        self.b.x.scatter_forward()
        self.load_factor = fem.Constant(msh, PETSc.ScalarType(0.0))

        self.u, self.d = fem.Function(self.V), fem.Function(self.Q)
        self.d_n, self.d_lb, self.d_ub = fem.Function(self.Q), fem.Function(self.Q), fem.Function(self.Q)
        self.d_ub.x.array[:] = 1.0
        self.H, self.H_n, self.Ybar = fem.Function(self.D0), fem.Function(self.D0), fem.Function(self.D0)
        self.H.x.array[:len(self.Y0_arr)] = self.Y0_arr
        self.H_n.x.array[:] = self.H.x.array
        self.g = fem.Function(self.Q)                         # ground vertical displacement (base)

        t, ell = fem.Constant(msh, PETSc.ScalarType(self.t)), fem.Constant(msh, PETSc.ScalarType(self.ell))
        I = ufl.Identity(2)

        def eps(w):
            return ufl.sym(ufl.grad(w))

        mu = self.E / (2.0 * (1.0 + self.nu))
        lam = self.E * self.nu / (1.0 - self.nu ** 2)        # plane stress

        def sbar(w):
            e = eps(w)
            return lam * ufl.tr(e) * I + 2.0 * mu * e

        def principal(s):
            pm = 0.5 * (s[0, 0] + s[1, 1])
            R = ufl.sqrt(0.25 * (s[0, 0] - s[1, 1]) ** 2 + s[0, 1] ** 2 + eta_split ** 2)
            return pm + R, pm - R, R

        def omega(dd):
            n = (1.0 - dd) ** p
            q = self.a1 * dd * (1.0 + a2 * dd + a3 * dd * dd)
            return n / (n + q)

        s = sbar(self.u)
        s1, s2, R = principal(s)
        P1 = (s - s2 * I) / (2.0 * R)
        P2 = (s1 * I - s) / (2.0 * R)
        s_plus = _ramp(s1, eta_split) * P1 + _ramp(s2, eta_split) * P2
        w = k_res + (1.0 - k_res) * omega(self.d)
        sigma = s - self.dmg * (1.0 - w) * s_plus

        # P1 strain is constant per cell: one point suffices for the displacement form, while the
        # rational degradation in the phase-field form gets a slightly higher rule
        dx = ufl.Measure('dx', domain=msh, metadata={'quadrature_degree': quad_degree})
        dxu = ufl.Measure('dx', domain=msh, metadata={'quadrature_degree': 1})
        v, du = ufl.TestFunction(self.V), ufl.TrialFunction(self.V)
        Fu = t * ufl.inner(sigma, eps(v)) * dxu - self.load_factor * t * ufl.dot(self.b, v) * dxu
        # proximal term c (u - u_prev): keeps each displacement solve well posed when cracking turns part
        # of the structure into a near-mechanism (e.g. a block separating and falling onto the base
        # springs); it vanishes as the staggered iterations converge (checked through max |u - u_prev|)
        self.u_prox = fem.Function(self.V)
        self.c_prox = fem.Constant(msh, PETSc.ScalarType(prox))
        self.c_base, self.c_max = float(prox), 1e4 * float(prox)
        Fu += self.c_prox * t * ufl.dot(self.u - self.u_prox, v) * dx
        self.kn = fem.Constant(msh, PETSc.ScalarType(kn))
        self.kt = fem.Constant(msh, PETSc.ScalarType(kt))
        if base_facets is not None and len(base_facets):
            facets = np.sort(np.asarray(base_facets, dtype=np.int32))
            ft_tags = dmesh.meshtags(msh, tdim - 1, facets, np.full(facets.size, 1, dtype=np.int32))
            ds = ufl.Measure('ds', domain=msh, subdomain_data=ft_tags)
            # compression-only springs with a C1 quadratic ramp over the first eta_c of penetration:
            # zero force at zero gap, full stiffness k_n beyond eta_c, continuous tangent
            z = self.u[1] - self.g
            ec = eta_contact
            contact = ufl.conditional(ufl.lt(z, -ec), z + 0.5 * ec,
                                      ufl.conditional(ufl.lt(z, 0.0), -z * z / (2.0 * ec), 0.0))
            Fu += t * self.kn * contact * v[1] * ds(1) + t * self.kt * self.u[0] * v[0] * ds(1)
        self.bcs_u = list(bcs_u(self.V) if callable(bcs_u) else (bcs_u or []))
        self._Fu_form = fem.form(Fu)
        self.problem_u = NonlinearProblem(
            Fu, self.u, bcs=self.bcs_u, J=ufl.derivative(Fu, self.u, du), petsc_options_prefix='u_',
            # absolute tolerance in N; with regularized split and contact the attainable residual of
            # a cracked wall is ~0.5-1 N, so atol_u is set to 1e-4 of the applied load by the drivers
            # (commercial codes typically use force-norm ratios of 1e-3 to 1e-2)
            petsc_options={'snes_type': u_snes_type, 'snes_linesearch_type': 'bt', 'snes_rtol': 1e-9,
                           'snes_atol': atol_u, 'snes_stol': 0.0, 'snes_max_it': 50, **LU})
        self._u_linesearch = u_snes_type == 'newtonls'
        self._atol_u = atol_u

        q, dd_ = ufl.TestFunction(self.Q), ufl.TrialFunction(self.Q)
        dv = ufl.variable(self.d)
        domega = ufl.diff(omega(dv), dv)
        Gf = self.Gf
        Fd = self.dmg * t * (domega * self.H * q
                             + Gf / (C0 * ell) * ((2.0 - 2.0 * self.d) * q
                                                  + 2.0 * ell ** 2 * ufl.dot(ufl.grad(self.d), ufl.grad(q)))) * dx \
            + 1e-10 * self.d * q * dx
        self.problem_d = NonlinearProblem(
            Fd, self.d, J=ufl.derivative(Fd, self.d, dd_), petsc_options_prefix='d_',
            petsc_options={'snes_type': 'vinewtonrsls', 'snes_rtol': 1e-9, 'snes_atol': 1e-8,
                           'snes_stol': 0.0, 'snes_max_it': 50, **LU})
        self.problem_d.solver.setVariableBounds(self.d_lb.x.petsc_vec, self.d_ub.x.petsc_vec)

        s1_pos = ufl.conditional(ufl.gt(s1, 0.0), s1, 0.0)
        pts = self.D0.element.interpolation_points
        pts = pts() if callable(pts) else pts
        self._Y_expr = fem.Expression(s1_pos ** 2 / (2.0 * self.E), pts)
        self._dmg_mask = cd['dmg'] > 0.5
        # upper bound 0 on nodes not attached to any damageable cell
        ub = np.zeros_like(self.d_ub.x.array)
        ub[self.Q.dofmap.list[np.flatnonzero(self._dmg_mask)].ravel()] = 1.0
        self.d_ub.x.array[:] = ub
        self.history = []
        self.debug = False
        self.u_hook = None          # optional replacement of solve_u inside the staggered loop

    def true_residual_norm(self):
        """Euclidean norm of the equilibrium residual without the proximal term (Dirichlet rows zeroed)."""
        c = float(self.c_prox.value)
        self.c_prox.value = 0.0
        b = fem.assemble_vector(self._Fu_form)
        b.scatter_reverse(la.InsertMode.add)
        self.c_prox.value = c
        for bc in self.bcs_u:
            b.array[bc.dof_indices()[0]] = 0.0
        return float(np.linalg.norm(b.array))

    def prox_continuation(self, c_start=1e-2, n_levels=7, reps=3):
        """Walk to equilibrium through well-posed regularized problems with decreasing damping
        (proximal-point iterations), e.g. to let a block cut off by a crack settle back onto the
        compression-only base springs; returns the best true residual norm reached (u is left at
        the best iterate)."""
        best_u, best_f = self.u.x.array.copy(), self.true_residual_norm()
        for k in range(n_levels):
            c = c_start * 10.0 ** (-0.5 * k)
            info = []
            for _ in range(reps):
                self.c_prox.value = c
                self.u_prox.x.array[:] = self.u.x.array
                snes = self.problem_u.solver
                snes.setTolerances(max_it=20)
                self.problem_u.solve()
                r, its = snes.getConvergedReason(), snes.getIterationNumber()
                if r <= 0 and r not in (PETSc.SNES.ConvergedReason.DIVERGED_MAX_IT,
                                        PETSc.SNES.ConvergedReason.DIVERGED_LINE_SEARCH):
                    self.u.x.array[:] = self.u_prox.x.array          # diverged: discard the solve
                du = np.abs(self.u.x.array - self.u_prox.x.array).max()
                info.append(f'{r}/{its}/{du:.2e}')
            f = self.true_residual_norm()
            if self.debug:
                print(f'      continuation c={c:.1e}: true |F| = {f:.1f} N  (reason/its/du: {" ".join(info)})',
                      flush=True)
            if np.isfinite(f) and f < best_f:
                best_u, best_f = self.u.x.array.copy(), f
            if best_f < self._atol_u:
                break
        self.u.x.array[:] = best_u
        self.c_prox.value = 0.0
        self.u_prox.x.array[:] = self.u.x.array
        return best_f

    def residual_report(self, n=8):
        """Debug: where is the out-of-balance force? Prints the largest nodal residuals with the local
        phase field and the share of |F|^2 on base nodes, fully cracked nodes (d > 0.9) and the rest."""
        from scipy.spatial import cKDTree
        b = fem.assemble_vector(self._Fu_form)
        b.scatter_reverse(la.InsertMode.add)
        r = b.array.reshape(-1, 2)
        nn = np.linalg.norm(r, axis=1)
        xv = self.V.tabulate_dof_coordinates()[:, :2]
        xq = self.Q.tabulate_dof_coordinates()[:, :2]
        _, iq = cKDTree(xq).query(xv)
        dn = self.d.x.array[iq]
        base = np.isclose(xv[:, 1], 0.0)
        crack = dn > 0.9
        tot = (nn ** 2).sum()
        share = dict(base=(nn[base] ** 2).sum() / tot, cracked=(nn[crack & ~base] ** 2).sum() / tot)
        share['other'] = 1.0 - share['base'] - share['cracked']
        top = np.argsort(nn)[::-1][:n]
        print(f'      residual |F| = {np.sqrt(tot):.1f} N; shares of |F|^2: ' +
              ', '.join(f'{k} {v:.2f}' for k, v in share.items()), flush=True)
        print('      largest nodal residuals (x, y, |r|, d): ' +
              '; '.join(f'({xv[k, 0]:.0f},{xv[k, 1]:.0f}) {nn[k]:.1f} {dn[k]:.2f}' for k in top), flush=True)
        if base.any():
            uy = self.u.x.array.reshape(-1, 2)[:, 1]
            gq = self.g.x.array[iq]
            kb = top[base[top]]
            print('      base nodes among them (x, r_y, gap u_y - g): ' +
                  '; '.join(f'({xv[k, 0]:.0f}) {r[k, 1]:+.1f} {uy[k] - gq[k]:+.4f}' for k in kb), flush=True)

    def reaction(self, dofs):
        """Sum of the residual (= support reaction on the body) over the given V dofs (unrolled)."""
        b = fem.assemble_vector(self._Fu_form)
        b.scatter_reverse(la.InsertMode.add)
        return float(b.array[dofs].sum())

    # ------------------------------------------------------------------
    def set_materials(self, ft=None, Gf=None, E=None, body_scale=None, kn=None):
        """Update per-cell parameters in place (forms stay compiled)."""
        n = len(self.Y0_arr)
        if ft is not None:
            self.ft.x.array[:n] = ft
        if Gf is not None:
            self.Gf.x.array[:n] = Gf
        if E is not None:
            self.E.x.array[:n] = E
        E_, ft_, Gf_ = self.E.x.array[:n], self.ft.x.array[:n], self.Gf.x.array[:n]
        self.a1.x.array[:n] = 4.0 * E_ * Gf_ / (np.pi * self.ell * ft_ ** 2)
        self.Y0_arr = ft_ ** 2 / (2.0 * E_)
        if body_scale is not None:
            if not hasattr(self, '_b0'):
                self._b0 = self.b.x.array.copy()
            self.b.x.array[:] = self._b0 * body_scale
        if kn is not None:
            self.kn.value = kn
        for f in (self.ft, self.Gf, self.E, self.a1, self.b):
            f.x.scatter_forward()

    def reset(self):
        """Zero fields and history for a new simulation on the same mesh/forms."""
        for f in (self.u, self.d, self.d_n, self.d_lb, self.g):
            f.x.array[:] = 0.0
        n = len(self.Y0_arr)
        self.H.x.array[:n] = self.Y0_arr
        self.H_n.x.array[:] = self.H.x.array
        self.load_factor.value = 0.0

    def _update_history(self):
        self.Ybar.interpolate(self._Y_expr)
        n = len(self.Y0_arr)
        h = np.maximum(self.H_n.x.array[:n], self.Ybar.x.array[:n])
        h = np.maximum(h, self.Y0_arr)
        h[~self._dmg_mask] = self.Y0_arr[~self._dmg_mask]
        self.H.x.array[:n] = h

    def solve_u(self, max_it=50, accept_max_it=False):
        """Displacement solve at fixed d. accept_max_it: an inner (inexact) solve of the staggered
        loop may stop after max_it Newton steps; the outer loop restores consistency."""
        snes = self.problem_u.solver
        snes.setTolerances(max_it=max_it)
        CR = PETSc.SNES.ConvergedReason
        u_start = self.u.x.array.copy()
        self.problem_u.solve()
        reason, its = snes.getConvergedReason(), snes.getIterationNumber()
        if accept_max_it and reason in (CR.DIVERGED_MAX_IT, CR.DIVERGED_LINE_SEARCH):
            # inexact inner solve: the backtracking iterate never increases the residual, and the
            # staggered loop continues from it
            return 99, its
        if accept_max_it and reason <= 0:
            self.u.x.array[:] = u_start
            return 99, its
        if self._u_linesearch and (reason == CR.DIVERGED_LINE_SEARCH or reason == CR.DIVERGED_MAX_IT):
            # piecewise-smooth split/contact: backtracking on |F| can stall; restart from the last
            # backtracking iterate with the critical-point, secant and full-step line searches and
            # keep the best iterate
            ls = snes.getLineSearch()
            u0 = self.u.x.array.copy()
            best_u, best_f = u0, snes.getFunctionNorm()
            for kind in ('cp', 'l2', 'basic'):
                if self.debug:
                    print(f'      u-SNES reason {reason} after {its} its, |F| {snes.getFunctionNorm():.3e}; '
                          f'retry with {kind} line search', flush=True)
                self.u.x.array[:] = u0
                ls.setType(kind)
                self.problem_u.solve()
                its += snes.getIterationNumber()
                reason = snes.getConvergedReason()
                fn = snes.getFunctionNorm()
                if reason > 0:
                    break
                if np.isfinite(fn) and fn < best_f:
                    best_u, best_f = self.u.x.array.copy(), fn
            ls.setType('bt')
            if reason <= 0:
                self.u.x.array[:] = best_u
                # residual floor of cracked states: accept up to 10 x tolerance (5e-3 of the load for
                # the facade, within the 1e-2 force-norm ratios customary for quasi-brittle analyses)
                if best_f < 10.0 * self._atol_u:
                    reason = 98
                if self.debug:
                    self.residual_report()
                self.last_fnorm = float(best_f)               # residual of the iterate kept, not of the last try
            else:
                self.last_fnorm = float(snes.getFunctionNorm())
        else:
            self.last_fnorm = float(snes.getFunctionNorm())
        if reason <= 0 and self.debug:
            print(f'      u-SNES reason {reason} after {its} its, |F| {snes.getFunctionNorm():.3e}', flush=True)
        return reason, its

    def solve_d(self):
        self.d_lb.x.array[:] = self.d_n.x.array
        self.d.x.array[:] = np.clip(np.maximum(self.d.x.array, self.d_n.x.array), 0.0, self.d_ub.x.array)
        self.problem_d.solve()
        return self.problem_d.solver.getConvergedReason()

    def staggered(self, tol=1e-3, max_it=100, relax=1.0, anderson=0, u_inner_it=None, tol_u=1e-2):
        """Alternate minimization at the current loading; returns (converged, iterations).

        The fixed-point map G: d -> (solve u at d, update history, solve d) is optionally
        accelerated by Anderson mixing of depth `anderson` (type II, projected onto the bounds,
        restarted when the residual grows) or by over-relaxation `relax`.
        u_inner_it: cap on Newton steps of the inner displacement solves (inexact alternate
        minimization); the final displacement solve is always run to convergence.
        Convergence: max |G(d) - d| < tol and, with the proximal term, max |u - u_prev| < tol_u (mm);
        the out-of-balance force left by the proximal term, c t |u - u_prev|, is then ~1e-5 of the
        self-weight, while a block still moving towards a new contact state keeps iterating.
        """
        Gh, Fh = [], []
        err_prev = np.inf
        inner = dict(max_it=u_inner_it, accept_max_it=True) if u_inner_it else {}
        prox = self.c_base > 0.0
        c = self.c_base
        self.c_prox.value = c
        stalled = 0
        for it in range(1, max_it + 1):
            t0 = time.time()
            x = self.d.x.array.copy()
            self.u_prox.x.array[:] = self.u.x.array
            ru, nu_it = (self.u_hook or self.solve_u)(**inner)
            du = np.abs(self.u.x.array - self.u_prox.x.array).max() if prox else 0.0
            if prox:
                # adaptive proximal coefficient (Levenberg-Marquardt-like): stiffen the regularization
                # when the capped Newton solve stalls or is slow, relax it when it converges quickly
                if ru == 99:
                    c = min(c * (10.0 if nu_it == 0 else 3.0), self.c_max)
                elif nu_it <= 3:
                    c = max(c / 3.0, self.c_base)
                self.c_prox.value = c
            t1 = time.time()
            if ru <= 0 and Gh:                       # accelerated iterate not admissible: fall back
                if self.debug:
                    print(f'    u-solve failed after mixing (reason {ru}); restart from G(d)', flush=True)
                self.d.x.array[:] = Gh[-1]
                Gh, Fh = [], []
                x = self.d.x.array.copy()
                ru, nu_it = self.solve_u(**inner)
            if ru <= 0:
                if self.debug:
                    print(f'    u-solve failed: reason {ru} after {nu_it} its', flush=True)
                self.c_prox.value = self.c_base
                return False, it
            self._update_history()
            rd = self.solve_d()
            t2 = time.time()
            if rd <= 0:
                self.c_prox.value = self.c_base
                return False, it
            g = self.d.x.array.copy()
            f = g - x
            err = np.abs(f).max()
            if self.debug:
                print(f'    it {it}: u {nu_it} its {t1 - t0:.2f}s, d reason {rd} '
                      f'{self.problem_d.solver.getIterationNumber()} its {t2 - t1:.2f}s, '
                      f'dmax {g.max():.3f} err {err:.2e}' + (f' du {du:.1e} c {c:.0e}' if prox else ''), flush=True)
            # converged only if the last (possibly capped) displacement solve reached equilibrium:
            # after unstable growth the damped Newton steps need several staggered iterations to
            # remove the out-of-balance force even though the phase field no longer changes
            # phase field converged but the capped displacement solve sits on its residual floor:
            # go to the final solve after a few such iterations instead of iterating on
            stalled = stalled + 1 if (err < tol and du < tol_u and ru == 99) else 0
            if (err < tol and du < tol_u and ru != 99) or stalled >= 4:
                # final solve of the unregularized equilibrium (fallbacks accept a residual within a
                # few times the tolerance)
                self.c_prox.value = 0.0
                self.u_prox.x.array[:] = self.u.x.array
                ru, _ = (self.u_hook or self.solve_u)()
                if ru <= 0 and prox and self.u_hook is None:
                    # e.g. a block cut off by a crack hanging above the settled base springs:
                    # proximal continuation towards equilibrium, then the exact solve again
                    f = self.prox_continuation()
                    ru2, _ = self.solve_u()
                    self.last_fnorm = self.true_residual_norm()
                    if ru2 > 0:
                        ru = ru2
                    elif min(f, self.last_fnorm) < 10.0 * self._atol_u:
                        ru = 98
                    if self.debug:
                        print(f'      after continuation: reason {ru}, true |F| = {self.last_fnorm:.1f} N', flush=True)
                self.c_prox.value = self.c_base
                return ru > 0, it
            if anderson > 0:
                if err > 2.0 * err_prev:                 # mixing diverging: restart the history
                    Gh, Fh = [], []
                Gh.append(g)
                Fh.append(f)
                if len(Fh) > anderson + 1:
                    Gh.pop(0)
                    Fh.pop(0)
                if len(Fh) >= 2:
                    dF = np.diff(np.array(Fh), axis=0).T
                    dG = np.diff(np.array(Gh), axis=0).T
                    gamma = np.linalg.lstsq(dF, f, rcond=1e-10)[0]
                    xn = g - dG @ gamma
                    self.d.x.array[:] = np.clip(xn, self.d_n.x.array, self.d_ub.x.array)
            elif relax != 1.0:
                self.d.x.array[:] = np.clip(x + relax * f, self.d_n.x.array, self.d_ub.x.array)
            err_prev = err
        self.c_prox.value = self.c_base
        return False, max_it

    def commit(self):
        self.d_n.x.array[:] = self.d.x.array
        self.H_n.x.array[:] = self.H.x.array

    def snapshot(self):
        return (self.u.x.array.copy(), self.d.x.array.copy(), self.H.x.array.copy())

    def restore(self, snap):
        self.u.x.array[:], self.d.x.array[:], self.H.x.array[:] = snap[0], snap[1], snap[2]

    def run_settlement(self, phi_nodes, S_max, dS0=0.25, dS_min=0.02, dS_max=0.5, record_at=(),
                       tol=3e-3, max_it=60, max_it_hard=400, n_load_steps=5, relax=1.0, callback=None,
                       stop_damage_fraction=None, verbose=True, anderson=0, keep_work=False, u_inner_it=None,
                       max_unconverged=3, accept_fnorm_factor=20.0, jump_dS=0.25):
        """Apply gravity/plate loads, then the settlement S * phi (phi given at P1 dofs).

        keep_work=False: a step not converged within max_it is restarted with half the increment.
        keep_work=True: the step keeps iterating up to max_it_hard (unstable crack growth at fixed
        load needs many iterations whatever the increment); only solver failures halve the step,
        and slow steps shrink the next increment.
        """
        t0 = time.time()
        log = []
        self.g.x.array[:] = 0.0
        for k in range(1, n_load_steps + 1):              # initial loads at zero settlement
            self.load_factor.value = k / n_load_steps
            ok, it = self.staggered(tol, max_it_hard, relax, anderson, u_inner_it)
            self.commit()
            log.append(dict(S=0.0, lf=k / n_load_steps, ok=ok, it=it, dmax=float(self.d.x.array.max()),
                            sec=time.time() - t0))
            if verbose:
                print(f'load {k}/{n_load_steps}: ok={ok} it={it} dmax={self.d.x.array.max():.3f} '
                      f't={time.time() - t0:.0f}s', flush=True)
        records = []
        n_unconverged = 0
        record_at = sorted(record_at)
        S, dS = 0.0, dS0
        while S < S_max - 1e-9:
            S_try = min(S + dS, S_max)
            nxt = [r for r in record_at if S < r - 1e-9]
            if nxt and S_try > nxt[0] + 1e-9:
                S_try = nxt[0]                           # land exactly on record points
            snap = self.snapshot()
            self.g.x.array[:] = S_try * phi_nodes
            ok, it = self.staggered(tol, max_it_hard if keep_work else max_it, relax, anderson, u_inner_it)
            if not ok and dS > dS_min:
                self.restore(snap)
                dS = max(dS / 2.0, dS_min)
                continue
            if not ok and not keep_work:                 # unstable growth: iterate longer
                ok, it2 = self.staggered(tol, max_it_hard, relax, anderson, u_inner_it)
                it += it2
            if not ok and jump_dS > dS:
                # stalled at the minimum increment in a non-smooth contact/crack state: retry once
                # with a larger increment (at most to the next record point), which often lands on
                # the smooth side of the state change; otherwise fall back to the stalled state
                fn = getattr(self, 'last_fnorm', np.inf)
                S_jump = min(S + jump_dS, S_max)
                if nxt and S_jump > nxt[0] + 1e-9:
                    S_jump = nxt[0]
                if S_jump > S_try + 1e-9:
                    fail_state = self.snapshot()
                    self.restore(snap)
                    self.g.x.array[:] = S_jump * phi_nodes
                    ok_j, it_j = self.staggered(tol, max_it_hard, relax, anderson, u_inner_it)
                    it += it_j
                    if ok_j:
                        ok, S_try = True, S_jump
                        if verbose:
                            print(f'S={S_jump:.3f}: converged by jumping over the stalled increment', flush=True)
                    else:
                        self.restore(fail_state)
                        self.g.x.array[:] = S_try * phi_nodes
                        self.last_fnorm = fn
            if not ok:
                fn = getattr(self, 'last_fnorm', np.inf)
                if n_unconverged < max_unconverged and fn < accept_fnorm_factor * self._atol_u:
                    # accept a nearly balanced state (residual below ~1% of the load) a few times per
                    # run rather than losing all later stages; flagged in the log
                    n_unconverged += 1
                    unconverged_step = dict(unconverged=True, fnorm=fn)
                    if verbose:
                        print(f'S={S_try:.3f}: accepted unconverged (|F| = {fn:.1f} N)', flush=True)
                else:                                    # give up: keep the records obtained so far
                    self.restore(snap)
                    log.append(dict(S=S_try, lf=1.0, ok=False, it=it, failed=True, dS=dS, fnorm=fn,
                                    sec=time.time() - t0))
                    if verbose:
                        print(f'S={S_try:.3f}: no convergence at the minimum increment; stopping', flush=True)
                    break
            else:
                unconverged_step = {}
            self.commit()
            S = S_try
            dmax = float(self.d.x.array.max())
            frac = float((self.d.x.array > 0.95).mean())
            log.append(dict(S=S, lf=1.0, ok=ok, it=it, dmax=dmax, frac95=frac, dS=dS, sec=time.time() - t0,
                            res=self.true_residual_norm(), **unconverged_step))     # accepted state
            if verbose:
                print(f'S={S:.3f} dS={dS:.3f} ok={ok} it={it} dmax={dmax:.3f} frac(d>0.95)={frac:.4f} '
                      f't={time.time() - t0:.0f}s', flush=True)
            if any(abs(S - r) < 1e-9 for r in record_at):
                records.append(dict(S=S, d=self.d.x.array.copy(), u=self.u.x.array.copy()))
                if callback is not None:
                    callback(S, self)
            # once cracks are open the staggered scheme needs 9-10 iterations per step for any
            # increment (campaign statistics), so growth is allowed up to 11 iterations; with < 8 the
            # increment never recovered from a reduction and cracked runs crawled at dS_min
            if ok and it < 12:
                dS = min(dS * 1.5, dS_max)
            elif keep_work and it > max_it:
                dS = max(dS / 2.0, dS_min)
            if stop_damage_fraction is not None and frac > stop_damage_fraction:
                log[-1]['stopped'] = True
                break
        return records, log

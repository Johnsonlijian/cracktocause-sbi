"""Phase-field regularized cohesive zone model (PF-CZM, Wu 2017) on linear triangles.

Formulation (per unit thickness t):
    Pi(u, d) = int omega(d) psi0(eps) dV + (Gf / c0) int (alpha(d) / ell + ell |grad d|^2) dV
    alpha(d) = 2 d - d^2,  c0 = pi
    omega(d) = (1 - d)^p / ((1 - d)^p + a1 d (1 + a2 d + a3 d^2)),  a1 = 4 lch / (pi ell),  lch = E Gf / ft^2
The crack driving force is Rankine-type, Ybar = <sigma_bar_1>^2 / (2E), with history
H = max(Y0, max_t Ybar), Y0 = ft^2 / (2E), so no damage develops before the tensile strength.
(p, a2, a3) select the softening law; with these choices the regularized response reproduces
the corresponding cohesive law and is (nearly) insensitive to ell.

Discretisation: P1 displacement and P1 phase field; one-point quadrature for the bulk
stiffness, nodal (lumped) quadrature for the local phase-field terms. Solution by alternate
minimisation (staggered) with a box-constrained Newton solve for d (lb = d_n, ub = 1).
Material parameters may vary element-wise, which is how random fields enter.
"""
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

C0 = np.pi
SOFTENING = {                                   # (p, a2, a3), Wu & Nguyen (2018)
    'linear': (2.0, -0.5, 0.0),
    'exponential': (2.5, 2.0 ** (5.0 / 3.0) - 3.0, 0.0),
    'cornelissen': (2.0, 1.3868, 0.6567),
}


def omega(d, a1, p, a2, a3):
    d = np.clip(d, 0.0, 1.0)
    n = (1.0 - d) ** p
    q = a1 * d * (1.0 + a2 * d + a3 * d * d)
    return n / (n + q)


def omega_derivs(d, a1, p, a2, a3):
    """First and second derivatives of omega with respect to d."""
    d = np.clip(d, 0.0, 1.0 - 1e-12)
    om = 1.0 - d
    n, n1, n2 = om ** p, -p * om ** (p - 1.0), p * (p - 1.0) * om ** (p - 2.0)
    q = a1 * d * (1.0 + a2 * d + a3 * d * d)
    q1 = a1 * (1.0 + 2.0 * a2 * d + 3.0 * a3 * d * d)
    q2 = a1 * (2.0 * a2 + 6.0 * a3 * d)
    s, s1 = n + q, n1 + q1
    num = n1 * q - n * q1
    w1 = num / s ** 2
    w2 = ((n2 * q - n * q2) * s - 2.0 * num * s1) / s ** 3
    return w1, w2


def spectral_split(s):
    """Positive part of 2D stresses s = [sx, sy, txy] in principal space and its tangent.

    Returns s_plus (M,3), the tangent P = d s_plus / d s (M,3,3) in stress-Voigt form,
    and the major principal stress s1 (M,). Uses the 2D isotropic-function derivative
    P = sum_i H(s_i) w_i v_i^T + c (I - sum_i w_i v_i^T),  c = (<s1> - <s2>) / (s1 - s2),
    with w_i = [nx^2, ny^2, nx ny] and v_i = [nx^2, ny^2, 2 nx ny] of eigenvector n_i.
    """
    sx, sy, txy = s[:, 0], s[:, 1], s[:, 2]
    p, a = 0.5 * (sx + sy), 0.5 * (sx - sy)
    R = np.sqrt(a * a + txy * txy)
    s1, s2 = p + R, p - R
    scale = np.abs(s1) + np.abs(s2) + 1e-300
    big = R > 1e-12 * scale
    Rs = np.where(big, R, 1.0)
    c2 = np.where(big, a / Rs, 1.0)                 # cos(2 theta) of the major direction
    sn2 = np.where(big, txy / Rs, 0.0)              # sin(2 theta)
    w1 = np.stack([(1 + c2) / 2, (1 - c2) / 2, sn2 / 2], axis=1)
    v1 = np.stack([(1 + c2) / 2, (1 - c2) / 2, sn2], axis=1)
    w2 = np.stack([(1 - c2) / 2, (1 + c2) / 2, -sn2 / 2], axis=1)
    v2 = np.stack([(1 - c2) / 2, (1 + c2) / 2, -sn2], axis=1)
    pos1, pos2 = np.maximum(s1, 0.0), np.maximum(s2, 0.0)
    s_plus = pos1[:, None] * w1 + pos2[:, None] * w2
    h1, h2 = (s1 > 0.0).astype(float), (s2 > 0.0).astype(float)
    diff = s1 - s2
    wide = diff > 1e-12 * scale
    c = np.where(wide, (pos1 - pos2) / np.where(wide, diff, 1.0), h1)
    W1 = w1[:, :, None] * v1[:, None, :]
    W2 = w2[:, :, None] * v2[:, None, :]
    P = (h1[:, None, None] * W1 + h2[:, None, None] * W2
         + c[:, None, None] * (np.eye(3)[None, :, :] - W1 - W2))
    return s_plus, P, s1


def elasticity(E, nu, plane):
    M = E.size
    D = np.zeros((M, 3, 3))
    if plane == 'stress':
        f = E / (1.0 - nu ** 2)
        D[:, 0, 0] = D[:, 1, 1] = f
        D[:, 0, 1] = D[:, 1, 0] = f * nu
        D[:, 2, 2] = f * (1.0 - nu) / 2.0
    elif plane == 'strain':
        f = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        D[:, 0, 0] = D[:, 1, 1] = f * (1.0 - nu)
        D[:, 0, 1] = D[:, 1, 0] = f * nu
        D[:, 2, 2] = f * (1.0 - 2.0 * nu) / 2.0
    else:
        raise ValueError(plane)
    return D


class PFCZM:
    """PF-CZM model on a fixed triangular mesh.

    E, nu, ft, Gf: scalars or per-element arrays. damageable: per-element bool mask;
    non-damageable elements (e.g. a soil-structure interface layer) stay elastic.
    split='none' degrades the whole effective stress (linear u-problem); split='spectral'
    degrades only its positive principal part, sigma = omega(d) sbar+ + sbar-, so closed
    cracks keep transmitting compression (nonlinear u-problem, Newton with consistent tangent).
    no_tension: per-element mask of non-damageable elements that carry no tension at all
    (sigma = sbar-), e.g. a soil-structure interface that may gap.
    """

    def __init__(self, nodes, tris, E, nu, ft, Gf, ell, thickness=1.0, plane='stress',
                 softening='linear', k_res=1e-8, damageable=None, split='none', no_tension=None,
                 no_tension_residual=1e-3):
        self.nodes = np.asarray(nodes, float)
        self.tris = np.asarray(tris, np.int64)
        M, N = self.tris.shape[0], self.nodes.shape[0]
        self.M, self.N, self.ndof = M, N, 2 * N

        def per_elem(v):
            return np.broadcast_to(np.asarray(v, float), (M,)).copy()

        self.E, self.nu, self.ft, self.Gf = per_elem(E), per_elem(nu), per_elem(ft), per_elem(Gf)
        self.ell, self.t, self.k_res = float(ell), float(thickness), float(k_res)
        self.p, self.a2, self.a3 = SOFTENING[softening]
        self.dmg = np.ones(M, bool) if damageable is None else np.asarray(damageable, bool)
        self.no_tension = np.zeros(M, bool) if no_tension is None else np.asarray(no_tension, bool)
        if np.any(self.no_tension & self.dmg):
            raise ValueError('no_tension elements must be non-damageable')
        self.nt_res = float(no_tension_residual)    # small tensile stiffness keeps contact Newton stable
        self.split = 'spectral' if np.any(self.no_tension) else split
        if self.split not in ('none', 'spectral'):
            raise ValueError(split)

        x = self.nodes[self.tris]
        x1, y1, x2, y2, x3, y3 = x[:, 0, 0], x[:, 0, 1], x[:, 1, 0], x[:, 1, 1], x[:, 2, 0], x[:, 2, 1]
        two_a = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        if np.any(two_a <= 0.0):
            raise ValueError('triangles must be counter-clockwise and non-degenerate')
        self.A = 0.5 * two_a
        b = np.stack([y2 - y3, y3 - y1, y1 - y2], axis=1) / two_a[:, None]
        c = np.stack([x3 - x2, x1 - x3, x2 - x1], axis=1) / two_a[:, None]
        B = np.zeros((M, 3, 6))
        B[:, 0, 0::2] = b
        B[:, 1, 1::2] = c
        B[:, 2, 0::2] = c
        B[:, 2, 1::2] = b
        self.B = B
        self.D = elasticity(self.E, self.nu, plane)
        self.Ke0 = self.t * self.A[:, None, None] * np.einsum('mki,mkl,mlj->mij', B, self.D, B)
        dofs = np.empty((M, 6), np.int64)
        dofs[:, 0::2] = 2 * self.tris
        dofs[:, 1::2] = 2 * self.tris + 1
        self.dofs = dofs
        self._ku_rows = np.repeat(dofs, 6, axis=1).ravel()
        self._ku_cols = np.tile(dofs, (1, 6)).ravel()

        # phase-field operators, assembled over damageable elements only
        self.a1 = 4.0 * self.E * self.Gf / (np.pi * self.ell * self.ft ** 2)
        self.Y0 = self.ft ** 2 / (2.0 * self.E)
        tw = self.tris[self.dmg]
        self._tw, self._twr = tw, tw.ravel()
        aw, gfw = self.A[self.dmg], self.Gf[self.dmg]
        bw, cw = b[self.dmg], c[self.dmg]
        G = bw[:, :, None] * bw[:, None, :] + cw[:, :, None] * cw[:, None, :]
        wl = (2.0 * self.ell / C0) * gfw * self.t * aw
        self.Klap = sp.csr_matrix(((wl[:, None, None] * G).ravel(),
                                   (np.repeat(tw, 3, axis=1).ravel(), np.tile(tw, (1, 3)).ravel())),
                                  shape=(N, N))
        self._mw = self.t * aw / 3.0                 # nodal quadrature weight per element node
        self._gw = gfw / (C0 * self.ell)
        self._a1w = self.a1[self.dmg]
        self._mg_node = np.bincount(self._twr, np.repeat(self._mw * self._gw, 3), N)
        touched = np.zeros(N, bool)
        touched[self._twr] = True
        self.d_fixed = ~touched                      # nodes outside the damageable region keep d = 0
        self.fixed = np.zeros(0, np.int64)
        self.free = np.arange(self.ndof)

    # ------------------------------------------------------------------ boundary conditions
    def set_dirichlet(self, dofs):
        """Prescribed dofs; their values are supplied (in sorted-unique order) to run()."""
        self.fixed = np.unique(np.asarray(dofs, np.int64))
        mask = np.ones(self.ndof, bool)
        mask[self.fixed] = False
        self.free = np.flatnonzero(mask)

    # ------------------------------------------------------------------ operators
    def weights(self, d):
        """Element factors on the tensile stress part (with residual stiffness)."""
        dc = d[self.tris].mean(axis=1)
        w = np.ones(self.M)
        w[self.dmg] = omega(dc[self.dmg], self._a1w, self.p, self.a2, self.a3)
        w = self.k_res + (1.0 - self.k_res) * w
        w[self.no_tension] = self.nt_res
        return w

    def stiffness(self, d):
        w = self.weights(d)
        data = (w[:, None, None] * self.Ke0).ravel()
        return sp.csr_matrix((data, (self._ku_rows, self._ku_cols)), shape=(self.ndof, self.ndof))

    def _assemble_vec(self, fe):
        return np.bincount(self.dofs.ravel(), fe.ravel(), self.ndof)

    def _internal(self, u, w):
        """Internal forces and spectral tangent pieces for the split model."""
        eps = np.einsum('mij,mj->mi', self.B, u[self.dofs])
        sbar = np.einsum('mij,mj->mi', self.D, eps)
        s_plus, P, _ = spectral_split(sbar)
        sig = sbar - (1.0 - w)[:, None] * s_plus
        fint = self._assemble_vec((self.t * self.A)[:, None] * np.einsum('mki,mk->mi', self.B, sig))
        return fint, P

    def equilibrium(self, d, u0, u_fixed, f_ext=None, tol=1e-6, maxit=30):
        """Displacements at fixed phase field.

        Returns (u, internal force vector, relative residual). With maxit small this is an
        inexact Newton update, which is sufficient inside the staggered loop.
        """
        if self.split == 'none':
            K = self.stiffness(d)
            u = self.solve_u(K, u_fixed, f_ext)
            return u, K @ u, 0.0
        w = self.weights(d)
        u = u0.copy()
        u[self.fixed] = u_fixed
        fext = np.zeros(self.ndof) if f_ext is None else f_ext
        tA = self.t * self.A
        fint, P = self._internal(u, w)
        rel = np.inf
        for _ in range(maxit):
            r = (fint - fext)[self.free]
            scale = max(np.abs(fint).max(), np.abs(fext).max(), 1e-30)
            rel = np.abs(r).max() / scale
            if rel <= tol:
                break
            Ct = self.D - (1.0 - w)[:, None, None] * np.einsum('mij,mjk->mik', P, self.D)
            Ke = tA[:, None, None] * np.einsum('mki,mkl,mlj->mij', self.B, Ct, self.B)
            K = sp.csr_matrix((Ke.ravel(), (self._ku_rows, self._ku_cols)), shape=(self.ndof, self.ndof))
            Kf = K[self.free][:, self.free].tocsc()
            u[self.free] += spla.spsolve(Kf, -r, permc_spec='COLAMD')
            fint, P = self._internal(u, w)
        else:
            r = (fint - fext)[self.free]
            rel = np.abs(r).max() / max(np.abs(fint).max(), np.abs(fext).max(), 1e-30)
        return u, fint, rel

    def solve_u(self, K, u_fixed, f_ext=None):
        u = np.zeros(self.ndof)
        u[self.fixed] = u_fixed
        Kf = K[self.free]
        rhs = -(Kf[:, self.fixed] @ u[self.fixed])
        if f_ext is not None:
            rhs = rhs + f_ext[self.free]
        u[self.free] = spla.spsolve(Kf[:, self.free].tocsc(), rhs, permc_spec='MMD_AT_PLUS_A')
        return u

    def stresses(self, u):
        eps = np.einsum('mij,mj->mi', self.B, u[self.dofs])
        return np.einsum('mij,mj->mi', self.D, eps)          # effective (undamaged) stress

    def driving_force(self, u):
        s = self.stresses(u)
        s1 = 0.5 * (s[:, 0] + s[:, 1]) + np.sqrt(0.25 * (s[:, 0] - s[:, 1]) ** 2 + s[:, 2] ** 2)
        return np.maximum(s1, 0.0) ** 2 / (2.0 * self.E)

    def residual_d(self, d, H_w):
        den = d[self._tw]
        w1, w2 = omega_derivs(den, self._a1w[:, None], self.p, self.a2, self.a3)
        r = self._mw[:, None] * (H_w[:, None] * w1 + self._gw[:, None] * (2.0 - 2.0 * den))
        j = self._mw[:, None] * (H_w[:, None] * w2 - 2.0 * self._gw[:, None])
        R = np.bincount(self._twr, r.ravel(), self.N) + self.Klap @ d
        jd = np.bincount(self._twr, j.ravel(), self.N)
        return R, jd

    def solve_d(self, d, lb, H, tol=1e-9, maxit=40):
        """Box-constrained Newton (active set) for the phase field at fixed history H."""
        H_w = H[self.dmg]
        d = np.clip(np.maximum(d, lb), 0.0, 1.0)
        d[self.d_fixed] = 0.0
        floor = 0.05 * self._mg_node                       # keeps the Newton matrix SPD
        for _ in range(maxit):
            R, jd = self.residual_d(d, H_w)
            act = self.d_fixed | ((d <= lb + 1e-14) & (R > 0.0)) | ((d >= 1.0 - 1e-14) & (R < 0.0))
            ina = np.flatnonzero(~act)
            if ina.size == 0:
                break
            J = self.Klap[ina][:, ina] + sp.diags(np.maximum(jd[ina], floor[ina]))
            delta = spla.spsolve(J.tocsc(), -R[ina], permc_spec='MMD_AT_PLUS_A')
            dn = d.copy()
            dn[ina] += delta
            dn = np.clip(dn, lb, 1.0)
            dn[self.d_fixed] = 0.0
            step = np.abs(dn - d).max()
            d = dn
            if step < tol:
                break
        return d

    # ------------------------------------------------------------------ energies
    def fracture_energy(self, d):
        den = d[self._tw]
        alpha = 2.0 * den - den ** 2
        return float((self._mw[:, None] * self._gw[:, None] * alpha).sum() + 0.5 * d @ (self.Klap @ d))

    # ------------------------------------------------------------------ driver
    def run(self, lambdas, u_fixed_fun, reaction_dofs=None, f_ext_fun=None, tol=1e-4,
            max_stag=500, snapshot_at=(), verbose=False, newton_per_stag=2, tol_u=1e-6,
            relax=1.0):
        """Quasi-static loading through the load factors `lambdas`.

        u_fixed_fun(lam) returns the prescribed values for self.fixed (sorted order).
        Inside the staggered loop the displacement problem gets only `newton_per_stag`
        Newton updates (inexact); a step is converged when the phase-field change is below
        `tol` and the relative equilibrium residual below `tol_u`.
        relax > 1 over-relaxes the phase-field update of alternate minimization
        (d <- d + relax (d_hat - d), projected onto [d_n, 1]); relax = 1 is plain staggering.
        Returns final fields, a per-step history and phase-field snapshots.
        """
        d = np.zeros(self.N)
        H = self.Y0.copy()
        u = np.zeros(self.ndof)
        snap_set = {int(i) for i in snapshot_at}
        hist = {k: [] for k in ('lam', 'reaction', 'n_stag', 'E_frac', 'E_el', 'dmax', 'seconds')}
        snaps = []
        t_start = time.time()
        for k, lam in enumerate(lambdas):
            d_n, H_n = d.copy(), H.copy()
            ufix = u_fixed_fun(lam)
            fext = None if f_ext_fun is None else f_ext_fun(lam)
            n_it = 0
            for n_it in range(1, max_stag + 1):
                u, _, rel = self.equilibrium(d, u, ufix, fext, tol=tol_u, maxit=newton_per_stag)
                H = np.maximum(H_n, self.driving_force(u))
                d_new = self.solve_d(d, d_n, H)
                err = np.abs(d_new - d).max()
                if relax != 1.0:
                    d_new = np.clip(d + relax * (d_new - d), d_n, 1.0)
                    d_new[self.d_fixed] = 0.0
                d = d_new
                if err < tol and rel < tol_u:
                    break
            u, fint, _ = self.equilibrium(d, u, ufix, fext, tol=tol_u)
            reac = float(fint[reaction_dofs].sum()) if reaction_dofs is not None else np.nan
            hist['lam'].append(float(lam))
            hist['reaction'].append(reac)
            hist['n_stag'].append(n_it)
            hist['E_frac'].append(self.fracture_energy(d))
            hist['E_el'].append(float(0.5 * u @ fint))
            hist['dmax'].append(float(d.max()))
            hist['seconds'].append(time.time() - t_start)
            if k in snap_set:
                snaps.append((float(lam), d.copy()))
            if verbose and (k % 10 == 0 or k == len(lambdas) - 1):
                print(f'step {k:4d} lam={lam:.5g} R={reac:.5g} stag={n_it} dmax={d.max():.3f} '
                      f't={time.time() - t_start:.1f}s', flush=True)
        return {'u': u, 'd': d, 'H': H, 'hist': {k: np.asarray(v) for k, v in hist.items()}, 'snaps': snaps}

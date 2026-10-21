"""Parametric facade-on-interface model under a settlement trough, for dataset generation.

theta (physical units, mm / N / MPa):
    S_max   maximum greenfield settlement at the end of loading
    x0      trough centre (facade spans 0..LX)
    i       trough width parameter (Peck)
    K       trough width factor, z0 = i / K (sets horizontal movements, O'Reilly-New)
nuisance:
    E_int   interface Young modulus;  tension: 'notension' | 'elastic'
    q_floor floor/roof line load per level
    E, ft, Gf, ft_cov, ft_corr   masonry properties and strength-field statistics
"""
import numpy as np

from .mesh import nodes_where, structured_tri_mesh
from .observe import crack_map
from .pfczm import PFCZM
from .randomfield import gaussian_field, lognormal_property

LX, LY, T_WALL, T_INT = 10000.0, 6000.0, 200.0, 100.0
OPENINGS = [
    (1500.0, 2500.0, 0.0, 2200.0),
    (4400.0, 5600.0, 900.0, 2100.0), (7400.0, 8600.0, 900.0, 2100.0),
    (1400.0, 2600.0, 3700.0, 4900.0), (4400.0, 5600.0, 3700.0, 4900.0), (7400.0, 8600.0, 3700.0, 4900.0),
]
FLOOR_LEVELS = (3000.0, LY)
RHO_G = 1.8e-5
NU = 0.2


def sample_prior(rng):
    """One draw of all parameters (wide engineering ranges)."""
    return {
        'S_max': float(np.exp(rng.uniform(np.log(5.0), np.log(60.0)))),
        'x0': float(rng.uniform(-0.5 * LX, 1.5 * LX)),
        'i': float(rng.uniform(2000.0, 8000.0)),
        'K': float(rng.uniform(0.3, 0.6)),
        'E_int': float(np.exp(rng.uniform(np.log(10.0), np.log(1000.0)))),
        'tension': 'notension' if rng.random() < 0.5 else 'elastic',
        'q_floor': float(rng.uniform(0.0, 25.0)),
        'E': float(rng.uniform(1500.0, 6000.0)),
        'ft': float(rng.uniform(0.10, 0.30)),
        'Gf': float(np.exp(rng.uniform(np.log(0.005), np.log(0.03)))),
        'ft_cov': float(rng.uniform(0.05, 0.25)),
        'ft_corr': float(rng.uniform(300.0, 1500.0)),
        'field_seed': int(rng.integers(0, 2 ** 31 - 1)),
    }


def simulate(theta, h=100.0, ell=150.0, n_steps=40, n_stages=10, img_shape=(96, 160),
             tol=1e-3, max_stag=200, relax=1.6):
    """Run one facade simulation; return crack maps at n_stages settlement stages."""
    xs = np.linspace(0.0, LX, int(round(LX / h)) + 1)
    ys = np.concatenate([np.linspace(-T_INT, 0.0, 3)[:-1], np.linspace(0.0, LY, int(round(LY / h)) + 1)])

    def keep(xc, yc):
        k = np.ones_like(xc, bool)
        for xl, xh, yl, yh in OPENINGS:
            k &= ~((xc > xl) & (xc < xh) & (yc > yl) & (yc < yh))
        return k

    nodes, tris = structured_tri_mesh(xs, ys, keep_cell=keep)
    cen = nodes[tris].mean(axis=1)
    wall = cen[:, 1] > 0.0
    rng = np.random.default_rng(theta['field_seed'])
    z = gaussian_field(0.0, LX, -T_INT, LY, theta['ft_corr'], h, rng)(cen[:, ::-1])
    ft = np.where(wall, lognormal_property(theta['ft'], theta['ft_cov'], z), 1e6)
    E = np.where(wall, theta['E'], theta['E_int'])
    model = PFCZM(nodes, tris, E, NU, ft, theta['Gf'], ell, thickness=T_WALL, plane='stress',
                  softening='linear', damageable=wall, split='spectral', k_res=1e-6,
                  no_tension=(~wall) if theta['tension'] == 'notension' else None)
    base = nodes_where(nodes, lambda x, y: np.isclose(y, -T_INT))
    model.set_dirichlet(np.concatenate([2 * base, 2 * base + 1]))
    xb = nodes[base, 0]
    S = theta['S_max'] * np.exp(-(xb - theta['x0']) ** 2 / (2.0 * theta['i'] ** 2))
    z0 = theta['i'] / theta['K']
    vals = np.concatenate([-(xb - theta['x0']) / z0 * S, -S])[np.argsort(np.concatenate([2 * base, 2 * base + 1]))]

    f = np.zeros(model.ndof)
    w_e = RHO_G * T_WALL * model.A * wall
    np.add.at(f, 2 * tris + 1, -np.repeat(w_e[:, None] / 3.0, 3, axis=1))
    for y_level in FLOOR_LEVELS:
        lv = nodes_where(nodes, lambda x, y: np.isclose(y, y_level))
        lv = lv[np.argsort(nodes[lv, 0])]
        f[2 * lv + 1] -= theta['q_floor'] * np.gradient(nodes[lv, 0])

    lams = np.concatenate([[0.0], np.linspace(0.0, 1.0, n_steps + 1)[1:]])
    stage_idx = np.unique(np.round(np.linspace(0, n_steps, n_stages + 1)[1:]).astype(int))
    res = model.run(lams, lambda lam: lam * vals, f_ext_fun=lambda lam: f, tol=tol,
                    max_stag=max_stag, snapshot_at=stage_idx, relax=relax)
    bbox = (0.0, LX, 0.0, LY)
    maps = np.stack([crack_map(nodes, tris[wall], d, bbox, img_shape) for _, d in res['snaps']])
    stage_lams = np.array([lam for lam, _ in res['snaps']])
    return {
        'maps': maps.astype(np.uint8), 'stage_lams': stage_lams,
        'n_stag': res['hist']['n_stag'], 'seconds': float(res['hist']['seconds'][-1]),
    }

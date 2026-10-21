"""Elastic diagnostic of the façade: where does the undamaged wall first reach f_t under the imposed
settlement, and how does that depend on the interface stiffness, the loads and the profile?

usage: python diag_elastic_facade.py OUT_DIR
Prints, for each variant, the settlement at which max sigma_1 = f_t (elastic, linear in S after the
initial loads are applied) and the locations of the five largest sigma_1 values (masonry only).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
from crackcause import giardina as G  # noqa: E402
from crackcause.fx_facade import base_facets, build_mesh, cell_data  # noqa: E402
from crackcause.fx_pfczm import FxPFCZM  # noqa: E402
from crackcause.settlement import hinge_profile  # noqa: E402

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
msh = build_mesh(10.0)
cd, mid = cell_data(msh)
masonry = cd['dmg'] > 0.5
cd_el = dict(cd)
cd_el['ft'] = np.full_like(cd['ft'], 1e3)                         # no damage
model = FxPFCZM(msh, cd_el, 20.0, G.T, softening='linear', base_facets=base_facets(msh), kn=G.INTERFACE['kn'],
                kt=G.INTERFACE['kt'], eta_split=1e-2, eta_contact=5e-2, prox=0.0, atol_u=1e-3)
from dolfinx import fem  # noqa: E402
import ufl  # noqa: E402

eps = ufl.sym(ufl.grad(model.u))
mu = model.E / (2 * (1 + model.nu))
lam = model.E * model.nu / (1 - model.nu ** 2)
s = lam * ufl.tr(eps) * ufl.Identity(2) + 2 * mu * eps
s1 = 0.5 * (s[0, 0] + s[1, 1]) + ufl.sqrt(0.25 * (s[0, 0] - s[1, 1]) ** 2 + s[0, 1] ** 2)
pts = model.D0.element.interpolation_points
pts = pts() if callable(pts) else pts
S1 = fem.Function(model.D0)
expr = fem.Expression(s1, pts)
xq = model.Q.tabulate_dof_coordinates()[:, 0]
xs, ps = G.settlement_shape()
variants = {
    'measured profile, kn=0.7': (np.interp(xq, xs, ps), 0.7, 1.0),
    'hinge fit (no uplift), kn=0.7': (hinge_profile(xq, 732.0, 98.0), 0.7, 1.0),
    'measured profile, kn=0.2': (np.interp(xq, xs, ps), 0.2, 1.0),
    'measured profile, kn=2.0': (np.interp(xq, xs, ps), 2.0, 1.0),
    'measured profile, no loads': (np.interp(xq, xs, ps), 0.7, 0.0),
}
ft = G.MASONRY['ft']
for name, (phi, kn, q) in variants.items():
    model.set_materials(body_scale=q, kn=kn)
    model.reset()
    model.load_factor.value = 1.0
    model.g.x.array[:] = 0.0
    model.solve_u()
    S1.interpolate(expr)
    base = S1.x.array[:len(masonry)].copy()
    res = []
    for S in (1.0, 2.0, 3.0):
        model.g.x.array[:] = S * phi
        model.solve_u()
        S1.interpolate(expr)
        res.append(S1.x.array[:len(masonry)].copy())
    # find S at which the peak masonry stress reaches f_t (stress ~ linear in S once contact is stable)
    peaks = [r[masonry].max() for r in res]
    i = np.argsort(np.where(masonry, res[1], -np.inf))[::-1][:5]
    print(f'{name}: peak sigma1 at S=1,2,3 mm: ' + ', '.join(f'{p:.3f}' for p in peaks)
          + f' MPa (ft = {ft}); top-5 locations at S=2: '
          + '; '.join(f'({mid[k, 0]:.0f},{mid[k, 1]:.0f}) {res[1][k]:.3f}' for k in i), flush=True)
    np.save(os.path.join(out, f's1_{name.split(",")[0].replace(" ", "_")}_kn{kn}_q{q}.npy'), res[1])
np.save(os.path.join(out, 'mid.npy'), mid)
np.save(os.path.join(out, 'masonry.npy'), masonry)

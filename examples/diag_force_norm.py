"""Norms of the external nodal force vector of the façade (to express residual tolerances as the
force-norm ratios used in nonlinear FE practice).

usage: python diag_force_norm.py
"""
import os
import sys

import numpy as np
import ufl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
from crackcause import giardina as G  # noqa: E402
from crackcause.fx_facade import base_facets, build_mesh, cell_data  # noqa: E402
from crackcause.fx_pfczm import FxPFCZM  # noqa: E402
from dolfinx import fem, la  # noqa: E402

msh = build_mesh(10.0)
cd, _ = cell_data(msh)
model = FxPFCZM(msh, cd, 20.0, G.T, softening='linear', base_facets=base_facets(msh), kn=G.INTERFACE['kn'],
                kt=G.INTERFACE['kt'], eta_split=1e-2, eta_contact=5e-2, prox=1e-4, atol_u=6.0)
v = ufl.TestFunction(model.V)
dx = ufl.Measure('dx', domain=msh, metadata={'quadrature_degree': 1})
f_ext = fem.form(G.T * ufl.dot(model.b, v) * dx)
b = fem.assemble_vector(f_ext)
b.scatter_reverse(la.InsertMode.add)
a = b.array.reshape(-1, 2)
nodal = np.linalg.norm(a, axis=1)
print(f'sum of vertical loads {a[:, 1].sum():.1f} N; 2-norm of external force vector {np.linalg.norm(b.array):.1f} N; '
      f'max nodal force {nodal.max():.1f} N; nodes with |f| > 1 N: {(nodal > 1).sum()} of {len(nodal)}')
# equilibrium state under the initial loads: 2-norm of the internal force vector (= reactions + loads)
model.load_factor.value = 1.0
model.solve_u()
fi = fem.form(G.T * ufl.inner(ufl.sym(ufl.grad(model.u)) * 0.0, ufl.sym(ufl.grad(v))) * dx)
print('reference force norms ready')

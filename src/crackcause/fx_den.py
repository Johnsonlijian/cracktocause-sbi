"""Nooru-Mohamed (1992) double-edge-notched (DEN) mixed-mode test in dolfinx.

Specimen 200 x 200 x 50 mm with 25 x 5 mm notches at mid-height on both sides (2 mm mortar,
thesis sections 3.1-3.2, 4.1, 4.6). Loading frame (thesis Fig. 3.2-3.3): the upper L-shaped
platen (top + upper-left face) is carried by frames whose plate springs prevent rotation; the
lower platen (bottom + lower-right face) is fixed. Model: the upper platen is a stiff elastic
L-region whose top edge moves uniformly (vertical translation V, no rotation); the lateral force
Ps acts on its vertical leg as a uniform body force; the lower platen is a Dirichlet condition.
Load-path 4: Ps ramped at P = 0 (V adjusted until the vertical reaction vanishes), then axial
elongation delta = V - V(end of shear) under constant Ps. Units: mm, N, MPa.
"""
import numpy as np
from mpi4py import MPI

L, T = 200.0, 50.0
NOTCH_D, NOTCH_W = 25.0, 5.0
Y_N0, Y_N1 = 0.5 * (L - NOTCH_W), 0.5 * (L + NOTCH_W)
PLATEN = 20.0                                  # modelled platen thickness
MORTAR = dict(E=32800.0, nu=0.2, ft=3.0, Gf=0.11)
STEEL = dict(E=2.1e6, nu=0.3)                  # 10 x steel: practically rigid platen
# the side legs of the L-shaped platens cover 85 mm of the side faces (thesis Fig. 4.30d), leaving
# 12.5 mm of free face next to each notch
SIDE_LEN = 85.0
SIDE = (-PLATEN, 0.0, L - SIDE_LEN, L)         # vertical leg of the upper platen (x0, x1, y0, y1)
TOP = (-PLATEN, L, L, L + PLATEN)              # horizontal leg
# glued faces: the adhesive layer and platen compliance (not modelled) relieve the corner singularities
# of the rigid platens; damage is therefore excluded within these distances from the glued faces
NODAMAGE_TOP_BOTTOM, NODAMAGE_SIDE = 15.0, 10.0


def build_mesh(h, h_platen=None, verbose=0):
    import gmsh
    from dolfinx.io import gmsh as gio

    gmsh.initialize()
    gmsh.option.setNumber('General.Terminal', verbose)
    gmsh.model.add('den')
    occ = gmsh.model.occ
    spec = occ.addRectangle(0.0, 0.0, 0.0, L, L)
    n1 = occ.addRectangle(0.0, Y_N0, 0.0, NOTCH_D, NOTCH_W)
    n2 = occ.addRectangle(L - NOTCH_D, Y_N0, 0.0, NOTCH_D, NOTCH_W)
    body, _ = occ.cut([(2, spec)], [(2, n1), (2, n2)])
    top = occ.addRectangle(TOP[0], TOP[2], 0.0, TOP[1] - TOP[0], TOP[3] - TOP[2])
    side = occ.addRectangle(SIDE[0], SIDE[2], 0.0, SIDE[1] - SIDE[0], SIDE[3] - SIDE[2])
    occ.fragment(body, [(2, top), (2, side)])
    occ.synchronize()
    surfaces = [s[1] for s in gmsh.model.getEntities(2)]
    gmsh.model.addPhysicalGroup(2, surfaces, 1)
    # fine mesh in the specimen, coarser in the platen
    hp = h_platen or 2.0 * h
    f = gmsh.model.mesh.field.add('Box')
    gmsh.model.mesh.field.setNumber(f, 'VIn', h)
    gmsh.model.mesh.field.setNumber(f, 'VOut', hp)
    for k, v in zip(['XMin', 'XMax', 'YMin', 'YMax'], [0.0, L, 0.0, L]):
        gmsh.model.mesh.field.setNumber(f, k, v)
    gmsh.model.mesh.field.setAsBackgroundMesh(f)
    gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
    gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0)
    gmsh.option.setNumber('Mesh.Algorithm', 6)
    gmsh.model.mesh.generate(2)
    data = gio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    gmsh.finalize()
    return data.mesh


def _in(mid, r):
    x0, x1, y0, y1 = r
    return (mid[:, 0] > x0) & (mid[:, 0] < x1) & (mid[:, 1] > y0) & (mid[:, 1] < y1)


def cell_data(msh, Ps=0.0, E=MORTAR['E'], ft=MORTAR['ft'], Gf=MORTAR['Gf'], nu=MORTAR['nu']):
    """Ps (N, > 0 = compressive shear, Fig. 4.1b) as body force on the vertical platen leg."""
    tdim = msh.topology.dim
    n = msh.topology.index_map(tdim).size_local
    from dolfinx.mesh import compute_midpoints
    mid = compute_midpoints(msh, tdim, np.arange(n, dtype=np.int32))[:, :2]
    side, top = _in(mid, SIDE), _in(mid, TOP)
    plat = side | top
    x, y = mid[:, 0], mid[:, 1]
    near_glue = (y < NODAMAGE_TOP_BOTTOM) | (y > L - NODAMAGE_TOP_BOTTOM)         | ((x < NODAMAGE_SIDE) & (y > L - SIDE_LEN)) | ((x > L - NODAMAGE_SIDE) & (y < SIDE_LEN))
    vol = (SIDE[1] - SIDE[0]) * (SIDE[3] - SIDE[2]) * T
    cd = dict(E=np.where(plat, STEEL['E'], E), nu=np.where(plat, STEEL['nu'], nu),
              ft=np.where(plat, 1e3, ft), Gf=np.full(n, Gf), rho_g=np.zeros(n),
              dmg=(~plat & ~near_glue).astype(float), fx=np.where(side, Ps / vol, 0.0), fy=np.zeros(n))
    return cd, mid


def boundary_conditions(msh, Vc):
    """Return (bcs factory for FxPFCZM, unrolled y-dofs of the platen top edge)."""
    from dolfinx import fem
    from dolfinx.mesh import locate_entities_boundary
    fdim = msh.topology.dim - 1
    bottom = locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
    right = locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], L) & (x[1] < SIDE_LEN + 1e-9))
    top = locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], TOP[3]))
    fixed = np.unique(np.concatenate([bottom, right])).astype(np.int32)
    store = {}

    def factory(V):
        dofs_fix = fem.locate_dofs_topological(V, fdim, fixed)
        dofs_top = fem.locate_dofs_topological(V.sub(1), fdim, top)
        store['top_y'] = dofs_top
        return [fem.dirichletbc(np.zeros(2, dtype=np.float64), dofs_fix, V),
                fem.dirichletbc(Vc, dofs_top, V.sub(1))]
    return factory, store

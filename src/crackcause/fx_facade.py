"""gmsh mesh and cell data for the Giardina facade (and parametric variants) in dolfinx."""
import numpy as np
from mpi4py import MPI

from . import giardina as G


def _inside(xy, rects):
    m = np.zeros(len(xy), bool)
    idx = -np.ones(len(xy), int)
    for k, (x0, x1, y0, y1) in enumerate(rects):
        hit = (xy[:, 0] > x0) & (xy[:, 0] < x1) & (xy[:, 1] > y0) & (xy[:, 1] < y1)
        m |= hit
        idx[hit] = k
    return m, idx


def build_mesh(h, h_far=None, openings=G.OPENINGS, lintels=G.LINTELS, loads=G.LOADS, verbose=0):
    """Triangular mesh of the facade (conforming to openings, lintels and load plates)."""
    import gmsh
    from dolfinx.io import gmsh as gio

    gmsh.initialize()
    gmsh.option.setNumber('General.Terminal', verbose)
    gmsh.model.add('facade')
    occ = gmsh.model.occ
    outer = occ.addRectangle(0.0, 0.0, 0.0, G.W, G.H)
    holes = [occ.addRectangle(x0, y0, 0.0, x1 - x0, y1 - y0) for x0, x1, y0, y1 in openings]
    wall, _ = occ.cut([(2, outer)], [(2, s) for s in holes])
    inserts = [occ.addRectangle(x0, y0, 0.0, x1 - x0, y1 - y0) for x0, x1, y0, y1 in lintels]
    inserts += [occ.addRectangle(x - G.PLATE_W / 2, y - G.PLATE_H / 2, 0.0, G.PLATE_W, G.PLATE_H) for x, y, _ in loads]
    occ.fragment(wall, [(2, s) for s in inserts])
    occ.synchronize()
    surfaces = [s[1] for s in gmsh.model.getEntities(2)]
    gmsh.model.addPhysicalGroup(2, surfaces, 1)
    gmsh.option.setNumber('Mesh.CharacteristicLengthMax', h)
    gmsh.option.setNumber('Mesh.CharacteristicLengthMin', h * 0.5)
    gmsh.option.setNumber('Mesh.Algorithm', 6)
    gmsh.model.mesh.generate(2)
    data = gio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    gmsh.finalize()
    return data.mesh


def cell_data(msh, ft_field=None, plates=True, lintels=None):
    """Per-cell material arrays: masonry (damageable), wood lintels, load plates (masonry + load).
    `lintels` defaults to those of the tested facade (pass those of a variant from giardina.geometry)."""
    tdim = msh.topology.dim
    n = msh.topology.index_map(tdim).size_local
    from dolfinx.mesh import compute_midpoints
    mid = compute_midpoints(msh, tdim, np.arange(n, dtype=np.int32))[:, :2]
    lint, _ = _inside(mid, G.LINTELS if lintels is None else lintels)
    rects = [(x - G.PLATE_W / 2, x + G.PLATE_W / 2, y - G.PLATE_H / 2, y + G.PLATE_H / 2) for x, y, _ in G.LOADS]
    in_plate, which = _inside(mid, rects)
    m, w = G.MASONRY, G.WOOD
    cd = dict(E=np.where(lint, w['E'], m['E']), nu=np.where(lint, w['nu'], m['nu']),
              ft=np.full(n, m['ft']) if ft_field is None else np.asarray(ft_field, float),
              Gf=np.full(n, m['Gf']), rho_g=np.where(lint, w['rho_g'], m['rho_g']),
              dmg=(~lint & ~in_plate).astype(float), fx=np.zeros(n), fy=np.zeros(n))
    cd['ft'] = np.where(lint, 1e3, cd['ft'])
    if plates:
        vol = G.PLATE_W * G.PLATE_H * G.T
        for k, (_, _, F) in enumerate(G.LOADS):
            cd['fy'][which == k] = -F / vol
    return cd, mid


def base_facets(msh):
    from dolfinx.mesh import locate_entities_boundary
    return locate_entities_boundary(msh, msh.topology.dim - 1, lambda x: np.isclose(x[1], 0.0))

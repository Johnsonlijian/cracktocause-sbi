"""Structured triangular meshes on tensor-product grids with holes and notches.

Units are left to the caller (the examples use mm, N, MPa).
"""
import numpy as np


def graded_coords(a, b, h_fine, h_coarse=None, bands=(), growth=0.2):
    """1D node coordinates on [a, b].

    Spacing is h_fine inside any (lo, hi) interval of `bands` and grows linearly with
    distance (rate `growth`) up to h_coarse outside. With no bands the grid is uniform.
    """
    if h_coarse is None or not bands:
        n = max(1, int(round((b - a) / h_fine)))
        return np.linspace(a, b, n + 1)

    def spacing(x):
        dist = min(0.0 if lo <= x <= hi else min(abs(x - lo), abs(x - hi)) for lo, hi in bands)
        return min(h_coarse, h_fine + growth * dist)

    xs = [a]
    while xs[-1] < b:
        xs.append(xs[-1] + spacing(xs[-1]))
    xs = np.asarray(xs)
    return a + (xs - a) * (b - a) / (xs[-1] - a)


def structured_tri_mesh(xs, ys, keep_cell=None, diag='alternate'):
    """Triangulate the grid xs x ys; returns (nodes (N,2), tris (M,3)) counter-clockwise.

    keep_cell(xc, yc) -> bool mask over cell centres; False cells are removed (holes).
    diag='alternate' uses checkerboard diagonals; diag='cross' splits each cell into
    four triangles around a centre node, which reduces crack-path mesh bias.
    """
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    nx, ny = len(xs) - 1, len(ys) - 1
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    nodes = np.column_stack([X.ravel(), Y.ravel()])
    I, J = np.meshgrid(np.arange(nx), np.arange(ny), indexing='xy')
    I, J = I.ravel(), J.ravel()
    if keep_cell is not None:
        xc = 0.5 * (xs[I] + xs[I + 1])
        yc = 0.5 * (ys[J] + ys[J + 1])
        keep = np.asarray(keep_cell(xc, yc), bool)
        I, J = I[keep], J[keep]

    def nid(i, j):
        return j * (nx + 1) + i

    n00, n10, n01, n11 = nid(I, J), nid(I + 1, J), nid(I, J + 1), nid(I + 1, J + 1)
    if diag == 'cross':
        centres = np.column_stack([0.5 * (xs[I] + xs[I + 1]), 0.5 * (ys[J] + ys[J + 1])])
        c = nodes.shape[0] + np.arange(I.size)
        nodes = np.vstack([nodes, centres])
        tris = np.vstack([np.column_stack([n00, n10, c]), np.column_stack([n10, n11, c]),
                          np.column_stack([n11, n01, c]), np.column_stack([n01, n00, c])])
    else:
        flip = ((I + J) % 2 == 1)[:, None]
        t1 = np.where(flip, np.column_stack([n00, n10, n01]), np.column_stack([n00, n10, n11]))
        t2 = np.where(flip, np.column_stack([n10, n11, n01]), np.column_stack([n00, n11, n01]))
        tris = np.vstack([t1, t2])
    used = np.unique(tris)
    remap = -np.ones(nodes.shape[0], dtype=np.int64)
    remap[used] = np.arange(used.size)
    return nodes[used], remap[tris]


def nodes_where(nodes, cond):
    """Indices of nodes satisfying cond(x, y)."""
    return np.flatnonzero(cond(nodes[:, 0], nodes[:, 1]))

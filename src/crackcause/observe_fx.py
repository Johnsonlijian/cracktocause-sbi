"""Observation operator for FEniCSx runs: P1 phase field -> fixed-frame crack-line map.

Pipeline: rasterize d on a regular grid (linear interpolation on the triangulation) ->
band = d >= threshold inside the wall -> skeleton (crack centre lines) -> optional dilation.
The same map format is produced from digitized experimental crack drawings, so simulated and
observed crack patterns are directly comparable.
"""
import matplotlib.tri as mtri
import numpy as np
from scipy import ndimage

try:
    from skimage.morphology import skeletonize
except ImportError:          # pragma: no cover
    skeletonize = None


class Rasterizer:
    """Precomputes the interpolation from P1 nodal values to a (ny, nx) pixel grid."""

    def __init__(self, coords, cells, bbox, shape):
        x0, x1, y0, y1 = bbox
        ny, nx = shape
        self.tri = mtri.Triangulation(coords[:, 0], coords[:, 1], cells)
        self.finder = self.tri.get_trifinder()
        xs = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
        ys = y1 - (np.arange(ny) + 0.5) * (y1 - y0) / ny          # row 0 = top
        self.X, self.Y = np.meshgrid(xs, ys)
        self.shape = shape
        tri_idx = self.finder(self.X, self.Y)
        self.inside = tri_idx >= 0
        t = tri_idx[self.inside]
        v = cells[t]
        p = np.stack([self.X[self.inside], self.Y[self.inside]], 1)
        a, b, c = coords[v[:, 0]], coords[v[:, 1]], coords[v[:, 2]]
        det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
        l1 = ((b[:, 0] - p[:, 0]) * (c[:, 1] - p[:, 1]) - (c[:, 0] - p[:, 0]) * (b[:, 1] - p[:, 1])) / det
        l2 = ((c[:, 0] - p[:, 0]) * (a[:, 1] - p[:, 1]) - (a[:, 0] - p[:, 0]) * (c[:, 1] - p[:, 1])) / det
        self.verts, self.w = v, np.stack([l1, l2, 1.0 - l1 - l2], 1)

    def __call__(self, nodal):
        img = np.zeros(self.shape)
        img[self.inside] = (nodal[self.verts] * self.w).sum(1)
        return img


def crack_lines(d_img, inside, threshold=0.9, dilate=0):
    band = (d_img >= threshold) & inside
    if skeletonize is not None:
        lines = skeletonize(band)
    else:
        lines = band
    if dilate:
        lines = ndimage.binary_dilation(lines, iterations=dilate)
    return lines.astype(np.uint8)


def thin(lines):
    """Final step of the observation operator on the map grid: one-pixel-wide crack lines. Applied
    to the digitized experimental drawings, whose strokes are binned onto the grid and can be two
    pixels wide, so that observed and simulated maps measure crack length in the same way."""
    lines = np.asarray(lines) > 0
    return (skeletonize(lines) if skeletonize is not None else lines).astype(np.uint8)


def load_exp_maps(path):
    """Digitized experimental crack maps {key: thinned (ny, nx) uint8 map} from a data_public npz."""
    ex = np.load(path)
    return {k: thin(ex[k]) for k in ex.files if ex[k].ndim == 2}


class PointInterpolator:
    """Linear interpolation of P1 nodal fields (N,) or (N, k) at arbitrary points (NaN outside the mesh)."""

    def __init__(self, coords, cells):
        self.coords, self.cells = coords, cells
        self.finder = mtri.Triangulation(coords[:, 0], coords[:, 1], cells).get_trifinder()

    def __call__(self, nodal, P):
        t = self.finder(P[:, 0], P[:, 1])
        ok = t >= 0
        out = np.full((len(P),) + nodal.shape[1:], np.nan)
        v = self.cells[t[ok]]
        a, b, c = self.coords[v[:, 0]], self.coords[v[:, 1]], self.coords[v[:, 2]]
        p = P[ok]
        det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
        l1 = ((b[:, 0] - p[:, 0]) * (c[:, 1] - p[:, 1]) - (c[:, 0] - p[:, 0]) * (b[:, 1] - p[:, 1])) / det
        l2 = ((c[:, 0] - p[:, 0]) * (a[:, 1] - p[:, 1]) - (a[:, 0] - p[:, 0]) * (c[:, 1] - p[:, 1])) / det
        lam = np.stack([l1, l2, 1.0 - l1 - l2], 1)
        if nodal.ndim == 1:
            out[ok] = (nodal[v] * lam).sum(1)
        else:
            out[ok] = (nodal[v] * lam[:, :, None]).sum(1)
        return out, ok


def crack_widths(lines, ras, interp, u, deltas=(35.0, 28.0, 21.0, 14.0), radius=2.0):
    """Crack opening (mm) at every pixel of a crack-line map: the jump of the displacement u (N, 2) normal
    to the crack between two points at +-delta from the centre line, delta being the first of `deltas`
    for which both points lie in the structure. The normal is the minor principal direction of the
    crack pixels within `radius` pixels. The band of the PF-CZM (alpha = 2d - d^2) has half-width
    pi*ell/2 (31 mm for ell = 20 mm), so delta = 35 mm spans the whole band; the elastic stretch over 2*delta
    is of order 1e-3 mm. Returns a float32 (ny, nx) map (0 off the lines, NaN where no estimate)."""
    from scipy.spatial import cKDTree
    W = np.zeros(lines.shape, np.float32)
    iy, ix = np.nonzero(lines)
    if len(iy) == 0:
        return W
    pts = np.stack([ix, -iy], 1).astype(float)                # pixel coordinates, y up
    nrm = np.full((len(pts), 2), np.nan)
    for k, nb in enumerate(cKDTree(pts).query_ball_point(pts, radius)):
        if len(nb) < 2:
            continue
        q = pts[nb] - pts[nb].mean(0)
        _, V = np.linalg.eigh(q.T @ q)
        nrm[k] = V[:, 0]                                       # minor axis = normal to the line
    X = np.stack([ras.X[iy, ix], ras.Y[iy, ix]], 1)
    w = np.full(len(pts), np.nan)
    for dl in deltas:
        todo = np.isnan(w) & ~np.isnan(nrm[:, 0])
        if not todo.any():
            break
        up, okp = interp(u, X[todo] + dl * nrm[todo])
        um, okm = interp(u, X[todo] - dl * nrm[todo])
        jump = ((up - um) * nrm[todo]).sum(1)
        good = okp & okm
        idx = np.flatnonzero(todo)[good]
        w[idx] = np.maximum(jump[good], 0.0)
    W[iy, ix] = w
    return W

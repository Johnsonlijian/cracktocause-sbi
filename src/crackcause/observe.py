"""Observation operator: phase field on the mesh -> fixed-frame crack map image.

First version: rasterize d, threshold the crack band, and optionally apply inspection-style
corruption (occlusion, false positives/negatives). Width channel and skeletonization are planned.
"""
import matplotlib.tri as mtri
import numpy as np


def rasterize(nodes, tris, values, bbox, shape):
    """Linear interpolation of nodal values onto a pixel grid (rows = y descending)."""
    x0, x1, y0, y1 = bbox
    ny, nx = shape
    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], tris)
    interp = mtri.LinearTriInterpolator(tri, values)
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y1, y0, ny)
    X, Y = np.meshgrid(xs, ys)
    img = interp(X, Y)
    return np.ma.filled(img, np.nan)


def crack_map(nodes, tris, d, bbox, shape=(96, 160), threshold=0.9):
    """Binary crack map (1 = crack band), NaN-free; openings and outside are 0."""
    img = rasterize(nodes, tris, d, bbox, shape)
    return (np.nan_to_num(img, nan=0.0) >= threshold).astype(np.float32)


def corrupt(img, rng, p_occlude=0.5, max_boxes=3, box_frac=(0.05, 0.2), p_flip=0.002):
    """Inspection-style corruption: random occluding boxes and salt noise."""
    out = img.copy()
    ny, nx = out.shape
    if rng.random() < p_occlude:
        for _ in range(rng.integers(1, max_boxes + 1)):
            h = int(ny * rng.uniform(*box_frac))
            w = int(nx * rng.uniform(*box_frac))
            r0, c0 = rng.integers(0, ny - h), rng.integers(0, nx - w)
            out[r0:r0 + h, c0:c0 + w] = 0.0
    flips = rng.random(out.shape) < p_flip
    out[flips] = 1.0 - out[flips]
    return out

"""Mechanism cracks of the facade crack maps (10 mm pixels) and the two hinge lines they form on."""
import numpy as np
from scipy import ndimage

PIX = 10.0
HINGES = dict(H1=640.0, H2=990.0)   # positions of the mechanism-crack clusters in the facade campaign (mm)


def main_crack(m):
    """(x in mm, vertical extent as a fraction of the height) of the crack component (eight-neighbour
    connectivity) with the largest vertical extent; x is the median abscissa of its pixels."""
    lab, n = ndimage.label(m > 0, structure=np.ones((3, 3)))
    best, ext = 0, 0
    for k in range(1, n + 1):
        rows = np.nonzero(lab == k)[0]
        e = rows.max() - rows.min() + 1
        if e > ext:
            ext, best = e, k
    if best == 0:
        return np.nan, 0.0
    return (np.median(np.nonzero(lab == best)[1]) + 0.5) * PIX, ext / m.shape[0]


def band_hinge(m, min_span=0.3, tol=60.0):
    """Hinge line whose band (|x - x_h| <= tol) contains crack pixels spanning at least min_span of the
    height, also when the crack line is interrupted by openings or drawn in segments; the line with the
    larger span if both qualify, 'none' if neither. Returns (hinge, {line: span})."""
    rows, cols = np.nonzero(m)
    x = (cols + 0.5) * PIX
    span = {}
    for name, xh in HINGES.items():
        r = rows[np.abs(x - xh) <= tol]
        span[name] = float((r.max() - r.min() + 1) / m.shape[0]) if len(r) else 0.0
    best = max(span, key=span.get)
    return (best if span[best] >= min_span else 'none'), span


def hinge_of(x, ext, min_extent=0.3, tol=60.0):
    """'H1'/'H2' for a mechanism crack (extent >= min_extent) within tol of a hinge line, 'other' for a
    mechanism crack elsewhere, 'none' without a mechanism crack."""
    if not (np.isfinite(x) and ext >= min_extent):
        return 'none'
    for name, xh in HINGES.items():
        if abs(x - xh) <= tol:
            return name
    return 'other'

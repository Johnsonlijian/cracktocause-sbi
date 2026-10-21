"""Mesh and length-scale sensitivity of the observation: crack maps and widest crack opening of reference
facade runs at common stages, against a reference run.

usage: python compare_mesh.py REFERENCE.npz OTHER.npz [OTHER.npz ...]
For every stage recorded in both runs: crack pixels, w_max, and the symmetric chamfer distance (mm)
and the fraction of crack pixels within 30 mm of the other map (precision and recall with respect to
the reference run).
"""
import os
import sys

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
sys.path.insert(0, os.path.join(HERE, '..', 'scripts'))
from crackcause import giardina as G  # noqa: E402
from crackcause.observe_fx import PointInterpolator, Rasterizer, crack_lines, crack_widths  # noqa: E402
from train_npe import max_width  # noqa: E402

PIX = 10.0                                                # mm per pixel


def observe(path):
    z = np.load(path)
    c, cells, dmg = z['coords'], z['cells'], z['dmg']
    ras = Rasterizer(c, cells[dmg > 0.5], (0.0, G.W, 0.0, G.H), (119, 143))
    ip = PointInterpolator(c, cells)
    out = {}
    for S, d, u in zip(z['S'], z['d'], z['u']):
        lines = crack_lines(ras(d), ras.inside, 0.9)
        out[round(float(S), 3)] = (lines > 0, max_width(crack_widths(lines, ras, ip, u.reshape(-1, 2)), lines))
    return out


ref = observe(sys.argv[1])
print('reference', os.path.basename(sys.argv[1]))
for path in sys.argv[2:]:
    oth = observe(path)
    print('==', os.path.basename(path))
    for S in sorted(set(ref) & set(oth)):
        (a, wa), (b, wb) = ref[S], oth[S]
        if not a.any() and not b.any():
            print(f'  S = {S:5.2f}: no cracks in either')
            continue
        if not a.any() or not b.any():
            print(f'  S = {S:5.2f}: cracks in one map only ({int(a.sum())} / {int(b.sum())} px)')
            continue
        da, db = ndimage.distance_transform_edt(~a) * PIX, ndimage.distance_transform_edt(~b) * PIX
        prec, rec = float((da[b] <= 30.0).mean()), float((db[a] <= 30.0).mean())
        ch = 0.5 * (da[b].mean() + db[a].mean())
        print(f'  S = {S:5.2f}: px {int(a.sum()):4d} / {int(b.sum()):4d}, w_max {wa:5.2f} / {wb:5.2f} mm, '
              f'chamfer {ch:5.1f} mm, precision {prec:.2f}, recall {rec:.2f}')

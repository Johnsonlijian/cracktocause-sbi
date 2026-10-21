"""Widest crack opening of reference simulations at the stages of the facade test, against the widest
opening measured in the test (Giardina et al. 2013, Fig. 7).

usage: python compare_widths.py RESULT.npz [RESULT.npz ...]
RESULT.npz: output of fx_giardina.py (records S, d, u). Prints w_max at every recorded stage and at the
experimental stages, together with the measured value.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
sys.path.insert(0, os.path.join(HERE, '..', 'scripts'))
from crackcause import giardina as G  # noqa: E402
from crackcause.observe_fx import PointInterpolator, Rasterizer, crack_lines, crack_widths  # noqa: E402
from train_npe import CONFIGS, exp_max_widths, max_width  # noqa: E402

cfg = CONFIGS['facade']
w_exp = exp_max_widths(os.path.join(HERE, '..', cfg['exp_widths']), cfg['exp_A'])
S_exp = {k: v[0] for k, v in cfg['exp']}
for path in sys.argv[1:]:
    z = np.load(path)
    c, cells, dmg = z['coords'], z['cells'], z['dmg']
    ras = Rasterizer(c, cells[dmg > 0.5], (0.0, G.W, 0.0, G.H), cfg['shape'])
    ip = PointInterpolator(c, cells)
    print('==', os.path.basename(path))
    for S, d, u in zip(z['S'], z['d'], z['u']):
        lines = crack_lines(ras(d), ras.inside, 0.9)
        wm = max_width(crack_widths(lines, ras, ip, u.reshape(-1, 2)), lines)
        key = [k for k, s in S_exp.items() if abs(s - S) < 1e-6]
        tag = f'   <- {key[0]}: measured {w_exp[key[0]]}' if key else ''
        print(f'  S = {S:5.2f} mm: {int(lines.sum()):4d} crack px, w_max = {wm:5.2f} mm{tag}')

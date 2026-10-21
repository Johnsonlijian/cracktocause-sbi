"""Geometry, loads and settlement profile of the 1/10-scale masonry facade test
(Giardina et al. 2012, Eng. Struct. 45; numerical model in Giardina et al. 2013, Eng. Struct. 54).

Units: mm, N, MPa. Values read from Fig. 3 and Tables 1-2 of the 2013 paper; load-plate
positions estimated from the dimensioned drawing (about +/-5 mm).
"""
import os

import numpy as np

W, H, T = 1428.0, 1186.0, 50.0                    # facade width, height, thickness

_WIN_X = [(84.0, 210.0), (294.0, 420.0), (504.0, 630.0), (798.0, 924.0), (1008.0, 1134.0), (1218.0, 1344.0)]
OPENINGS = ([(x0, x1, 862.0, 1080.0) for x0, x1 in _WIN_X]            # top storey windows
            + [(x0, x1, 538.0, 756.0) for x0, x1 in _WIN_X]           # middle storey windows
            + [(84.0, 210.0, 106.0, 432.0), (798.0, 924.0, 106.0, 432.0),   # tall ground-floor openings
               (294.0, 630.0, 214.0, 432.0), (1008.0, 1344.0, 214.0, 432.0)])  # wide ground-floor windows
LINTEL_DEPTH, LINTEL_OVERHANG = 25.0, 21.0
LINTELS = [(x0 - LINTEL_OVERHANG, x1 + LINTEL_OVERHANG, y1, y1 + LINTEL_DEPTH) for x0, x1, _, y1 in OPENINGS]

# (x, y, F[N]) of the 35 vertical loads (Table 1: F1..F10), applied on 20 x 12 mm plates
_F = dict(F1=230.0, F2=296.0, F3=319.0, F4=392.0, F5=358.0, F6=469.0, F7=195.0, F8=498.0, F9=304.0, F10=467.0)
LOADS = (
    [(x, 1146.0, _F[k]) for x, k in zip([42, 252, 462, 672, 756, 966, 1176, 1386],
                                          ['F1', 'F2', 'F2', 'F1', 'F1', 'F2', 'F2', 'F1'])]
    + [(x, 824.0, _F[k]) for x, k in zip([96, 306, 516, 618, 810, 1020, 1230, 1332],
                                           ['F3', 'F4', 'F4', 'F3', 'F3', 'F4', 'F4', 'F3'])]
    + [(20.0, 470.0, _F['F5']), (1408.0, 470.0, _F['F5'])]
    + [(x, 500.0, _F[k]) for x, k in zip([210, 378, 546, 712, 922, 1092, 1260],
                                           ['F6', 'F7', 'F7', 'F8', 'F6', 'F7', 'F7'])]
    + [(126.0, 64.0, _F['F9']), (839.0, 64.0, _F['F9'])]
    + [(x, 147.0, _F['F10']) for x in [316, 399, 524, 609, 1030, 1113, 1240, 1323]]
)
PLATE_W, PLATE_H = 20.0, 12.0

MASONRY = dict(E=3000.0, nu=0.2, rho_g=1.9e-9 * 9810.0, ft=0.1, Gf=0.01)   # Table 2
WOOD = dict(E=11000.0, nu=0.15, rho_g=0.5e-9 * 9810.0)
INTERFACE = dict(kn=0.7, kt=1e-3)                  # N/mm^3; kt ~ 0 in the paper (1 N/m^3)

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROFILE_CSV = os.path.join(_HERE, '..', '..', 'data_public', 'giardina2013_fig8_applied_profiles.csv')


# Fig. 5 crack-map stages (nominal 2.5, 5, 7.5, 10 mm) expressed as S = settlement at x = 30 mm of the
# digitized applied profiles (Fig. 8a-d)
EXP_STAGES_NOMINAL = (2.5, 5.0, 7.5, 10.0)
EXP_STAGES_S = (2.26, 4.54, 6.81, 9.10)


def geometry(variant='test'):
    """Openings and lintels of the facade as tested or of a variant for numerical interventions:
    'solid' has no openings and no lintels; 'piers<d>' widens the two piers that flank the central door by d mm
    on their outer sides (the edges at x = 630 move to 630 - d, those at x = 1008 to 1008 + d; window widths
    shrink accordingly, everything else is unchanged)."""
    if variant == 'test':
        return list(OPENINGS), list(LINTELS)
    if variant == 'solid':
        return [], []
    if variant.startswith('piers'):
        d = float(variant[len('piers'):])
        ops = [(x0 + d if x0 == 1008.0 else x0, x1 - d if x1 == 630.0 else x1, y0, y1) for x0, x1, y0, y1 in OPENINGS]
        return ops, [(x0 - LINTEL_OVERHANG, x1 + LINTEL_OVERHANG, y1, y1 + LINTEL_DEPTH) for x0, x1, _, y1 in ops]
    raise ValueError(f'unknown facade variant {variant!r}')


def settlement_shape():
    """Normalized measured profile phi(x) with phi(30) = -1 (from Fig. 8d, S = 10 mm)."""
    data = np.genfromtxt(_PROFILE_CSV, delimiter=',', names=True)
    x, v = data['x_mm'], data['v_d_mm']
    return x, v / -np.interp(30.0, x, v)


def check():
    tot = sum(F for _, _, F in LOADS)
    assert len(LOADS) == 35, len(LOADS)
    assert abs(tot - 12224.0) < 1.0, tot
    for x0, x1, y0, y1 in OPENINGS:
        assert 0 < x0 < x1 < W and 0 < y0 < y1 < H
    for x, y, _ in LOADS:
        for x0, x1, y0, y1 in OPENINGS:
            assert not (x0 < x < x1 and y0 < y < y1), (x, y)
    return tot


if __name__ == '__main__':
    print('total vertical load', check(), 'N in', len(LOADS), 'points;', len(OPENINGS), 'openings')
    xs, phi = settlement_shape()
    print('phi(0)=%.3f phi(800)=%.3f phi(1000)=%.3f phi(1428)=%.3f' % tuple(np.interp([0, 800, 1000, 1428], xs, phi)))

"""Digitize the 'applied' (cyan) settlement profiles in Giardina et al. (2013) Fig. 8.

usage: python -I -X utf8 digitize_giardina_fig8.py FIG8_PNG OUT_CSV
Each panel's axes frame is detected from dark pixels; x is mapped to [0, 1428] mm (facade width)
and y to [-10, +1] mm (axis limits printed on the figure).
"""
import sys

import matplotlib.image as mpimg
import numpy as np

img = mpimg.imread(sys.argv[1])[:, :, :3]
if img.max() <= 1.0:
    img = img * 255.0
R, G, B = img[..., 0], img[..., 1], img[..., 2]
cyan = (R < 150) & (G > 150) & (B > 170) & ((B - R) > 50)
dark = (R < 80) & (G < 80) & (B < 80)
H, W = R.shape


def frame(r0, r1, c0, c1):
    """Axes frame inside a panel window: long dark horizontal/vertical lines."""
    sub = dark[r0:r1, c0:c1]
    rows = np.where(sub.sum(axis=1) > 0.6 * (c1 - c0) * 0.5)[0]
    cols = np.where(sub.sum(axis=0) > 0.6 * (r1 - r0) * 0.5)[0]
    return r0 + rows.min(), r0 + rows.max(), c0 + cols.min(), c0 + cols.max()


panels = {'a': (0, H // 2, 0, W // 2), 'b': (0, H // 2, W // 2, W),
          'c': (H // 2, H, 0, W // 2), 'd': (H // 2, H, W // 2, W)}
out = []
for name, (r0, r1, c0, c1) in panels.items():
    top, bot, left, right = frame(r0, r1, c0, c1)
    xs, ys = [], []
    for c in range(left + 2, right - 1):
        rr = np.where(cyan[top:bot, c])[0]
        if rr.size:
            # keep the top-most contiguous run: the legend sample line lies lower in the panel
            breaks = np.where(np.diff(rr) > 2)[0]
            run = rr[:breaks[0] + 1] if breaks.size else rr
            xs.append((c - left) / (right - left) * 1428.0)
            ys.append(1.0 - run.mean() / (bot - top) * 11.0)
    xs, ys = np.array(xs), np.array(ys)
    grid = np.arange(0.0, 1429.0, 25.0)
    yi = np.interp(grid, xs, ys)
    out.append((name, grid, yi))
    print(name, f'frame rows {top}-{bot} cols {left}-{right}; v(0)={yi[0]:.2f} v(30)={np.interp(30, grid, yi):.2f} '
          f'zero-crossing ~{grid[np.argmax(yi > -0.05)]:.0f} mm, max {yi.max():.2f}')

with open(sys.argv[2], 'w') as fh:
    fh.write('x_mm,' + ','.join(f'v_{n}_mm' for n, _, _ in out) + '\n')
    for k in range(out[0][1].size):
        fh.write(f'{out[0][1][k]:.0f},' + ','.join(f'{o[2][k]:.4f}' for o in out) + '\n')
print('saved', sys.argv[2])

"""Digitize the experimental crack patterns in Giardina et al. (2013) Fig. 5 (right column).

usage: python -I -X utf8 digitize_giardina_fig5.py PAPER_PDF OUT_DIR [OUT_NPZ]
Renders the figure column at 600 dpi, finds each panel's facade outline (gray rectangle),
maps pixels to facade coordinates (0..1428 x 0..1186 mm), keeps black strokes (cracks), removes
text/arrow components, and writes crack maps on a 10 mm grid plus overlay images for checking
(and, with OUT_NPZ, the data_public file with keys S0, S2.5, S5, S7.5, S10). The crack label "3",
drawn with thinner strokes than the other labels, passes the thickness test in three panels and is
removed by its position (components lying entirely inside LABEL_BOXES_MM).
"""
import os
import sys

import fitz
import numpy as np
from scipy import ndimage

W_MM, H_MM = 1428.0, 1186.0
STAGES = [0.0, 2.5, 5.0, 7.5, 10.0]
LABEL_BOXES_MM = [(840.0, 940.0, 420.0, 550.0)]      # (x0, x1, y0, y1) of crack label "3"

pdf, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
doc = fitz.open(pdf)
pg = doc[3]
r = pg.rect
clip = fitz.Rect(r.x0 + 0.735 * r.width, r.y0 + 0.085 * r.height, r.x1 - 0.005 * r.width, r.y0 + 0.715 * r.height)
pix = pg.get_pixmap(dpi=600, clip=clip)
img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].astype(int)
R, G, B = img[..., 0], img[..., 1], img[..., 2]
gray = (np.abs(R - G) < 18) & (np.abs(G - B) < 18) & (R > 95) & (R < 215)
dark = (R < 90) & (G < 90) & (B < 90)

# panel outlines: long gray horizontal runs give top/bottom edges, vertical runs give sides
row_frac = gray.sum(1) / img.shape[1]
rows = np.where(row_frac > 0.45)[0]
groups = np.split(rows, np.where(np.diff(rows) > 5)[0] + 1)
hlines = [int(g.mean()) for g in groups if g.size]
print('horizontal outline rows:', hlines)
pairs, used = [], set()
for i, top in enumerate(hlines):          # pair each top edge with the line one panel-height below
    if i in used:
        continue
    for j in range(i + 1, len(hlines)):
        if 550 < hlines[j] - top < 650 and j not in used:
            pairs.append((top, hlines[j]))
            used.update({i, j})
            break
panels = []
for top, bot in pairs:
    band = gray[top:bot]
    col_frac = band.sum(0) / (bot - top)
    cols = np.where(col_frac > 0.6)[0]
    cg = np.split(cols, np.where(np.diff(cols) > 5)[0] + 1)
    vlines = [int(g.mean()) for g in cg if g.size]
    panels.append((top, bot, vlines[0], vlines[-1]))
print('panels (top, bottom, left, right):', panels)
assert len(panels) == len(STAGES), len(panels)

nx, ny = 143, 119                        # 10 mm pixels in facade coordinates
summary = {}
for (top, bot, left, right), S in zip(panels, STAGES):
    sub = dark[top - 3:bot + 4, left - 3:right + 4].copy()
    lab, n = ndimage.label(sub, structure=np.ones((3, 3)))
    keep = np.zeros_like(sub)
    dist = ndimage.distance_transform_edt(sub)
    objs = ndimage.find_objects(lab)
    for k, sl in enumerate(objs, start=1):
        m = lab[sl] == k
        dk = dist[sl][m]
        # font glyphs and arrows are drawn with thicker strokes than the crack lines
        text_like = dk.max() >= 2.9 or np.median(dk) >= 1.9
        yy_k, xx_k = np.nonzero(m)
        xk = (xx_k + sl[1].start - 3) / (right - left) * W_MM
        yk = H_MM - (yy_k + sl[0].start - 3) / (bot - top) * H_MM
        label = any(x0 <= xk.min() and xk.max() <= x1 and y0 <= yk.min() and yk.max() <= y1
                    for x0, x1, y0, y1 in LABEL_BOXES_MM)
        if not text_like and not label:
            keep[sl][m] = True
    yy, xx = np.nonzero(keep)
    x_mm = (xx - 3) / (right - left) * W_MM
    y_mm = H_MM - (yy - 3) / (bot - top) * H_MM
    grid = np.zeros((ny, nx), np.uint8)
    ix = np.clip((x_mm / W_MM * nx).astype(int), 0, nx - 1)
    iy = np.clip(((H_MM - y_mm) / H_MM * ny).astype(int), 0, ny - 1)
    grid[iy, ix] = 1
    np.save(os.path.join(out_dir, f'exp_crackmap_S{S:g}.npy'), grid)
    summary[S] = int(grid.sum())
    # overlay for visual check
    ov = img[top - 3:bot + 4, left - 3:right + 4].copy()
    ov[keep] = [255, 0, 255]
    fitz.Pixmap(fitz.csRGB, ov.shape[1], ov.shape[0], np.ascontiguousarray(ov.astype(np.uint8)).tobytes(),
                False).save(os.path.join(out_dir, f'exp_overlay_S{S:g}.png'))
print('crack pixels per stage (10 mm grid):', summary)
if len(sys.argv) > 3:
    # the 0 mm panel is the undamaged facade drawn in black outlines: no cracks by definition
    maps = {f'S{S:g}': (np.load(os.path.join(out_dir, f'exp_crackmap_S{S:g}.npy')) if S > 0
                        else np.zeros((ny, nx), np.uint8)) for S in STAGES}
    np.savez_compressed(sys.argv[3], **maps,
                        grid_mm=10.0, note='Giardina et al. 2013 Fig.5 experimental crack maps, 10 mm grid, row 0 = top')
    print('written', sys.argv[3])

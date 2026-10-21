"""Digitize the experimental crack-width curves of Giardina et al. (2013) Fig. 7 (cracks 1-6).

usage: python -I -X utf8 digitize_giardina_fig7.py PAPER_PDF OUT_CSV [CHECK_DIR]
Renders the figure at 400 dpi, locates the six axes frames (x: applied displacement A = 4..12 mm,
y: crack width 0..12 mm), keeps the long connected dark strokes inside each frame (the solid
experimental curve; the dashed numerical curve consists of short dashes, the inset sketch is masked),
and samples them every 0.25 mm of A where they exist. Writes crack, A_mm, width_mm rows.
"""
import os
import sys

import fitz
import numpy as np
from scipy import ndimage

pdf, out_csv = sys.argv[1], sys.argv[2]
check = sys.argv[3] if len(sys.argv) > 3 else None
doc = fitz.open(pdf)
pg = doc[4]
r = pg.rect
clip = fitz.Rect(r.x0 + 0.12 * r.width, r.y0 + 0.315 * r.height, r.x1 - 0.12 * r.width, r.y0 + 0.66 * r.height)
pix = pg.get_pixmap(dpi=400, clip=clip)
img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].astype(int)
dark = img.max(2) < 110
H, W = dark.shape


def runs(idx):
    g = np.split(idx, np.where(np.diff(idx) > 3)[0] + 1)
    return [int(np.round(x.mean())) for x in g if len(x)]


rows = runs(np.where(dark.sum(1) > 0.18 * W)[0])
cols = runs(np.where(dark.sum(0) > 0.25 * H)[0])
# frame edges: (top, bottom) per panel row, (left, right) per panel column; skip tick/label rows
frame_rows = [(t, b) for t, b in zip(rows[:-1], rows[1:]) if 400 < b - t < 480]
frame_cols = [(lft, rgt) for lft, rgt in zip(cols[:-1], cols[1:]) if 540 < rgt - lft < 620]
assert len(frame_rows) == 2 and len(frame_cols) == 3, (rows, cols)
out = ['crack,A_mm,width_mm']
crack = 0
for (top, bot) in frame_rows:
    for (lft, rgt) in frame_cols:
        crack += 1
        sub = dark[top + 4:bot - 3, lft + 4:rgt - 3].copy()
        ys = np.arange(top + 4, bot - 3)
        xs = np.arange(lft + 4, rgt - 3)
        xd = 4.0 + 8.0 * (xs - lft) / (rgt - lft)
        yd = 12.0 * (bot - ys) / (bot - top)
        sub[np.ix_(yd > 5.4, xd < 8.4)] = False          # inset sketch (top left)
        lab, n = ndimage.label(sub, structure=np.ones((3, 3)))
        keep = np.zeros_like(sub)
        for k, sl in enumerate(ndimage.find_objects(lab), start=1):
            if sl[1].stop - sl[1].start > 0.6 / 8.0 * (rgt - lft):     # longer than 0.6 mm of A
                keep[sl][lab[sl] == k] = True
        for A in np.arange(4.0, 12.01, 0.25):
            j = np.argmin(np.abs(xd - A))
            hit = np.flatnonzero(keep[:, max(j - 1, 0):j + 2].any(1))
            if len(hit):
                out.append(f'{crack},{A:.2f},{float(np.median(yd[hit])):.2f}')
        if check:
            ov = img[top:bot, lft:rgt].copy()
            ov[4:-3, 4:-3][keep] = [255, 0, 255]
            fitz.Pixmap(fitz.csRGB, ov.shape[1], ov.shape[0], np.ascontiguousarray(ov.astype(np.uint8)).tobytes(),
                        False).save(os.path.join(check, f'fig7_crack{crack}.png'))
open(out_csv, 'w').write('\n'.join(out) + '\n')
print(f'{len(out) - 1} points written to {out_csv}')

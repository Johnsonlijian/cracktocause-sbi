"""Digitize the load-path 4 crack patterns of Nooru-Mohamed (1992), thesis Fig. 4.28 (p. 90).

usage: python -I -X utf8 digitize_nooru_fig428.py THESIS_PDF OUT_DIR

The scanned page (PDF page 102, embedded 300 dpi image) shows three DEN specimens (200 mm):
(a) 48-03 load-path 4a (Ps = 5 kN), (b) 46-05 4b (10 kN), (c) 47-06 4c (27.5 kN, maximum shear).
Front-face cracks are drawn as solid lines and rear-face cracks as dashed lines.

Steps per panel: binarize; register with the specimen side lines (x = 0, 200 mm) and the notch
edges (notch centre y = 100 mm), isotropic scale; remove the outline, the letters/labels and the
label leader arrows (rectangles and segments below, read off the scan); skeletonize the remaining
strokes; skeleton components longer than 13 mm are solid (front face), shorter ones are dashes
(rear face, bridged by a 1.2 mm dilation before skeletonizing). Output maps use 2 mm pixels on the
200 x 200 mm specimen (row 0 = top), identical in format to the simulated observation operator.
"""
import os
import sys

import pymupdf as fitz
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

L_MM, NOTCH_MM = 200.0, 25.0
GRID = 100                                             # 2 mm pixels
FIG_BOX = (0.395, 0.865, 0.20, 0.86)                   # figure crop (fractions of the page image)

# per panel, in figure-crop pixel coordinates: band rows, rectangles (x0, x1, y0, y1) and
# leader segments ((x0, y0), (x1, y1)) to remove
PANELS = {
    '4a': dict(band=(56, 327), Ps=5.0, specimen='48-03',
               rects=[(273, 296, 87, 115), (828, 850, 88, 117), (291, 315, 218, 246), (744, 767, 220, 248),
                      (480, 665, 50, 88), (250, 420, 252, 286), (548, 930, 287, 327), (446, 582, 86, 94),
                      (248, 438, 284, 293)],
               leaders=[((451, 90), (441, 132)), ((434, 288), (446, 233))]),
    '4b': dict(band=(402, 786), Ps=10.0, specimen='46-05',
               rects=[(315, 338, 429, 458), (297, 320, 646, 673), (829, 853, 546, 573), (788, 811, 689, 718),
                      (682, 855, 404, 434), (250, 415, 702, 732), (548, 930, 742, 786), (640, 855, 434, 442),
                      (248, 437, 730, 739)],
               leaders=[((644, 438), (632, 480)), ((433, 734), (444, 690))]),
    '4c': dict(band=(854, 1314), Ps=27.5, specimen='47-06',
               rects=[(250, 272, 984, 1011), (248, 260, 1162, 1189), (823, 845, 1017, 1042), (825, 842, 1184, 1211),
                      (645, 810, 857, 886), (248, 405, 1238, 1266), (548, 930, 1276, 1314), (637, 830, 886, 894),
                      (248, 420, 1266, 1274)],
               leaders=[((641, 890), (630, 935)), ((415, 1269), (426, 1234))]),
}


def segment_mask(shape, p0, p1, half_width=4):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    p0, p1 = np.array(p0, float), np.array(p1, float)
    d = p1 - p0
    t = np.clip(((xx - p0[0]) * d[0] + (yy - p0[1]) * d[1]) / (d @ d), 0, 1)
    dist = np.hypot(xx - (p0[0] + t * d[0]), yy - (p0[1] + t * d[1]))
    return dist <= half_width


def to_grid(ys, xs, reg):
    xl, scale, yc = reg
    x_mm = (xs - xl) / scale
    y_mm = 100.0 + (yc - ys) / scale
    ok = (x_mm >= 0) & (x_mm < L_MM) & (y_mm >= 0) & (y_mm < L_MM)
    g = np.zeros((GRID, GRID), np.uint8)
    g[((L_MM - y_mm[ok]) / 2.0).astype(int), (x_mm[ok] / 2.0).astype(int)] = 1
    return g


def main():
    pdf, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf)
    xref = doc[101].get_images()[0][0]
    pix = fitz.Pixmap(doc, xref)
    if pix.n > 1:
        pix = fitz.Pixmap(fitz.csGRAY, pix)
    page = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
    H, W = page.shape
    fig = page[int(FIG_BOX[0] * H):int(FIG_BOX[1] * H), int(FIG_BOX[2] * W):int(FIG_BOX[3] * W)]
    dark_all = fig < 128
    result, info = {}, {}
    for key, p in PANELS.items():
        r0, r1 = p['band']
        dark = np.zeros_like(dark_all)
        dark[r0:r1] = dark_all[r0:r1]
        # registration: side lines (long vertical runs) and notch edges (long runs next to the side)
        cols = dark[r0:r1].sum(0)
        vl = np.where(cols > 0.25 * (r1 - r0))[0]
        g = np.split(vl, np.where(np.diff(vl) > 3)[0] + 1)
        xl, xr = g[0].mean(), g[-1].mean()
        scale = (xr - xl) / L_MM
        def notch_edges(c0, c1):
            run = np.array([dark[r, c0:c1].mean() for r in range(dark.shape[0])])
            rows = np.where(run > 0.9)[0]
            e = np.split(rows, np.where(np.diff(rows) > 3)[0] + 1)
            return e[0].mean(), e[-1].mean()

        edges = [notch_edges(int(xl) + 4, int(xl) + 60), notch_edges(int(xr) - 60, int(xr) - 4)]
        yc = np.mean([0.5 * (t + b) for t, b in edges])      # notch centre line (drawing skew averaged)
        top_edge, bot_edge = edges[0]
        reg = (xl, scale, yc)
        # remove outline: side lines, notch edges up to the tip and the rounded tips
        rm = np.zeros_like(dark)
        for xs in (xl, xr):
            rm[:, int(xs) - 5:int(xs) + 6] = True
        tip = NOTCH_MM * scale
        yy, xx = np.mgrid[:dark.shape[0], :dark.shape[1]]
        for (t_e, b_e), (c0, c1), xt in zip(edges, [(int(xl), int(xl + tip)), (int(xr - tip), int(xr) + 1)],
                                             (xl + tip, xr - tip)):
            for row in (t_e, b_e):
                rm[int(row) - 4:int(row) + 5, c0:c1] = True
            rm |= np.hypot(xx - xt, yy - 0.5 * (t_e + b_e)) <= 0.5 * (b_e - t_e) + 6
        rm[:, :int(xl) - 5] = True
        rm[:, int(xr) + 6:] = True
        for x0, x1, y0, y1 in p['rects']:
            rm[y0:y1, x0:x1] = True
        for a, b in p['leaders']:
            rm |= segment_mask(dark.shape, a, b)
        strokes = dark & ~rm
        strokes = ndimage.binary_opening(strokes, np.ones((2, 2)))   # drop scan specks
        skel = skeletonize(strokes)
        lab, n = ndimage.label(skel, structure=np.ones((3, 3)))
        size = ndimage.sum(skel, lab, index=np.arange(1, n + 1))
        solid_ids = np.flatnonzero(size >= 13.0 * scale) + 1
        solid = np.isin(lab, solid_ids)
        dashes = skel & ~solid & np.isin(lab, np.flatnonzero(size >= 4) + 1)
        rear = skeletonize(ndimage.binary_dilation(dashes, iterations=4))
        front_g, rear_g = to_grid(*np.nonzero(solid), reg), to_grid(*np.nonzero(rear), reg)
        result[f'{key}_front'], result[f'{key}_rear'] = front_g, rear_g
        info[key] = dict(xl=float(xl), xr=float(xr), scale_px_per_mm=float(scale), notch_rows=[[float(v) for v in e] for e in edges],
                         front_px=int(front_g.sum()), rear_px=int(rear_g.sum()), Ps_kN=p['Ps'],
                         specimen=p['specimen'])
        print(key, info[key], flush=True)
        # overlay: solid red, dashes blue, removed grey
        sub = fig[r0:r1]
        ov = np.stack([sub] * 3, -1).astype(np.uint8).copy()
        ov[rm[r0:r1] & dark[r0:r1]] = [190, 190, 190]
        ov[ndimage.binary_dilation(solid, iterations=1)[r0:r1]] = [230, 0, 0]
        ov[ndimage.binary_dilation(rear, iterations=1)[r0:r1]] = [0, 90, 255]
        fitz.Pixmap(fitz.csRGB, ov.shape[1], ov.shape[0], np.ascontiguousarray(ov).tobytes(), False).save(
            os.path.join(out_dir, f'fig428_{key}_overlay.png'))
    np.savez_compressed(os.path.join(out_dir, 'nooru1992_fig428_crackmaps.npz'), **result,
                        info=str(info))


if __name__ == '__main__':
    main()

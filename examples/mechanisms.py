"""Mechanisms that govern what crack maps reveal, mined from the facade simulation campaign.

usage: python mechanisms.py DATA_DIR NPE_DIR OUT.json OUT_PREFIX [--boot 300]
A. Hinge quantization and selection: position of the mechanism crack (the crack component with the
   largest vertical extent, at least 30% of the height, at the last stage) against the pivot x_p of the
   settlement profile; distance to the window edges; logistic switch between the two hinge lines.
B. Onset law: the settlement at which visible cracking starts, S_c, as a power law of all parameters,
   fitted by maximum likelihood with interval censoring (S_c lies between two recorded stages) and right
   censoring (no visible crack up to the last stage); bootstrap intervals over simulations; shares of the
   parameters in the variance of ln S_c.
C. Post-onset kinematics (simulations with stored displacements): rigid-body fits of the two facade parts
   on either side of the mechanism crack, their relative rotation times the height against the widest
   crack opening, and the completion of the crack pattern.
D. Consequences for inference: posterior contraction of S on held-out maps against S - S_c, and the
   posterior width of x_p against the width of the x_p basin of the hinge seen on the map.
Writes OUT.json and the figures OUT_PREFIX_hinge.png, _onset.png, _rotation.png.
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
import matplotlib.colors
import matplotlib.ticker
import matplotlib.tri
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy import optimize, stats  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
sys.path.insert(0, os.path.join(HERE, '..', 'scripts'))
from crackcause import giardina as G  # noqa: E402
from crackcause.hinges import HINGES, hinge_of, main_crack  # noqa: E402
from crackcause.settlement import hinge_profile  # noqa: E402
from train_npe import CONFIGS  # noqa: E402

BLUE, ORANGE, AQUA, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#1baf7a', '#0b0b0b', '#898781', '#e1e0d9'
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.titlesize': 8})
ap = argparse.ArgumentParser()
ap.add_argument('data_dir')
ap.add_argument('npe_dir')
ap.add_argument('out_json')
ap.add_argument('prefix')
ap.add_argument('--boot', type=int, default=300)
ap.add_argument('--profile-hinges', default=None,
                help='results of profile_hinges.py: hinge frequencies of the reference runs of the test')
a = ap.parse_args()
H, X_REF, PIX = G.H, 30.0, 10.0
EDGES = sorted({x for x0, x1 in G._WIN_X for x in (x0, x1)})
HCOL = dict(H1=ORANGE, H2=BLUE)
X_EXP = 732.0                              # hinge fit of the applied profile of the test
P_NAMES = ['w', 'x_p', 'kn', 'ft', 'Gf', 'q']
P_TEX = [r'$w$', r'$x_p-x_\mathrm{ref}$', r'$k_n$', r'$f_t$', r'$G_f$', r'$q$']


def style(ax, grid=True):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(color=GRID, lw=0.5)
        ax.set_axisbelow(True)


# ---- per-simulation table ---------------------------------------------------------------------------------
heldout = np.load(os.path.join(a.npe_dir, 'heldout.npz'))
held_k = set(int(k) for k in heldout['sim'])
files = sorted(glob.glob(os.path.join(a.data_dir, 'sims', 'sim_*.npz')))
rows, held_maps = [], {}
density = {h: np.zeros(CONFIGS['facade']['shape']) for h in HINGES}
for k, f in enumerate(files):
    z = np.load(f)
    S, maps = z['S'], z['maps']
    if len(S) == 0:
        continue
    th = json.loads(str(z['theta']))
    px = maps.reshape(len(maps), -1).sum(1)
    on = np.flatnonzero(px >= 3)
    if len(on):
        j = on[0]
        lo_k, hi_k = (float(S[j - 1]) if j > 0 else 0.0), float(S[j])
    else:
        lo_k, hi_k = float(S[-1]), np.inf
    x_k, e_k = main_crack(maps[-1]) if px[-1] >= 3 else (np.nan, 0.0)
    h_k = hinge_of(x_k, e_k)
    if h_k in HINGES:
        density[h_k] += maps[-1] > 0
    on_line = bool(np.isfinite(x_k) and min(abs(x_k - xh) for xh in HINGES.values()) <= 60.0)
    rows.append(dict(k=k, file=f, lo=lo_k, hi=hi_k, Sc=hi_k if np.isfinite(hi_k) else np.nan, xm=float(x_k),
                     ext=float(e_k), hinge=h_k, on_line=on_line, has_u='u' in z.files, **th))
    if k in held_k:
        held_maps[k] = (S.copy(), maps.copy())


def col(key):
    return np.array([r[key] for r in rows], dtype=object if key == 'hinge' else float)


x_p, w, kn, ft, Gf, q = col('x_p'), col('w'), col('kn'), col('ft'), col('Gf'), col('q')
lo, hi, xm, hinge = col('lo'), col('hi'), col('xm'), col('hinge')
rep = dict(n_sims=len(rows), frac_cracking=float(np.isfinite(hi).mean()))
gen = np.random.default_rng(1)

# ---- A: hinge quantization and selection ------------------------------------------------------------------
mech = np.isin(hinge, ['H1', 'H2', 'other'])
d_edge = np.array([min(abs(x - e) for e in EDGES) if np.isfinite(x) else np.nan for x in xm])
x_rand = np.random.default_rng(0).uniform(0, G.W, 20000)
d_rand = np.array([min(abs(x - e) for e in EDGES) for x in x_rand])
basin = {h: [float(np.quantile(x_p[hinge == h], 0.05)), float(np.quantile(x_p[hinge == h], 0.95))] for h in HINGES}
sel = np.isin(hinge, list(HINGES))
y_sel = (hinge[sel] == 'H2').astype(float)


def logit_nll(beta, A, y):
    t = A @ beta
    return np.sum(np.logaddexp(0.0, t) - y * t)


def switch_fit(xp, y):
    """Logistic P(H2) = 1 / (1 + exp(-(x_p - x*) / s)); returns (x*, s)."""
    A = np.column_stack([np.ones(len(xp)), (xp - 900.0) / 100.0])
    b = optimize.minimize(logit_nll, np.zeros(2), args=(A, y), method='BFGS').x
    s = 100.0 / b[1]
    return 900.0 - b[0] * s, s


x_star, s_star = switch_fit(x_p[sel], y_sel)
bs = np.array([switch_fit(x_p[sel][i], y_sel[i]) for i in (gen.integers(0, sel.sum(), sel.sum()) for _ in range(a.boot))])
Lsel = np.column_stack([np.log(w), np.log(kn), np.log(ft), np.log(Gf), np.log(q)])[sel]
Zsel = (Lsel - Lsel.mean(0)) / Lsel.std(0)
A_full = np.column_stack([np.ones(sel.sum()), (x_p[sel] - x_star) / 100.0, Zsel])
r_full = optimize.minimize(logit_nll, np.zeros(A_full.shape[1]), args=(A_full, y_sel), method='BFGS')
se_full = np.sqrt(np.diag(r_full.hess_inv))
shift = {n: dict(mm_per_sd=float(-r_full.x[2 + i] / r_full.x[1] * 100.0), z=float(r_full.x[2 + i] / se_full[2 + i]))
         for i, n in enumerate(['w', 'kn', 'ft', 'Gf', 'q'])}
local = (hinge == 'none') & np.isfinite(xm)       # cracked by the last stage, but no mechanism crack
rep['hinge'] = dict(n_mechanism=int(mech.sum()), n_cracking=int(np.isfinite(hi).sum()),
                    n_local=int(local.sum()), local_on_hinge_line=float(np.mean(col('on_line')[local] > 0.5)),
                    local_extent_median=float(np.median(col('ext')[local])),
                    counts={h: int(np.sum(hinge == h)) for h in ('H1', 'H2', 'other')},
                    median_dist_to_window_edge_mm=float(np.median(d_edge[mech])),
                    frac_within_30mm_of_edge=float(np.mean(d_edge[mech] <= 30.0)),
                    frac_within_30mm_uniform=float(np.mean(d_rand <= 30.0)),
                    basin_xp_90=basin, switch_xp=float(x_star), switch_scale=float(s_star),
                    switch_xp_ci95=[float(v) for v in np.quantile(bs[:, 0], [0.025, 0.975])],
                    switch_band_10_90=[float(x_star - 2.197 * s_star), float(x_star + 2.197 * s_star)],
                    p_H2_at_exp_fit=float(1.0 / (1.0 + np.exp(-(X_EXP - x_star) / s_star))),
                    switch_shift_per_sd=shift,
                    accuracy_xp_only=float(np.mean((x_p[sel] > x_star).astype(float) == y_sel)),
                    accuracy_all=float(np.mean(((A_full @ r_full.x) > 0) == (y_sel > 0.5))))

# sensitivity of the hinge analysis to its two thresholds (minimum vertical extent, distance to the line)
sens = []
for min_ext in (0.2, 0.3, 0.4):
    for tol in (40.0, 60.0, 80.0):
        h_s = np.array([hinge_of(r['xm'], r['ext'], min_extent=min_ext, tol=tol) for r in rows], dtype=object)
        m_s = np.isin(h_s, list(HINGES))
        xs_, ss_ = switch_fit(x_p[m_s], (h_s[m_s] == 'H2').astype(float))
        mech_s = np.isin(h_s, ['H1', 'H2', 'other'])
        sens.append(dict(min_extent=min_ext, tol_mm=tol, n_mechanism=int(mech_s.sum()), n_on_lines=int(m_s.sum()),
                         switch_xp=float(xs_), switch_scale=float(ss_)))
rep['hinge']['threshold_sensitivity'] = sens

# ---- B: onset law, interval- and right-censored maximum likelihood ---------------------------------------
LX = np.column_stack([np.log(w), np.log(x_p - X_REF), np.log(kn), np.log(ft), np.log(Gf), np.log(q)])


def onset_nll(par, X, lo_, hi_):
    b, s = par[:-1], np.exp(par[-1])
    mu = b[0] + X @ b[1:]
    llo = np.where(lo_ > 0, np.log(np.maximum(lo_, 1e-12)), -np.inf)
    lhi = np.where(np.isfinite(hi_), np.log(np.where(np.isfinite(hi_), hi_, 1.0)), np.inf)
    p = stats.norm.cdf((lhi - mu) / s) - stats.norm.cdf((llo - mu) / s)
    return -np.sum(np.log(np.maximum(p, 1e-300)))


def onset_mid(lo_, hi_):
    """Geometric mid-point of the onset interval (the upper end for onset at the first stage)."""
    ok = np.isfinite(hi_)
    h_ = np.where(ok, hi_, 1.0)
    return np.where(lo_ > 0, np.sqrt(np.maximum(lo_, 1e-9) * h_), h_), ok


def onset_fit(X, lo_, hi_):
    m_, ok = onset_mid(lo_, hi_)
    A = np.column_stack([np.ones(ok.sum()), X[ok]])
    b0 = np.linalg.lstsq(A, np.log(m_[ok]), rcond=None)[0]
    par0 = np.r_[b0, np.log(np.std(np.log(m_[ok]) - A @ b0))]
    return optimize.minimize(onset_nll, par0, args=(X, lo_, hi_), method='L-BFGS-B').x


par = onset_fit(LX, lo, hi)
coef, sig = par[1:-1], float(np.exp(par[-1]))
boot = np.array([onset_fit(LX[i], lo[i], hi[i])[1:-1]
                 for i in (gen.integers(0, len(rows), len(rows)) for _ in range(a.boot))])
ci = np.quantile(boot, [0.025, 0.975], 0)
var_lin = float(np.var(LX @ coef))
share = coef ** 2 * np.var(LX, 0)
mu_c = par[0] + LX @ coef
rep['onset'] = dict(n=len(rows), n_censored=int(np.sum(~np.isfinite(hi))), sigma=sig,
                    pseudo_r2=var_lin / (var_lin + sig ** 2), const=float(par[0]),
                    exponents={n: float(c) for n, c in zip(P_NAMES, coef)},
                    exponents_ci95={n: [float(ci[0, i]), float(ci[1, i])] for i, n in enumerate(P_NAMES)},
                    variance_share={n: float(s / share.sum()) for n, s in zip(P_NAMES, share)})
# strength and critical opening w_c = 2 G_f / f_t: f_t^a G_f^b = 2^-b f_t^(a+b) w_c^b
i_ft, i_gf = P_NAMES.index('ft'), P_NAMES.index('Gf')
rep['onset']['ft_wc'] = dict(a_ft=float(coef[i_ft] + coef[i_gf]),
                             a_ft_ci95=[float(v) for v in np.quantile(boot[:, i_ft] + boot[:, i_gf], [0.025, 0.975])],
                             a_wc=float(coef[i_gf]))
# for comparison: least squares on the cracking simulations only (censoring ignored, onset at the first record)
crk = np.isfinite(hi)
A_n = np.column_stack([np.ones(crk.sum()), LX[crk]])
c_n = np.linalg.lstsq(A_n, np.log(hi[crk]), rcond=None)[0]
rep['onset']['naive'] = dict(exponents={n: float(c) for n, c in zip(P_NAMES, c_n[1:])},
                             r2=float(1 - np.var(np.log(hi[crk]) - A_n @ c_n) / np.var(np.log(hi[crk]))))
# angular distortion of the profile at onset, beta_c = S_c / (x_p - x_ref), and the onset at the test values
beta = onset_mid(lo, hi)[0][crk] / (x_p[crk] - X_REF)
x_test = np.log(np.array([98.0, X_EXP - X_REF, G.INTERFACE['kn'], G.MASONRY['ft'], G.MASONRY['Gf'], 1.0]))
Sc_test = float(np.exp(par[0] + x_test @ coef))
rep['onset']['beta_c_inv_q10_50_90'] = [float(1 / np.quantile(beta, p)) for p in (0.9, 0.5, 0.1)]
rep['onset']['Sc_test'] = dict(median=Sc_test, q05=Sc_test * float(np.exp(-1.645 * sig)),
                               q95=Sc_test * float(np.exp(1.645 * sig)), beta_inv=float((X_EXP - X_REF) / Sc_test))
# regime of the interface: exponent of k_n within ranges of k_n that contain equal numbers of cracking runs,
# against the characteristic length of the wall on the interface, L_c = (E H^3 / (3 k_n))^(1/4)
Lc = (G.MASONRY['E'] * H ** 3 / (3.0 * kn)) ** 0.25
edges = np.r_[np.log(kn).min(), np.quantile(np.log(kn[crk]), [0.25, 0.5, 0.75]), np.log(kn).max()]
regime = []
for e0, e1 in zip(edges[:-1], edges[1:]):
    m = (np.log(kn) >= e0) & (np.log(kn) <= e1)
    p_m = onset_fit(LX[m], lo[m], hi[m])
    b_m = [onset_fit(LX[m][i], lo[m][i], hi[m][i])[3] for i in (gen.integers(0, m.sum(), m.sum()) for _ in range(a.boot // 3))]
    regime.append(dict(kn_range=[float(np.exp(e0)), float(np.exp(e1))], Lc_over_W=float(np.median(Lc[m]) / G.W),
                       n=int(m.sum()), n_cracking=int(crk[m].sum()), a_kn=float(p_m[3]),
                       a_kn_ci95=[float(np.quantile(b_m, 0.025)), float(np.quantile(b_m, 0.975))]))
rep['onset']['kn_regimes'] = regime
rep['onset']['Lc_over_W_range'] = [float(Lc.min() / G.W), float(Lc.max() / G.W)]

# ---- C: post-onset kinematics ---------------------------------------------------------------------------
mesh = np.load(os.path.join(a.data_dir, 'mesh.npz'))
XY, cells = mesh['coords'], mesh['cells']


def rigid_fit(xy, u):
    """Least-squares small rigid motion u = t + phi (-(y - yc), x - xc); returns (phi, R^2)."""
    d = xy - xy.mean(0)
    A = np.zeros((2 * len(xy), 3))
    A[0::2, 0], A[1::2, 1], A[0::2, 2], A[1::2, 2] = 1.0, 1.0, -d[:, 1], d[:, 0]
    b = u.reshape(-1)
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    return sol[2], 1.0 - ((b - A @ sol) ** 2).sum() / max(((u - u.mean(0)) ** 2).sum(), 1e-30)


base = np.flatnonzero(XY[:, 1] < 1e-6)
base = base[np.argsort(XY[base, 0])]
kin, example, detached = [], None, []
for r in rows:
    if r['hinge'] not in HINGES or not r['has_u']:
        continue
    z = np.load(r['file'])
    S, maps, U, Wd = z['S'], z['maps'], z['u'], z['widths']
    px = maps.reshape(len(maps), -1).sum(1)
    # contact at the onset: share of the base length where the wall is above the settled ground
    j_on = int(np.flatnonzero(px >= 3)[0])
    gap = U[j_on][base, 1] - S[j_on] * hinge_profile(XY[base, 0], r['x_p'], r['w'])
    detached.append(float(np.mean(gap > 0.0)))
    L, R = XY[:, 0] < r['xm'] - 40.0, XY[:, 0] > r['xm'] + 40.0
    for j in range(len(S)):
        pL, r2L = rigid_fit(XY[L], U[j][L].astype(float))
        pR, r2R = rigid_fit(XY[R], U[j][R].astype(float))
        wm = float(np.nanmax(np.where(maps[j] > 0, Wd[j].astype(float), np.nan))) if px[j] else 0.0
        kin.append(dict(dS=float(S[j] - r['Sc']), rot_H=float((pL - pR) * H), wmax=wm, r2=float(min(r2L, r2R)),
                        frac_px=float(px[j] / max(px[-1], 1)), S=float(S[j])))
    if example is None and r['hinge'] == 'H1' and S[-1] - r['Sc'] >= 6.0:
        example = dict(r=r, u=U[-1].astype(float), d=z['d'][-1].astype(float), S=float(S[-1]))
K = {key: np.array([e[key] for e in kin]) for key in ('dS', 'rot_H', 'wmax', 'r2', 'frac_px', 'S')}
bins = np.array([-8, -4, -2, 0, 1, 2, 4, 8, 12])
mid = 0.5 * (bins[1:] + bins[:-1])


def binned(v, x, fn=np.median):
    return [float(fn(v[(x >= b0) & (x < b1)])) if np.any((x >= b0) & (x < b1)) else np.nan
            for b0, b1 in zip(bins[:-1], bins[1:])]


post = K['dS'] > 0.25
cw = np.polyfit(K['rot_H'][post], K['wmax'][post], 1)
r2w = float(1 - np.var(K['wmax'][post] - np.polyval(cw, K['rot_H'][post])) / np.var(K['wmax'][post]))
rep['kinematics'] = dict(n_sims=int(sum(1 for r in rows if r['hinge'] in HINGES and r['has_u'])), n_stages=len(kin),
                         detached_at_onset_median=float(np.median(detached)),
                         detached_at_onset_q90=float(np.quantile(detached, 0.9)),
                         bins=bins.tolist(), rigid_r2_median=binned(K['r2'], K['dS']),
                         rigid_r2_q10_post=float(np.quantile(K['r2'][post], 0.1)),
                         rot_H_median=binned(K['rot_H'], K['dS']), wmax_median=binned(K['wmax'], K['dS']),
                         completion_median=binned(K['frac_px'], K['dS']),
                         wmax_vs_rotH=dict(slope=float(cw[0]), intercept=float(cw[1]), r2=r2w),
                         r2_wmax_vs_dS=float(np.corrcoef(K['dS'][post], K['wmax'][post])[0, 1] ** 2),
                         r2_wmax_vs_S=float(np.corrcoef(K['S'][post], K['wmax'][post])[0, 1] ** 2))

# ---- D: consequences for inference ----------------------------------------------------------------------
box = CONFIGS['facade']['box']
Sc_by_k = {r['k']: r['Sc'] for r in rows}
S_true = heldout['theta'][:, 0]
sc_h = np.array([Sc_by_k.get(int(k), np.nan) for k in heldout['sim']])
con_S = 1.0 - (heldout['q95'][:, 0] - heldout['q05'][:, 0]) / (0.9 * (box.high[0] - box.low[0]))
dS_h = S_true - sc_h
jx = box.names.index('x_p')
wid_x = heldout['q95'][:, jx] - heldout['q05'][:, jx]
h_map = []
for k, s in zip(heldout['sim'], S_true):
    Sk, Mk = held_maps[int(k)]
    j = int(np.argmin(np.abs(Sk - s)))
    h_map.append(hinge_of(*main_crack(Mk[j])) if Mk[j].sum() >= 3 else 'none')
h_map = np.array(h_map, dtype=object)
pw_x = 0.9 * (box.high[jx] - box.low[jx])
xp_basin = {h: dict(n_maps=int(np.sum(h_map == h)), posterior_width90_mean=float(wid_x[h_map == h].mean()),
                    basin_width90=float(basin[h][1] - basin[h][0]),
                    contraction_npe=float(1 - wid_x[h_map == h].mean() / pw_x),
                    contraction_basin=float(1 - (basin[h][1] - basin[h][0]) / pw_x)) for h in HINGES}
crk_h = heldout['cracked'].astype(bool)
q05_S, q95_S = heldout['q05'][:, 0], heldout['q95'][:, 0]
rep['inference'] = dict(bins=bins.tolist(), contraction_S=binned(con_S, dS_h, np.mean),
                        contraction_S_never_cracking=float(np.mean(con_S[~np.isfinite(sc_h)])), xp_basin=xp_basin,
                        S_upper_at_prior_edge=float(np.mean(q95_S[crk_h] > box.high[0] - 1.0)),
                        S_q05_median_small=float(np.median(q05_S[crk_h & (S_true < 3.0)])),
                        S_q05_median_large=float(np.median(q05_S[crk_h & (S_true >= 9.0)])))
npe_rep = json.load(open(os.path.join(a.npe_dir, 'report.json')))
x_post = float(np.median([v['x_p']['median'] for key, v in npe_rep['experiment'].items() if not key.endswith('_w')]))
rep['hinge']['exp_posterior_median_xp'] = x_post
rep['hinge']['p_H2_at_exp_posterior'] = float(1.0 / (1.0 + np.exp(-(x_post - x_star) / s_star)))
json.dump(rep, open(a.out_json, 'w'), indent=1)

# ---- figure: hinges --------------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.45), gridspec_kw=dict(width_ratios=[1.0, 1.0, 0.9]))
ax = axs[0]
for h, dens in density.items():       # how often a pixel is cracked, separately for the two hinges
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(h, ['#ffffff', HCOL[h]])
    ax.imshow(np.ma.masked_less(dens, 1), origin='lower', extent=(0, G.W, 0, H), cmap=cmap, vmin=0,
              vmax=max(np.quantile(dens[dens > 0], 0.9), 1), interpolation='nearest')
for x0, x1, y0, y1 in G.OPENINGS:
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc='none', ec=MUTED, lw=0.5))
ax.add_patch(Rectangle((0, 0), G.W, H, fc='none', ec=INK, lw=0.6))
for name, xh in HINGES.items():
    ax.annotate(name, (xh, H), xytext=(xh, H + 80), ha='center', fontsize=7, color=HCOL[name],
                arrowprops=dict(arrowstyle='-', color=HCOL[name], lw=0.8))
ax.set_xlim(-20, G.W + 20)
ax.set_ylim(-20, H + 170)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title(f'(a) mechanism cracks, {int(sel.sum())} runs')
ax = axs[1]
for x0, x1 in G._WIN_X:
    ax.axhspan(x0, x1, color=GRID, lw=0)
for h in HINGES:
    m = hinge == h
    ax.scatter(x_p[m], xm[m], s=7, color=HCOL[h], edgecolors='white', linewidths=0.3, label=h, zorder=3)
m = hinge == 'other'
ax.scatter(x_p[m], xm[m], s=7, color=MUTED, edgecolors='white', linewidths=0.3, label='other', zorder=3)
ax.set_xlabel('pivot $x_p$ (mm)')
ax.set_ylabel('crack position (mm)')
ax.set_xlim(300, 1400)
ax.set_ylim(0, G.W)
ax.legend(frameon=False, fontsize=7, loc='upper left', handletextpad=0.2)
ax.set_title('(b) crack position; grey: windows')
style(ax, grid=False)
ax = axs[2]
eb = np.arange(300, 1401, 100)
for b0, b1 in zip(eb[:-1], eb[1:]):
    m = (x_p[sel] >= b0) & (x_p[sel] < b1)
    if m.sum() >= 3:
        k_, n_ = y_sel[m].sum(), m.sum()
        lo_, hi_ = stats.beta.ppf([0.05, 0.95], k_ + 0.5, n_ - k_ + 0.5)
        ax.errorbar(0.5 * (b0 + b1), k_ / n_, yerr=[[max(k_ / n_ - lo_, 0)], [max(hi_ - k_ / n_, 0)]], fmt='o',
                    ms=3.5, color=INK, ecolor=MUTED, elinewidth=0.8, capsize=0, zorder=3)
xx = np.linspace(300, 1400, 300)
ax.plot(xx, 1 / (1 + np.exp(-(xx - x_star) / s_star)), color=BLUE, lw=1.6)
ax.axvspan(*rep['hinge']['switch_band_10_90'], color=BLUE, alpha=0.08, lw=0)
ax.axvline(X_EXP, color=ORANGE, lw=1.0, ls='--')
ax.axvline(x_post, color=ORANGE, lw=1.0, ls=':')
ax.text(X_EXP - 12, 0.97, 'test:\nfitted pivot', ha='right', va='top', fontsize=6.5, color=ORANGE)
ax.text(x_post + 14, 0.03, 'posterior\nmedian', ha='left', va='bottom', fontsize=6.5, color=ORANGE)
if a.profile_hinges and os.path.exists(a.profile_hinges):
    # frequency of H2 in reference runs of the test with the measured profile and with its two-parameter fit
    summ = json.load(open(a.profile_hinges))['summary']
    for prof, mk, dx in (('measured', 'D', -14.0), ('hinge', 's', 14.0)):
        k_, n_ = summ[prof]['H2'], summ[prof]['n']
        if n_:
            lo_, hi_ = stats.beta.ppf([0.05, 0.95], k_ + 0.5, n_ - k_ + 0.5)
            ax.errorbar(X_EXP + dx, k_ / n_, yerr=[[max(k_ / n_ - lo_, 0)], [max(hi_ - k_ / n_, 0)]], fmt=mk, ms=4,
                        mfc='white' if prof == 'hinge' else ORANGE, mec=ORANGE, ecolor=ORANGE, elinewidth=0.8,
                        capsize=0, zorder=4)
            ax.text(X_EXP - 36, k_ / n_, 'measured\nprofile' if prof == 'measured' else 'fitted\nprofile',
                    ha='right', va='center', fontsize=6, color=ORANGE, linespacing=1.0, zorder=5,
                    bbox=dict(fc='white', ec='none', pad=0.4, alpha=0.9))
ax.set_xlabel('pivot $x_p$ (mm)')
ax.set_ylabel('probability of hinge H2')
ax.set_xlim(300, 1400)
ax.set_ylim(-0.03, 1.03)
ax.set_title(f'(c) hinge switch at {x_star:.0f} mm')
style(ax)
fig.tight_layout(w_pad=0.6)
fig.savefig(a.prefix + '_hinge.png', dpi=600)

# ---- figure: onset law -----------------------------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.0))
axs = axs.ravel()
ax = axs[0]
ax.bar(range(6), coef, color=BLUE, width=0.6)
ax.errorbar(range(6), coef, yerr=[coef - ci[0], ci[1] - coef], fmt='none', ecolor=INK, elinewidth=0.9, capsize=2)
ax.axhline(0, color=MUTED, lw=0.7)
ax.set_xticks(range(6))
ax.set_xticklabels(P_TEX, fontsize=7, rotation=30)
ax.set_ylabel('exponent $a_i$')
ax.set_title('(a) exponents, 95% intervals')
style(ax)
ax = axs[1]
crk = np.isfinite(hi)
pred = np.exp(mu_c)
obs, _ = onset_mid(lo, hi)
ax.scatter(pred[~crk], lo[~crk], s=7, marker='^', color=MUTED, edgecolors='none', alpha=0.5,
           label='no crack up to the last stage', zorder=2)
ax.scatter(pred[crk], obs[crk], s=6, color=BLUE, edgecolors='white', linewidths=0.25, label='onset observed', zorder=3)
ax.plot([0.3, 300], [0.3, 300], color=INK, lw=0.8)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.3, 300)
ax.set_ylim(0.3, 20)
ax.set_xlabel('predicted onset settlement (mm)')
ax.set_ylabel('$S_c$ in the simulation (mm)')
ax.legend(frameon=False, fontsize=6.5, loc='lower right', handletextpad=0.2)
ax.set_title(f'(b) censored fit, pseudo-$R^2$ = {rep["onset"]["pseudo_r2"]:.2f}')
fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}')
ax.xaxis.set_major_formatter(fmt)
ax.yaxis.set_major_formatter(fmt)
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
style(ax)
ax = axs[2]
sh = share / share.sum()
order = np.argsort(sh)
ax.barh(range(6), sh[order], color=[ORANGE if P_NAMES[i] == 'kn' else BLUE for i in order], height=0.6)
ax.set_yticks(range(6))
ax.set_yticklabels([P_TEX[i] for i in order], fontsize=7)
for yv, v in enumerate(sh[order]):
    ax.text(v + 0.02, yv, f'{100 * v:.0f}%', va='center', fontsize=6.5, color=INK)
ax.set_xlim(0, 1.0)
ax.set_xlabel(r'share of the variance of $\ln S_c$')
ax.set_title('(c) what sets the onset')
style(ax)
ax = axs[3]
lw_ = np.array([r_['Lc_over_W'] for r_ in regime])
ak = np.array([r_['a_kn'] for r_ in regime])
ak_ci = np.array([r_['a_kn_ci95'] for r_ in regime])
ax.axhline(-1.0, color=MUTED, lw=0.8, ls='--')
ax.axhline(-0.25, color=MUTED, lw=0.8, ls='--')
ax.text(lw_.min() - 0.05, -1.0, 'rigid-wall limit', va='bottom', ha='left', fontsize=6.5, color=MUTED)
ax.text(lw_.min() - 0.05, -0.25, 'flexible-wall limit', va='bottom', ha='left', fontsize=6.5, color=MUTED)
ax.errorbar(lw_, ak, yerr=[ak - ak_ci[:, 0], ak_ci[:, 1] - ak], fmt='o', ms=4, color=ORANGE, ecolor=INK, elinewidth=0.9,
            capsize=2, zorder=3)
ax.set_xlim(lw_.min() - 0.08, lw_.max() + 0.08)
ax.set_xlabel(r'characteristic length / wall width, $L_c/W$')
ax.set_ylabel('exponent of $k_n$')
ax.set_ylim(-1.3, 0.0)
ax.set_title('(d) exponent of $k_n$ by interface stiffness')
style(ax)
fig.tight_layout(w_pad=1.0, h_pad=1.2)
fig.savefig(a.prefix + '_onset.png', dpi=600)

# ---- figure: post-onset rotation ------------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw=dict(width_ratios=[1.15, 1.0, 1.0]))
ax = axs[0]
if example is not None:
    mag = 8.0
    xy = XY + mag * example['u']
    tri = matplotlib.tri.Triangulation(xy[:, 0], xy[:, 1], cells)
    left = (XY[cells, 0].mean(1) < example['r']['xm']).astype(float)
    ax.tripcolor(tri, facecolors=left, cmap=matplotlib.colors.ListedColormap([BLUE, ORANGE]), alpha=0.35,
                 rasterized=True)
    dc = example['d'][cells].mean(1)
    tri.set_mask(dc < 0.9)
    ax.tripcolor(tri, facecolors=np.ones(len(cells)), cmap=matplotlib.colors.ListedColormap([INK]), rasterized=True)
    ax.plot([0, G.W, G.W, 0, 0], [0, 0, H, H, 0], color=MUTED, lw=0.5, ls=':')
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'(a) deformed shape, x{mag:.0f}')
ax = axs[1]
ax.scatter(K['rot_H'][post], K['wmax'][post], s=4, color=BLUE, alpha=0.35, edgecolors='none', rasterized=True)
lim = float(np.quantile(np.r_[K['rot_H'][post], K['wmax'][post]], 0.995))
ax.plot([0, lim], [0, lim], color=INK, lw=0.8)
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_xlabel(r'rotation $\times$ height, $\Delta\varphi\, H$ (mm)')
ax.set_ylabel('widest crack opening (mm)')
ax.set_title(f'(b) opening vs. rotation, $R^2$ = {r2w:.2f}')
style(ax)
ax = axs[2]
ax.plot(mid, rep['kinematics']['completion_median'], '-o', color=ORANGE, ms=3.5, lw=1.6, mec='white', mew=0.5,
        label='crack pattern completed')
ax.plot(mid, rep['inference']['contraction_S'], '-o', color=AQUA, ms=3.5, lw=1.6, mec='white', mew=0.5,
        label='posterior contraction of $S$')
ax.axvline(0, color=MUTED, lw=0.8, ls='--')
ax.axhline(0, color=MUTED, lw=0.7)
ax.set_xlabel('settlement after onset $S - S_c$ (mm)')
ax.set_ylabel('fraction')
ax.set_ylim(-0.05, 1.02)
ax.legend(frameon=False, fontsize=6.5, loc='upper left')
ax.set_title('(c) after the onset')
style(ax)
fig.tight_layout(w_pad=0.6)
fig.savefig(a.prefix + '_rotation.png', dpi=600)
print(json.dumps(rep, indent=1))

"""Figure: bar verification of the production PF-CZM solver (traction-separation law, energy, d*-relations).

usage: python plot_verify_bar.py RUN_DIR OUT.png
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

run_dir, out = sys.argv[1], sys.argv[2]
FT, GF, WEAK = 0.1, 0.01, 0.95
ftw, wcw = WEAK * FT, 2 * GF / (WEAK * FT)
res = json.load(open(os.path.join(run_dir, 'verify_bar_fx.json')))
fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
w_lin = np.linspace(0, 1, 50)
axs[0].plot(w_lin, 1 - w_lin, 'k--', lw=1, label='linear softening law')
axs[1].plot([0, 1], [1, 0], 'k--', lw=1, label=r'$\sigma=f_t(1-d^*)$')
axs[2].plot([0, 1], [0, 1], 'k--', lw=1, label=r'$w=d^* w_c$')
for k, (ell, r) in enumerate(sorted(res.items(), key=lambda kv: float(kv[0]))):
    st = r['steps']
    s = np.array([x['sigma'] for x in st]) / ftw
    w = np.array([x['w'] for x in st]) / wcw
    d = np.array([x['dstar'] for x in st])
    lab = rf'$\ell$ = {float(ell):g} mm ($G/G_f$ = {r["energy"]:.3f})'
    axs[0].plot(np.maximum(w, 0), s, 'o-', ms=2.5, lw=0.8, color=f'C{k}', label=lab)
    axs[1].plot(d, s, 'o', ms=2.5, color=f'C{k}', label=rf'$\ell$ = {float(ell):g} mm')
    axs[2].plot(d, np.maximum(w, 0), 'o', ms=2.5, color=f'C{k}', label=rf'$\ell$ = {float(ell):g} mm')
axs[0].set_xlabel(r'crack opening $w/w_c$')
axs[0].set_ylabel(r'traction $\sigma/f_t$')
axs[1].set_xlabel(r'peak phase field $d^*$')
axs[1].set_ylabel(r'$\sigma/f_t$')
axs[2].set_xlabel(r'peak phase field $d^*$')
axs[2].set_ylabel(r'$w/w_c$')
for ax in axs:
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_xlim(-0.02, 1.08)
    ax.set_ylim(-0.02, 1.1)
fig.tight_layout()
fig.savefig(out, dpi=600)
print('saved', out)

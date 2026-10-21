"""Figure: information of a crack map on the settlement magnitude against the settlement after the onset.

usage: python plot_voi.py VOI.json MECHANISMS.json OUT.png
VOI.json: examples/value_of_information.py; MECHANISMS.json: examples/mechanisms.py (neural posterior).
"""
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, AQUA, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#1baf7a', '#0b0b0b', '#898781', '#e1e0d9'
VIOLET = '#6a4fc4'
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.titlesize': 8})
vo, me = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
b = np.array(vo['bins_dS'])
x = 0.5 * (b[1:] + b[:-1])
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1.45, 1.0]))
ax = axs[0]
ss = vo['same_simulation']
lo = np.array(ss['chamfer1.0_length0.2']['contraction90_by_dS'])
hi = np.array(ss['chamfer0.5_length0.1']['contraction90_by_dS'])
ax.fill_between(x, lo, hi, color=GRID, lw=0, label='all, and the strength field')
kp = vo['known_parameters']
for key, col, ls, lab in (('map+all', VIOLET, '-', 'all six parameters'),
                          ('map+kn+xp+w', BLUE, '-', r'$k_n$, $x_p$, $w$'),
                          ('map', AQUA, '-', 'nothing (map only)')):
    ax.plot(x, kp[key]['contraction90_by_dS'], ls, color=col, lw=1.6, marker='o', ms=3.5, mec='white', mew=0.5,
            label=lab)
mb = np.array(me['inference']['bins'])
mx = 0.5 * (mb[1:] + mb[:-1])
npe = np.array(me['inference']['contraction_S'])
keep = mx > 0
ax.plot(mx[keep], npe[keep], '--', color=AQUA, lw=1.3, marker='o', ms=3, mfc='white', mec=AQUA,
        label='nothing, neural posterior')
ax.set_xlabel('settlement after the onset, $S - S_c$ (mm)')
ax.set_ylabel('contraction of the 90% interval of $S$')
ax.set_ylim(0, 1.32)
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.set_xlim(0, 10.5)
ax.legend(frameon=False, fontsize=6.2, loc='upper center', ncol=2, columnspacing=0.8, handlelength=1.6, title='known besides the map', title_fontsize=6.2)
ax.set_title('(a) what a crack map records, and when')
ax.grid(color=GRID, lw=0.5)
ax.set_axisbelow(True)
ax = axs[1]
rows = [('crack map only', 'map'), (r'+ $k_n$', 'map+kn'), (r'+ $x_p$, $w$', 'map+xp+w'),
        (r'+ $f_t$, $G_f$, $q$', 'map+ft+Gf+q'), (r'+ $k_n$, $x_p$, $w$', 'map+kn+xp+w'), ('+ all six', 'map+all')]
vals = [kp[k]['contraction90'] for _, k in rows]
cols = [AQUA, BLUE, BLUE, BLUE, BLUE, VIOLET]
y = np.arange(len(rows))[::-1]
ax.barh(y, vals, color=cols, height=0.6)
for yy, v in zip(y, vals):
    ax.text(v + 0.01, yy, f'{v:.2f}', va='center', fontsize=6.5, color=INK)
ax.axvspan(ss['chamfer1.0_length0.2']['contraction90'], ss['chamfer0.5_length0.1']['contraction90'], color=GRID, lw=0)
ax.text(0.5 * (ss['chamfer1.0_length0.2']['contraction90'] + ss['chamfer0.5_length0.1']['contraction90']),
        2.5, 'strength\nfield\nalso\nknown', ha='center', va='center', fontsize=6, color=MUTED)
ax.set_yticks(y)
ax.set_yticklabels([r for r, _ in rows], fontsize=7)
ax.set_xlim(0, 0.8)
ax.set_xlabel('contraction, all maps with cracks')
ax.set_title('(b) value of knowing the building')
for a_ in axs:
    for s in ('top', 'right'):
        a_.spines[s].set_visible(False)
fig.tight_layout(w_pad=1.0)
fig.savefig(sys.argv[3], dpi=600)
print('saved', sys.argv[3])

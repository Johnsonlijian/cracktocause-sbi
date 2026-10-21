"""Figure: workflow of the method (simulation campaign, observation operator, amortized inference, mechanism
analysis, checks).

usage: python plot_workflow.py OUT.png
"""
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

BOXES = {
    'prior': (0.02, 0.64, 0.17, 0.30, 'Prior\n$S, x_p, w$ (settlement)\n$k_n, f_t, G_f, q$ (nuisance)\nstrength field'),
    'sim': (0.24, 0.64, 0.20, 0.30, 'PF-CZM simulator\nfaçade on compression-only\nsprings, settlement profile\nrobust staggered solver'),
    'obs': (0.49, 0.64, 0.20, 0.30, 'Observation operator\nvisible cracks $d \\geq 0.9$\nthinned crack map\nwidest opening $w_{\\max}$'),
    'npe': (0.74, 0.64, 0.24, 0.30, 'Neural posterior estimation\ncrack-map features + MLP\nneural spline flow\nnoise regularization, ensemble'),
    'mech': (0.02, 0.10, 0.22, 0.34, 'Mechanism analysis\nhinge quantization and selection\nonset law (censored regression)\npost-onset rigid-block rotation\n$\\rightarrow$ what a crack map determines'),
    'check': (0.27, 0.10, 0.18, 0.34, 'Checks\ncalibration on held-out\nsimulations\nmisspecification: distance\nto simulated maps'),
    'insp': (0.49, 0.12, 0.20, 0.30, 'Inspection\ntraced crack map\n(digitized, thinned)\nmeasured widest crack'),
    'post': (0.74, 0.12, 0.24, 0.30, 'Posterior $p(S, x_p, \\ldots \\mid$ map, $w_{\\max})$\nin milliseconds per survey'),
}
fig, ax = plt.subplots(figsize=(11, 4.4))
ax.set_xlim(0, 1)
ax.set_ylim(-0.04, 1)
ax.axis('off')
colors = {'prior': '#e8eef7', 'sim': '#e8eef7', 'obs': '#e8eef7', 'npe': '#eaf4ea', 'insp': '#fbeee6',
          'post': '#eaf4ea', 'check': '#f3f3f3', 'mech': '#fff3d6'}
for k, (x, y, w, h, txt) in BOXES.items():
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.01,rounding_size=0.015', fc=colors[k],
                                ec='0.35', lw=1.3 if k == 'mech' else 0.9))
    ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=8.2, linespacing=1.35,
            fontweight='normal')


def arrow(a, b, txt=None, dx=0.0, dy=0.03):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=12, lw=1.0, color='0.25'))
    if txt:
        ax.text((a[0] + b[0]) / 2 + dx, (a[1] + b[1]) / 2 + dy, txt, ha='center', fontsize=7.5, color='0.25')


arrow((0.19, 0.79), (0.24, 0.79), 'sample')
arrow((0.44, 0.79), (0.49, 0.79), 'fields')
arrow((0.69, 0.79), (0.74, 0.79), 'pairs')
arrow((0.69, 0.27), (0.74, 0.27), 'condition')
arrow((0.86, 0.64), (0.86, 0.42), 'trained once', dx=0.06)
arrow((0.59, 0.42), (0.59, 0.64), 'same operator', dx=0.07)
arrow((0.30, 0.64), (0.16, 0.44), 'campaign: maps,\nfields, onsets', dx=-0.07, dy=0.0)
# posterior queried on simulated and observed maps -> mechanism analysis and checks
ax.plot([0.86, 0.86, 0.13], [0.12, 0.02, 0.02], color='0.25', lw=1.0)
arrow((0.13, 0.02), (0.13, 0.10), None)
arrow((0.36, 0.02), (0.36, 0.10), None)
ax.text(0.62, -0.01, 'posterior queried on simulated and observed maps', ha='center', va='top', fontsize=7.5,
        color='0.25')
fig.tight_layout()
fig.savefig(sys.argv[1], dpi=600)
print('saved', sys.argv[1])

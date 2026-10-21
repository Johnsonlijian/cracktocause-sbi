"""Which hinge do reference simulations of the facade test form, for different shapes of the settlement profile?

usage: python profile_hinges.py OUT.json RESULT.npz [RESULT.npz ...]
RESULT.npz: outputs of fx_giardina.py (measured profile, measured profile without its uplift, hinge fit).
For every run, the crack map at the last record and at the experimental stages is classified by its
mechanism crack (crackcause.hinges); the experimental maps of Giardina et al. (2013, Fig. 5) are classified
in the same way.
"""
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
from crackcause import giardina as G  # noqa: E402
from crackcause.hinges import band_hinge, hinge_of, main_crack  # noqa: E402
from crackcause.observe_fx import Rasterizer, crack_lines, load_exp_maps  # noqa: E402

out, files = sys.argv[1], sys.argv[2:]
rep = dict(runs=[])
ex = load_exp_maps(os.path.join(HERE, '..', 'data_public', 'giardina2013_fig5_exp_crackmaps.npz'))


def classify(m):
    x, e = main_crack(m)
    hb, span = band_hinge(m)
    return dict(x_mm=float(x), extent=float(e), hinge=hinge_of(x, e), band_hinge=hb, span=span)


rep['experiment'] = {k: classify(m) for k, m in ex.items() if m.sum()}
for f in files:
    z = np.load(f)
    ras = Rasterizer(z['coords'], z['cells'][z['dmg'] > 0.5], (0.0, G.W, 0.0, G.H), (119, 143))
    name = os.path.basename(f)[:-4]
    prof = 'hinge' if name.endswith('_hinge') else 'measured_nouplift' if name.endswith('_measured_nouplift') else 'measured'
    m_seed = re.search(r'_seed(\d+)', name)
    run = dict(name=name, profile=prof, seed=int(m_seed.group(1)) if m_seed else 0, S_last=float(z['S'][-1]), stages={})
    for key, S_true in list(zip(['S2.5', 'S5', 'S7.5', 'S10'], G.EXP_STAGES_S)) + [('last', float(z['S'][-1]))]:
        j = int(np.argmin(np.abs(z['S'] - S_true)))
        if abs(z['S'][j] - S_true) > 0.3:
            continue
        run['stages'][key] = dict(S=float(z['S'][j]), **classify(crack_lines(ras(z['d'][j]), ras.inside, threshold=0.9)))
    rep['runs'].append(run)
    print(f"{name:60s} S_last {run['S_last']:5.2f}  " +
          '  '.join(f"{k}:{v['band_hinge']}({v['x_mm']:.0f},{v['span']['H1']:.2f}/{v['span']['H2']:.2f})"
                    for k, v in run['stages'].items()), flush=True)
summary = {}
for prof in ('measured', 'measured_nouplift', 'hinge'):
    rs = [r for r in rep['runs'] if r['profile'] == prof and 'last' in r['stages']]
    summary[prof] = dict(n=len(rs), **{h: sum(r['stages']['last']['band_hinge'] == h for r in rs)
                                       for h in ('H1', 'H2', 'none')},
                         seeds_H2=[r['seed'] for r in rs if r['stages']['last']['band_hinge'] == 'H2'])
rep['summary'] = summary
# paired comparison over the realizations of the strength field run with both profiles (McNemar, exact)
from scipy.stats import binomtest  # noqa: E402
last = {(r['profile'], r['seed']): r['stages']['last']['band_hinge'] for r in rep['runs'] if 'last' in r['stages']}
paired = {}
for other in ('hinge', 'measured_nouplift'):
    seeds = sorted(s for p, s in last if p == 'measured' and ('measured', s) in last and (other, s) in last)
    a = sum(last[('measured', s)] == 'H2' and last[(other, s)] != 'H2' for s in seeds)
    b = sum(last[('measured', s)] != 'H2' and last[(other, s)] == 'H2' for s in seeds)
    paired[other] = dict(n=len(seeds), only_measured_H2=int(a), only_other_H2=int(b),
                         both_H2=int(sum(last[('measured', s)] == 'H2' and last[(other, s)] == 'H2' for s in seeds)),
                         p_mcnemar=float(binomtest(a, a + b, 0.5).pvalue) if a + b else 1.0)
rep['paired_vs_measured'] = paired
json.dump(rep, open(out, 'w'), indent=1)
print(json.dumps(paired, indent=1))
print(json.dumps(dict(summary=summary, experiment=rep['experiment']), indent=1))

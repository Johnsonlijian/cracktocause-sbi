"""Statistics of a simulation campaign for the manuscript: outcome, run time, stages reached, flagged steps,
residuals of accepted steps.

usage: python campaign_stats.py DATA_DIR OUT.json [--facade]
"""
import argparse
import glob
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('data_dir')
ap.add_argument('out')
ap.add_argument('--facade', action='store_true', help='staged facade campaign (otherwise DEN)')
a = ap.parse_args()

rows = []
for f in sorted(glob.glob(os.path.join(a.data_dir, 'sims', 'sim_*.npz'))):
    try:
        z = np.load(f)
        log = json.loads(str(z['log']))
        th = json.loads(str(z['theta']))
    except Exception:                                   # file being written
        continue
    steps = [r for r in log if r.get('lf', 1.0) == 1.0 and 'S' in r and not r.get('failed', False)]
    rows.append(dict(status=str(z['status']), minutes=float(z['seconds']) / 60.0, theta=th,
                     n_stages=int(len(z['maps'])), S_last=float(z['S'][-1]) if a.facade and len(z['S']) else np.nan,
                     cracked=bool(len(z['maps']) and z['maps'][-1].sum() > 0),
                     flagged=sum(1 for r in log if r.get('unconverged', False)),
                     res=[r['res'] for r in steps if 'res' in r and np.isfinite(r['res'])],
                     widths='widths' in z.files, version=str(z['version']) if 'version' in z.files else ''))
n = len(rows)
st = {}
for r in rows:
    st[r['status']] = st.get(r['status'], 0) + 1
mins = np.array([r['minutes'] for r in rows])
crk = np.array([r['cracked'] for r in rows])
rep = dict(n=n, status=st, cpu_hours=float(mins.sum() / 60),
           minutes=dict(median=float(np.median(mins)), mean=float(mins.mean()), q90=float(np.quantile(mins, 0.9)),
                        max=float(mins.max()),
                        median_cracked=float(np.median(mins[crk])) if crk.any() else None,
                        median_uncracked=float(np.median(mins[~crk])) if (~crk).any() else None),
           cracked_fraction=float(crk.mean()),
           sims_with_flagged_steps=int(sum(1 for r in rows if r['flagged'] > 0)),
           flagged_steps=int(sum(r['flagged'] for r in rows)),
           sims_with_widths=int(sum(r['widths'] for r in rows)))
# residuals of accepted states are logged correctly from simulator version 2026-10-12b on
ok_res = [r['res'] for r in rows if r['res'] and r['version'] >= '2026-10-12b']
res = np.concatenate(ok_res) if ok_res else np.array([])
if len(res):
    rep['accepted_step_residual_N'] = dict(n=int(len(res)), median=float(np.median(res)),
                                           frac_le_6N=float((res <= 6.0 + 1e-9).mean()), frac_le_30N=float((res <= 30.0).mean()),
                                           frac_le_60N=float((res <= 60.0).mean()), max=float(res.max()))
if a.facade:
    S_last = np.array([r['S_last'] for r in rows])
    kn = np.array([r['theta']['kn'] for r in rows])
    reach = {}
    for s in (2.26, 4.54, 6.81, 9.10, 12.0):
        reach[str(s)] = dict(all=float((S_last >= s - 1e-6).mean()),
                             kn_gt_0_3=float((S_last[kn > 0.3] >= s - 1e-6).mean()) if (kn > 0.3).any() else None)
    rep['fraction_reaching_S'] = reach
    q = np.quantile(kn, [0.25, 0.5, 0.75])
    rep['failed_by_kn_quartile'] = [float(np.mean([r['status'] == 'failed' for r, k in zip(rows, kn) if lo <= k < hi]))
                                    for lo, hi in zip([0, *q], [*q, np.inf])]
json.dump(rep, open(a.out, 'w'), indent=1)
print(json.dumps(rep, indent=1))

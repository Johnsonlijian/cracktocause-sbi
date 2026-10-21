"""Write the numbers quoted in the manuscript as LaTeX macros (manuscript/numbers.tex) from the result files.

usage: python make_numbers.py NPE_FACADE_DIR NPE_DEN_DIR RESULTS_DIR OUT.tex
NPE_*_DIR: outputs of train_npe.py; RESULTS_DIR: outputs of make_results.sh (campaign_*.json,
identifiability_kn.json, nearest_*.txt). Macros are named \\N<Something>; values are formatted for
the text (counts as integers, fractions as decimals, settlements in mm with one decimal).
"""
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_npe import CONFIGS  # noqa: E402

fac_dir, den_dir, res_dir, out = sys.argv[1:5]
M = {}


def put(name, value):
    assert re.fullmatch(r'[A-Za-z]+', name), name
    if isinstance(value, str):
        value = re.sub(r'^-(?=0\.0*$)', '', value)        # no sign on a rounded zero
        value = re.sub(r'^-(?=[\d.])', '$-$', value)     # a typeset minus sign for negative numbers
    M[name] = value


def f1(x):
    return f'{x:.1f}'


def f2(x):
    return f'{x:.2f}'


def ld(path):
    return json.load(open(path)) if os.path.exists(path) else None


# campaigns
cf, cd = ld(os.path.join(res_dir, 'campaign_facade.json')), ld(os.path.join(res_dir, 'campaign_den.json'))
if cf:
    put('NSimFacade', f"{cf['n']:,}".replace(',', '{,}'))
    put('NSimWidths', f"{cf['sims_with_widths']:,}".replace(',', '{,}'))
    put('CpuHoursFacade', f"{cf['cpu_hours']:.0f}")
    put('MinCracked', f"{cf['minutes']['median_cracked']:.0f}")
    put('MinUncracked', f"{cf['minutes']['median_uncracked']:.1f}")
    put('FracCrackedFacade', f"{100 * cf['cracked_fraction']:.0f}")
    put('NFailedFacade', str(cf['status'].get('failed', 0)))
    put('NFlaggedSims', str(cf['sims_with_flagged_steps']))
    put('ReachNineAll', f"{100 * cf['fraction_reaching_S']['9.1']['all']:.0f}")
    put('ReachNineStiff', f"{100 * cf['fraction_reaching_S']['9.1']['kn_gt_0_3']:.0f}")
    r = cf.get('accepted_step_residual_N')
    if r:
        put('FracStepsAtol', f"{100 * r['frac_le_6N']:.0f}")
        put('FracStepsThirty', f"{100 * r['frac_le_30N']:.0f}")
        put('FracStepsSixty', f"{100 * r['frac_le_60N']:.1f}")
        put('NStepsRes', f"{r['n']:,}".replace(',', '{,}'))
        put('MaxStepResidual', f"{r['max']:.0f}")
if cd:
    put('NSimDen', f"{cd['n']:,}".replace(',', '{,}'))
    put('CpuHoursDen', f"{cd['cpu_hours']:.0f}")
    put('FracDenEarlyEnd', f"{100 * cd['status'].get('failed', 0) / cd['n']:.0f}")

# facade estimator: held-out calibration (map and map+w), stratified contraction
rep = ld(os.path.join(fac_dir, 'report.json'))
box = CONFIGS['facade']['box']
names = box.names
if rep:
    put('NPairsFacade', f"{rep['n_pairs']:,}".replace(',', '{,}'))
    put('NTestFacade', f"{rep['n_test']:,}".replace(',', '{,}'))
    for tag, key in (('', 'heldout'), ('W', 'heldout_w')):
        src = rep.get(key) or (rep if key == 'heldout' else None)        # older reports: top level
        if src is None:
            continue
        cov = src['coverage']['0.9']
        put(f'CovMin{tag}', f2(min(cov.values())))
        put(f'CovMax{tag}', f2(max(cov.values())))
        z = np.load(os.path.join(fac_dir, f'{key}.npz'))
        c = 1.0 - (z['q95'] - z['q05']) / (0.9 * (box.high - box.low))
        crk = z['cracked'].astype(bool)
        sel = z['has_width'].astype(bool) if tag == 'W' and 'has_width' in z.files else np.ones(len(c), bool)
        for n, short in (('S', 'S'), ('x_p', 'Xp'), ('log_kn', 'Kn'), ('log_w', 'W')):
            j = names.index(n)
            put(f'Con{short}Crk{tag}', f2(c[crk & sel, j].mean()))
            put(f'Con{short}Unc{tag}', f2(c[~crk & sel, j].mean()))
        lo, hi = z['q05'][:, 0], z['q95'][:, 0]
        put(f'WidthSCrk{tag}', f1((hi - lo)[crk & sel].mean()))
        put(f'CovSCrk{tag}', f2(((z['theta'][:, 0] >= lo) & (z['theta'][:, 0] <= hi))[crk & sel].mean()))
    # same pairs, map only, for the comparison with map+w
    if 'heldout_w' in rep:
        zw = np.load(os.path.join(fac_dir, 'heldout_w.npz'))
        z0 = np.load(os.path.join(fac_dir, 'heldout.npz'))
        sel = zw['has_width'].astype(bool) & zw['cracked'].astype(bool)
        put('NPairsWidthCrk', str(int(sel.sum())))
        put('WidthSCrkSameMap', f1((z0['q95'][:, 0] - z0['q05'][:, 0])[sel].mean()))
        put('WidthSCrkSameW', f1((zw['q95'][:, 0] - zw['q05'][:, 0])[sel].mean()))
    exp = rep.get('experiment', {})
    for key, tag in (('S2.5', 'One'), ('S5', 'Two'), ('S7.5', 'Three'), ('S10', 'Four'), ('S10_w', 'FourW')):
        if key not in exp:
            continue
        for n, short, fmt in (('S', 'S', f1), ('x_p', 'Xp', lambda v: f'{v:.0f}'), ('log_kn', 'Kn', lambda v: f2(np.exp(v)))):
            d = exp[key][n]
            put(f'Exp{tag}{short}Med', fmt(d['median']))
            put(f'Exp{tag}{short}Lo', fmt(d['q05']))
            put(f'Exp{tag}{short}Hi', fmt(d['q95']))
        put(f'Ood{tag}', f"{rep['ood_percentile'][key]:.0f}")
    # which stages' 90% intervals of S contain the imposed settlement (sentence for the results)
    stages = [('S2.5', '2.5'), ('S5', '5'), ('S7.5', '7.5'), ('S10', '10')]
    inside = [nom for k, nom in stages if k in exp and exp[k]['S']['q05'] <= exp[k]['S']['true'] <= exp[k]['S']['q95']]
    above = [(nom, exp[k]['S']['true']) for k, nom in stages if k in exp and exp[k]['S']['q05'] > exp[k]['S']['true']]

    def join(items):
        return items[0] if len(items) == 1 else ', '.join(items[:-1]) + ' and ' + items[-1]
    txt = (f'contain the imposed settlement at the {"stage" if len(inside) == 1 else "stages"} of {join(inside)}~mm'
           if inside else 'contain the imposed settlement at none of the stages')
    if above:
        txt += (f', whereas at {join([a for a, _ in above])}~mm they lie above it '
                f'({join([f"{t:.2f}" for _, t in above])}~mm)')
    put('ExpSCoverText', txt)
    if 'experiment_widths' in rep and rep['experiment_widths'].get('S10') is not None:
        put('ExpWmaxFour', f1(rep['experiment_widths']['S10']))

wi = ld(os.path.join(res_dir, 'width_information.json'))
if wi:
    put('RsqWidth', f2(wi['r2_width']))
    put('RsqLength', f2(wi['r2_length']))
    put('RsqWidthAllTrue', f2(wi['r2_width_all_true']))
    put('NMapsWidthInfo', f"{wi['n_maps']:,}".replace(',', '{,}'))
    b = wi['wmax_by_S']['6-9']
    put('WmaxSixNineLo', f1(b['q10']))
    put('WmaxSixNineHi', f1(b['q90']))

pp = ld(os.path.join(res_dir, 'ppc_S10.json'))
if pp:
    d = np.array(pp['chamfer_obs_mm'], float)
    w = np.array(pp['wmax_pred'], float)
    put('PpcN', str(pp['n']))
    put('PpcNClose', str(int((d < 100.0).sum())))
    put('PpcDistMed', f"{pp['chamfer_obs_median']:.0f}")
    put('PpcDistBetween', f"{pp['chamfer_between_median']:.0f}")
    put('PpcDistCloseLo', f"{d[d < 100.0].min():.0f}" if (d < 100.0).any() else '--')
    put('PpcDistCloseHi', f"{d[d < 100.0].max():.0f}" if (d < 100.0).any() else '--')
    put('PpcDistFarLo', f"{d[d >= 100.0].min():.0f}" if (d >= 100.0).any() else '--')
    put('PpcDistFarHi', f"{d[d >= 100.0].max():.0f}" if (d >= 100.0).any() else '--')
    put('PpcWmaxLo', f1(w.min()))
    put('PpcWmaxHi', f1(w.max()))
    put('PpcNFar', str(int((d >= 100.0).sum())))
    put('PpcNUncracked', str(int(np.isnan(d).sum())))                  # predicted maps without cracks
    if pp.get('wmax_obs') is not None:
        put('PpcWmaxPct', f"{100 * np.mean(w < pp['wmax_obs']):.0f}")

bu = ld(os.path.join(res_dir, 'budget.json'))
if bu:
    rows = [f"{r['n_train_sims']} & {r['coverage90_mean']:.2f} & {r['contraction_cracked_x_p']:.2f} & "
            f"{r['contraction_cracked_log_kn']:.2f} & {r['contraction_cracked_S']:.2f}" + r' \\' for r in bu]
    put('BudgetTableRows', '\n'.join(rows))
    put('BudgetNSmall', str(bu[0]['n_train_sims']))
    put('BudgetConXpSmall', f2(bu[0]['contraction_cracked_x_p']))
    put('BudgetConXpMid', f2(bu[1]['contraction_cracked_x_p']))
    put('BudgetNMid', str(bu[1]['n_train_sims']))

aux = ld(os.path.join(res_dir, 'identifiability_kn.json'))
if aux:
    put('ConSCrkKnKnown', f2(aux['map_and_kn']['S']['cracked']['contraction90']))
    put('ConSCrkKnMapOnly', f2(aux['map_only']['S']['cracked']['contraction90']))

# hinges of reference runs with three shapes of the settlement profile (examples/profile_hinges.py)
pf = ld(os.path.join(res_dir, 'profile_hinges.json'))
if pf:
    put('ProfExpSpan', f"{100 * pf['experiment']['S10']['span']['H2']:.0f}")
    for prof, tag in (('measured', 'Measured'), ('hinge', 'Fit'), ('measured_nouplift', 'NoUplift')):
        put(f'ProfN{tag}', str(pf['summary'][prof]['n']))
        put(f'ProfHTwo{tag}', str(pf['summary'][prof]['H2']))
    for other, tag in (('hinge', 'Fit'), ('measured_nouplift', 'NoUplift')):
        pv = pf['paired_vs_measured'][other]
        put(f'ProfPairN{tag}', str(pv['n']))
        put(f'ProfOnlyMeasured{tag}', str(pv['only_measured_H2']))
        put(f'ProfOnlyOther{tag}', str(pv['only_other_H2']))
        put(f'ProfBoth{tag}', str(pv['both_H2']))
        put(f'ProfP{tag}', f"{pv['p_mcnemar']:.2g}")

# conditional estimators (examples/conditional_npe.py)
ck = ld(os.path.join(res_dir, 'conditional_kn.json'))
if ck:
    put('AuxConSMap', f2(ck['map']['cracked']['contraction90']['S']))
    put('AuxConSKn', f2(ck['map+kn']['cracked']['contraction90']['S']))
    put('AuxConKnMap', f2(ck['map']['cracked']['contraction90']['log_kn']))
    put('AuxConKnKn', f2(ck['map+kn']['cracked']['contraction90']['log_kn']))
    put('AuxCovSKn', f2(ck['map+kn']['cracked']['coverage90']['S']))
    put('AuxCovSMap', f2(ck['map']['cracked']['coverage90']['S']))
# paired fine-mesh campaign (examples/mesh_mechanisms.py, examples/mesh_posterior.py)
mm = ld(os.path.join(res_dir, 'mesh_mechanisms.json'))
if mm:
    put('MeshNPairs', str(mm['n_pairs']))
    put('MeshNFailed', str(mm['n_failed_h5']))
    put('MeshNCrackBoth', str(mm['n_cracking_both']))
    put('MeshNCrackOne', str(mm['n_cracking_one_mesh']))
    put('MeshOnsetOne', f"{100 * mm['onset_within_one_record']:.0f}")
    put('MeshNLineBoth', str(mm['n_line_both']))
    put('MeshLineSame', str(mm['line_same']))
    put('MeshNLineOne', str(mm['n_line_one_mesh']))
    put('MeshNHingeBoth', str(mm['n_hinge_both']))
    put('MeshHingeSame', str(mm['hinge_same_when_both']))
    put('MeshChamferMed', f"{mm['chamfer_last_mm_median']:.0f}")
    xs = sorted(f"{d['x_p']:.0f}" for d in mm.get('line_diff', []))
    put('MeshDiffXp', ' and '.join([', '.join(xs[:-1]), xs[-1]]) if len(xs) > 1 else (xs[0] if xs else '---'))
    put('MeshNDiff', str(len(xs)))
    put('MeshChamferFifty', f"{100 * mm['chamfer_last_within_50mm']:.0f}")
mp = ld(os.path.join(res_dir, 'mesh_posterior.json'))
if mp:
    put('MeshPostNTrain', str(mp['n_train_sims']))
    put('MeshPostNPairs', str(mp['n_pair_sims']))
    put('MeshPostNMaps', f"{mp['n_pair_maps']:,}".replace(',', '{,}'))
    for tag, key in (('Coarse', 'h10'), ('Fine', 'h5')):
        cv = mp[key]['coverage90'].values()
        put(f'MeshCov{tag}Lo', f2(min(cv)))
        put(f'MeshCov{tag}Hi', f2(max(cv)))
        put(f'MeshConXp{tag}', f2(mp[key]['contraction_cracked']['x_p']))
        put(f'MeshConS{tag}', f2(mp[key]['contraction_cracked']['S']))
        put(f'MeshCovXp{tag}', f2(mp[key]['coverage90_cracked']['x_p']))
        put(f'MeshCovS{tag}', f2(mp[key]['coverage90_cracked']['S']))
    pm = mp['paired_maps']
    put('MeshDXp', f"{pm['median_abs_diff_xp_median_mm']:.0f}")
    put('MeshDS', f1(pm['median_abs_diff_S_median_mm']))

# noise regularization of S (examples/noise_sensitivity.py)
ns = ld(os.path.join(res_dir, 'noise_sensitivity.json'))
if ns:
    put('AuxNoiseConS', f2(ns['noise_all']['cracked']['contraction_S']))
    put('AuxNoiseCovS', f2(ns['noise_all']['cracked']['coverage_S']))
    put('AuxNoNoiseConS', f2(ns['no_noise_on_S']['cracked']['contraction_S']))
    put('AuxNoNoiseCovS', f2(ns['no_noise_on_S']['cracked']['coverage_S']))

# model-free value of information for S (examples/value_of_information.py)
vo = ld(os.path.join(res_dir, 'value_of_information.json'))
if vo:
    tags = {'map': 'Map', 'map+kn': 'Kn', 'map+xp+w': 'Profile', 'map+kn+xp+w': 'KnProfile',
            'map+ft+Gf+q': 'Material', 'map+all': 'All'}
    for key, tag in tags.items():
        if key in vo['known_parameters']:
            kp = vo['known_parameters'][key]
            put(f'AuxVoi{tag}', f2(kp['contraction90']))
            put(f'AuxVoiCov{tag}', f2(kp['coverage90']))
            put(f'AuxVoiEarly{tag}', f2(kp['contraction90_by_dS'][0]))      # within 1 mm after the onset
            put(f'AuxVoiLate{tag}', f2(kp['contraction90_by_dS'][-1]))      # 8-12 mm after the onset
            put(f'AuxVoiUpper{tag}', f"{100 * kp['upper_at_prior_edge']:.0f}")
    for i, (key, tag) in enumerate((('chamfer0.5_length0.1', 'Strict'), ('chamfer1.0_length0.2', 'Loose'))):
        put(f'AuxVoiSameEarly{tag}', f2(vo['same_simulation'][key]['contraction90_by_dS'][0]))
        put(f'AuxVoiSameLate{tag}', f2(vo['same_simulation'][key]['contraction90_by_dS'][-1]))
    ss = vo['same_simulation']
    put('AuxVoiSameStrict', f2(ss['chamfer0.5_length0.1']['contraction90']))
    put('AuxVoiSameLoose', f2(ss['chamfer1.0_length0.2']['contraction90']))
    put('AuxVoiNMaps', f"{vo['n_maps']:,}".replace(',', '{,}'))

# image-space misspecification percentiles
for fn, prefix in (('nearest_facade.txt', 'Img'), ('nearest_den.txt', 'ImgDen')):
    p = os.path.join(res_dir, fn)
    if os.path.exists(p):
        for line in open(p, encoding='utf-8-sig'):
            m = re.match(r'(\S+): nearest chamfer ([\d.]+) px .*percentile (\d+)', line)
            if m:
                key = re.sub(r'[^A-Za-z]', '', re.sub(r'^S', '', m.group(1)).replace('2.5', 'One').replace('10', 'Four')
                             .replace('_front', 'Front').replace('_rear', 'Rear').replace('4a', 'A')
                             .replace('4b', 'B').replace('4c', 'C'))
                put(f'{prefix}Pct{key}', m.group(3))
                put(f'{prefix}Dist{key}', f"{10 * float(m.group(2)):.0f}" if prefix == 'Img' else f"{2 * float(m.group(2)):.0f}")

# DEN estimator
rd = ld(os.path.join(den_dir, 'report.json'))
if rd:
    put('NTestDen', str(rd['n_test']))
    h = rd.get('heldout', rd)
    put('ConPsDen', f2(h['contraction_90']['Ps']))
    put('CovPsDen', f2(h['coverage']['0.9']['Ps']))
    for key, d in rd.get('experiment', {}).items():
        tag = key.replace('4a', 'A').replace('4b', 'B').replace('4c', 'C').replace('_front', 'Front').replace('_rear', 'Rear')
        put(f'Ps{tag}Med', f1(d['Ps']['median']))
        put(f'Ps{tag}Lo', f1(d['Ps']['q05']))
        put(f'Ps{tag}Hi', f1(d['Ps']['q95']))
        put(f'OodDen{tag}', f"{rd['ood_percentile'][key]:.0f}")

# mechanisms of the facade campaign (examples/mechanisms.py)
me = ld(os.path.join(res_dir, 'mechanisms.json'))
if me:
    hg, on, kin, inf = me['hinge'], me['onset'], me['kinematics'], me['inference']
    put('MechNCracking', str(hg['n_cracking']))
    put('MechNMech', str(hg['n_mechanism']))
    put('MechNHOne', str(hg['counts']['H1']))
    put('MechNHTwo', str(hg['counts']['H2']))
    put('MechNOther', str(hg['counts']['other']))
    put('MechNLocal', str(hg['n_local']))
    put('MechLocalOnLine', f"{100 * hg['local_on_hinge_line']:.0f}")
    put('MechLocalExt', f"{100 * hg['local_extent_median']:.0f}")
    put('MechDetachedMed', f"{100 * kin['detached_at_onset_median']:.0f}")
    put('MechDetachedNinety', f"{100 * kin['detached_at_onset_q90']:.0f}")
    put('MechContactNinety', f"{100 * (1 - kin['detached_at_onset_q90']):.0f}")
    put('MechEdgeFrac', f"{100 * hg['frac_within_30mm_of_edge']:.0f}")
    put('MechEdgeUniform', f"{100 * hg['frac_within_30mm_uniform']:.0f}")
    put('MechEdgeMedian', f"{hg['median_dist_to_window_edge_mm']:.0f}")
    for h, tag in (('H1', 'One'), ('H2', 'Two')):
        put(f'MechBasin{tag}Lo', f"{hg['basin_xp_90'][h][0]:.0f}")
        put(f'MechBasin{tag}Hi', f"{hg['basin_xp_90'][h][1]:.0f}")
        b = inf['xp_basin'][h]
        put(f'MechBasinWidth{tag}', f"{b['basin_width90']:.0f}")
        put(f'MechPostWidth{tag}', f"{b['posterior_width90_mean']:.0f}")
        put(f'MechPostN{tag}', str(b['n_maps']))
    put('MechSwitch', f"{hg['switch_xp']:.0f}")
    put('MechSwitchLo', f"{hg['switch_xp_ci95'][0]:.0f}")
    put('MechSwitchHi', f"{hg['switch_xp_ci95'][1]:.0f}")
    put('MechSwitchScale', f"{hg['switch_scale']:.0f}")
    ts = hg.get('threshold_sensitivity')
    if ts:
        put('MechSensSwitchLo', f"{min(t['switch_xp'] for t in ts):.0f}")
        put('MechSensSwitchHi', f"{max(t['switch_xp'] for t in ts):.0f}")
        put('MechSensOnLineLo', f"{min(100 * t['n_on_lines'] / max(t['n_mechanism'], 1) for t in ts):.0f}")
        put('MechSensOnLineHi', f"{max(100 * t['n_on_lines'] / max(t['n_mechanism'], 1) for t in ts):.0f}")
    put('MechBandLo', f"{hg['switch_band_10_90'][0]:.0f}")
    put('MechBandHi', f"{hg['switch_band_10_90'][1]:.0f}")
    put('MechShiftW', f"{hg['switch_shift_per_sd']['w']['mm_per_sd']:.0f}")
    put('MechShiftKn', f"{hg['switch_shift_per_sd']['kn']['mm_per_sd']:.0f}")
    put('MechAccX', f"{100 * hg['accuracy_xp_only']:.0f}")
    put('MechAccAll', f"{100 * hg['accuracy_all']:.0f}")
    put('MechPHTwoExpFit', f"{100 * hg['p_H2_at_exp_fit']:.0f}")
    put('MechExpPostXp', f"{hg['exp_posterior_median_xp']:.0f}")
    put('MechPHTwoExpPost', f"{100 * hg['p_H2_at_exp_posterior']:.0f}")
    put('MechOnsetN', str(on['n']))
    put('MechOnsetCens', str(on['n_censored']))
    put('MechOnsetRsq', f2(on['pseudo_r2']))
    put('MechOnsetSigma', f2(on['sigma']))
    for p, tag in (('w', 'W'), ('x_p', 'Xp'), ('kn', 'Kn'), ('ft', 'Ft'), ('Gf', 'Gf'), ('q', 'Q')):
        put(f'MechA{tag}', f2(on['exponents'][p]).replace('-', '$-$'))
        put(f'MechA{tag}Lo', f2(on['exponents_ci95'][p][0]).replace('-', '$-$'))
        put(f'MechA{tag}Hi', f2(on['exponents_ci95'][p][1]).replace('-', '$-$'))
        put(f'MechShare{tag}', f"{100 * on['variance_share'][p]:.0f}")
    put('MechAFtWc', f2(on['ft_wc']['a_ft']))
    put('MechAFtWcLo', f2(on['ft_wc']['a_ft_ci95'][0]))
    put('MechAFtWcHi', f2(on['ft_wc']['a_ft_ci95'][1]))
    put('MechNaiveRsq', f2(on['naive']['r2']))
    put('MechNaiveAKn', f2(on['naive']['exponents']['kn']).replace('-', '$-$'))
    bq = on['beta_c_inv_q10_50_90']
    put('MechBetaInvLo', f"{bq[0]:.0f}")
    put('MechBetaInvMed', f"{bq[1]:.0f}")
    put('MechBetaInvHi', f"{bq[2]:.0f}")
    st = on['Sc_test']
    put('MechScTest', f1(st['median']))
    put('MechScTestLo', f1(st['q05']))
    put('MechScTestHi', f1(st['q95']))
    put('MechBetaInvTest', f"{st['beta_inv']:.0f}")
    put('MechLcWLo', f2(on['Lc_over_W_range'][0]))
    put('MechLcWHi', f2(on['Lc_over_W_range'][1]))
    for r_, tag in ((on['kn_regimes'][0], 'Soft'), (on['kn_regimes'][-1], 'Stiff')):
        put(f'MechReg{tag}A', f2(r_['a_kn']).replace('-', '$-$'))
        put(f'MechReg{tag}Lo', f2(r_['a_kn_ci95'][0]).replace('-', '$-$'))
        put(f'MechReg{tag}Hi', f2(r_['a_kn_ci95'][1]).replace('-', '$-$'))
        put(f'MechReg{tag}LcW', f1(r_['Lc_over_W']))
    put('MechKinNSims', str(kin['n_sims']))
    put('MechKinNStages', f"{kin['n_stages']:,}".replace(',', '{,}'))
    post_bins = [i for i, b0 in enumerate(kin['bins'][:-1]) if b0 >= 0]
    rr = [kin['rigid_r2_median'][i] for i in post_bins]
    put('MechRigidRsqLo', f2(min(rr)))
    put('MechRigidRsqHi', f2(max(rr)))
    put('MechWRotSlope', f2(kin['wmax_vs_rotH']['slope']))
    put('MechWRotInt', f1(kin['wmax_vs_rotH']['intercept']))
    put('MechWRotRsq', f2(kin['wmax_vs_rotH']['r2']))
    put('MechWdSRsq', f2(kin['r2_wmax_vs_dS']))
    put('MechWSRsq', f2(kin['r2_wmax_vs_S']))
    comp = dict(zip([f"{kin['bins'][i]}_{kin['bins'][i + 1]}" for i in range(len(kin['bins']) - 1)], kin['completion_median']))
    put('MechCompleteOne', f"{100 * comp['0_1']:.0f}")
    put('MechCompleteTwo', f"{100 * comp['1_2']:.0f}")
    put('MechCompleteFour', f"{100 * comp['2_4']:.0f}")
    put('MechCompleteEight', f"{100 * comp['4_8']:.0f}")
    put('MechCompleteTwelve', f"{100 * comp['8_12']:.0f}")
    cs = dict(zip(comp.keys(), inf['contraction_S']))
    put('MechConSOne', f2(cs['0_1']))
    put('MechConSTwelve', f2(cs['8_12']))
    put('MechConSBefore', f2(inf['contraction_S_never_cracking']).replace('-', '$-$'))
    put('MechSUpperEdge', f"{100 * inf['S_upper_at_prior_edge']:.0f}")
    put('MechSLoSmall', f1(inf['S_q05_median_small']))
    put('MechSLoLarge', f1(inf['S_q05_median_large']))
    sh = on['variance_share']
    put('MechShareOnsetSet', f"{100 * (sh['kn'] + sh['x_p'] + sh['w']):.0f}")

# macros used in the manuscript but not (yet) available are set to a visible placeholder
import glob  # noqa: E402
PREFIX = r'(?:Con|Cov|Exp|Ood|Img|Ppc|Ps|Budget|NSteps|NSim|NPairs|NTest|NFailed|NFlagged|NMaps|Cpu|Min|Frac|Reach|Width|Wmax|Rsq|Max|Mesh|Mech|Aux|Prof)'
for tex in glob.glob(os.path.join(HERE, '..', 'manuscript', 'sections', '*.tex')):
    for name in re.findall(r'\\(' + PREFIX + r'[A-Za-z]*)', open(tex, encoding='utf8').read()):
        M.setdefault(name, r'\textbf{??}')

with open(out, 'w', encoding='utf8') as fh:
    fh.write('% generated by scripts/make_numbers.py; do not edit\n')
    for k in sorted(M):
        fh.write(f'\\newcommand{{\\{k}}}{{{M[k]}}}\n')
print(f'{len(M)} macros written to {out}')

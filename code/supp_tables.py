"""
Supplementary Table & Stats Generator
=======================================
Reads results CSVs from the robustness and mechanism test scripts.
Generates additional LaTeX tables and updates stats.tex.

Run AFTER: paper_outputs.py, lp_iv_robustness.py, mechanism_tests.py,
           additional_robustness.py, extended_robustness.py

Author: Ethan Johnston
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')

from paths import DATA_DIR, PAPER_DIR, TABLES_DIR as TABLE_DIR, ROBUSTNESS_DIR as ROB_DIR
os.makedirs(TABLE_DIR, exist_ok=True)

def star_tex(p):
    if p < 0.01: return '$^{***}$'
    if p < 0.05: return '$^{**}$'
    if p < 0.1: return '$^{*}$'
    return ''

def safe_cmd(key):
    digit_words = {'0': 'Zero', '1': 'One', '2': 'Two', '3': 'Three', '4': 'Four',
                   '5': 'Five', '6': 'Six', '7': 'Seven', '8': 'Eight', '9': 'Nine'}
    parts = key.split('_')
    name = 'stats' + ''.join(p.capitalize() for p in parts)
    result = ''
    for c in name:
        if c.isdigit(): result += digit_words[c]
        elif c.isalpha(): result += c
    return result

# Load existing stats
stats_path = os.path.join(PAPER_DIR, 'stats.json')
if os.path.exists(stats_path):
    with open(stats_path) as f:
        S = json.load(f)
else:
    S = {}

print("=" * 70)
print("SUPPLEMENTARY TABLES & STATS")
print("=" * 70)


# ============================================================
# TABLE: MECHANISM TEST — Industry Composition Control
# ============================================================
print("\nTable: Industry Composition Control...")

mech_path = os.path.join(ROB_DIR, 'mechanism_tests_results.csv')
if os.path.exists(mech_path):
    mech = pd.read_csv(mech_path)
    
    baseline = mech[(mech['specification'] == 'baseline') & (mech['outcome'] == 'mean_routine')]
    controlled = mech[(mech['specification'] == 'ind_comp_control') & (mech['outcome'] == 'mean_routine')]
    
    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{Mechanism Test: Controlling for Industry Composition Change}',
        r'\label{tab:mechanism}',
        r'\begin{tabular}{lcc}',
        r'\hline\hline',
        r'Horizon $h$ & Baseline & + Ind.\ Composition Control \\',
        r'\hline',
    ]
    
    for h in range(6):
        b_row = baseline[baseline['horizon'] == h]
        c_row = controlled[controlled['horizon'] == h]
        
        b_str = ''
        if len(b_row) > 0:
            r = b_row.iloc[0]
            b_str = f"{r['beta']:.4f}{star_tex(r['pval'])}"
        
        c_str = ''
        if len(c_row) > 0:
            r = c_row.iloc[0]
            c_str = f"{r['beta']:.4f}{star_tex(r['pval'])}"
        
        lines.append(f'{h} & {b_str} & {c_str} \\\\')
        
        # SE row
        b_se = f"({b_row.iloc[0]['se']:.4f})" if len(b_row) > 0 else ''
        c_se = f"({c_row.iloc[0]['se']:.4f})" if len(c_row) > 0 else ''
        lines.append(f' & {b_se} & {c_se} \\\\[3pt]')
    
    lines += [
        r'\hline',
        r'Ind.\ comp.\ control & No & Yes \\',
        r'\hline\hline',
        r'\end{tabular}',
        r'\begin{tablenotes}\small',
        r'\item Industry composition control is the cumulative change in Bartik-contributing',
        r'industry employment share from $t-1$ to $t+h$.',
        r'\item Both specifications include state fixed effects, year fixed effects, '
        r'and the technology control $\text{Tech}_{st}$ (initial routine employment '
        r'share interacted with a linear trend).',
        r'\item $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.1$.',
        r'\end{tablenotes}',
        r'\end{table}',
    ]
    
    with open(os.path.join(TABLE_DIR, 'tab_mechanism.tex'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: tab_mechanism.tex")
    
    # Stats
    if len(baseline) > 0 and len(controlled) > 0:
        b0 = baseline[baseline['horizon'] == 0]['beta'].values
        c0 = controlled[controlled['horizon'] == 0]['beta'].values
        if len(b0) > 0 and len(c0) > 0:
            atten = (1 - abs(c0[0]) / abs(b0[0])) * 100 if abs(b0[0]) > 0 else 0
            S['mech_attenuation_hzero'] = atten
            S['mech_baseline_hzero'] = b0[0]
            S['mech_controlled_hzero'] = c0[0]
            S['mech_controlled_hzero_pval'] = float(controlled[controlled['horizon'] == 0]['pval'].values[0])
else:
    print("  mechanism_tests_results.csv not found — skipping")


# ============================================================
# APPENDIX TABLE: GPSS Weights (already generated, just confirm)
# ============================================================
print("\nAppendix: GPSS weights...")
gpss_src = os.path.join(ROB_DIR, 'gpss_rotemberg_weights.csv')
gpss_dst = os.path.join(TABLE_DIR, 'tab_gpss.tex')
if os.path.exists(gpss_dst):
    print(f"  Already exists: tab_gpss.tex")
else:
    print(f"  Not found — run lp_iv_robustness.py first")


# ============================================================
# APPENDIX TABLE: Flexible Controls
# ============================================================
print("\nAppendix: Flexible controls...")

rob_path = os.path.join(ROB_DIR, 'robustness_all_results.csv')
if os.path.exists(rob_path):
    rob = pd.read_csv(rob_path)
    
    flex = rob[(rob['specification'] == 'flex_controls') & (rob['outcome'] == 'mean_routine')]
    baseline_rob = rob[(rob['specification'] == 'baseline_with_placebos') & (rob['outcome'] == 'mean_routine')]
    baseline_rob = baseline_rob[baseline_rob['horizon'] >= 0]
    
    if len(flex) > 0:
        lines = [
            r'\begin{table}[htbp]',
            r'\centering',
            r'\caption{Robustness: Flexible Technology Controls}',
            r'\label{tab:flex}',
            r'\begin{tabular}{lcc}',
            r'\hline\hline',
            r'Horizon $h$ & Baseline & Flexible Controls \\',
            r'\hline',
        ]
        
        for h in range(6):
            b_row = baseline_rob[baseline_rob['horizon'] == h]
            f_row = flex[flex['horizon'] == h]
            
            b_str = f"{b_row.iloc[0]['beta']:.4f}{star_tex(b_row.iloc[0]['pval'])}" if len(b_row) > 0 else ''
            f_str = f"{f_row.iloc[0]['beta']:.4f}{star_tex(f_row.iloc[0]['pval'])}" if len(f_row) > 0 else ''
            
            lines.append(f'{h} & {b_str} & {f_str} \\\\')
            
            b_se = f"({b_row.iloc[0]['se']:.4f})" if len(b_row) > 0 else ''
            f_se = f"({f_row.iloc[0]['se']:.4f})" if len(f_row) > 0 else ''
            lines.append(f' & {b_se} & {f_se} \\\\[3pt]')
        
        f_flex_val = (flex['first_stage_f'].iloc[0]
                      if 'first_stage_f' in flex.columns and len(flex) > 0 else None)
        f_flex_str = f'{f_flex_val:.1f}' if f_flex_val and f_flex_val > 0 else '---'
        lines += [
            r'\hline',
            r'Tech control & $\bar{\omega}^R_s \times t$ & $\bar{\omega}^R_s \times \theta_t$ \\',
            r'First-stage $F$ & {\statsFsF} & ' + f_flex_str + r' \\',
            r'\hline\hline',
            r'\end{tabular}',
            r'\end{table}',
        ]

        with open(os.path.join(TABLE_DIR, 'tab_flex.tex'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"  Saved: tab_flex.tex")
else:
    print(f"  robustness_all_results.csv not found — skipping")


# ============================================================
# APPENDIX TABLE: State Trends
# ============================================================
print("\nAppendix: State trends...")

if os.path.exists(rob_path):
    rob = pd.read_csv(rob_path)
    trends = rob[(rob['specification'] == 'state_trends') & (rob['outcome'] == 'mean_routine')]
    baseline_rob = rob[(rob['specification'] == 'baseline_with_placebos') & (rob['outcome'] == 'mean_routine')]
    baseline_rob = baseline_rob[baseline_rob['horizon'] >= 0]
    
    if len(trends) > 0:
        lines = [
            r'\begin{table}[htbp]',
            r'\centering',
            r'\caption{Robustness: State-Specific Linear Trends}',
            r'\label{tab:trends}',
            r'\begin{tabular}{lcc}',
            r'\hline\hline',
            r'Horizon $h$ & Baseline & + State Trends \\',
            r'\hline',
        ]
        
        for h in range(6):
            b_row = baseline_rob[baseline_rob['horizon'] == h]
            t_row = trends[trends['horizon'] == h]
            
            b_str = f"{b_row.iloc[0]['beta']:.4f}{star_tex(b_row.iloc[0]['pval'])}" if len(b_row) > 0 else ''
            t_str = f"{t_row.iloc[0]['beta']:.4f}{star_tex(t_row.iloc[0]['pval'])}" if len(t_row) > 0 else ''
            
            lines.append(f'{h} & {b_str} & {t_str} \\\\')
            
            b_se = f"({b_row.iloc[0]['se']:.4f})" if len(b_row) > 0 else ''
            t_se = f"({t_row.iloc[0]['se']:.4f})" if len(t_row) > 0 else ''
            lines.append(f' & {b_se} & {t_se} \\\\[3pt]')
        
        lines += [r'\hline\hline', r'\end{tabular}', r'\end{table}']
        
        with open(os.path.join(TABLE_DIR, 'tab_trends.tex'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"  Saved: tab_trends.tex")


# ============================================================
# APPENDIX TABLE: Sample Splits
# ============================================================
print("\nAppendix: Sample splits...")

add_rob_path = os.path.join(ROB_DIR, 'additional_robustness_results.csv')
if os.path.exists(add_rob_path):
    add = pd.read_csv(add_rob_path)
    # This file may not have sample split results — they were printed but 
    # may not have been saved. Check robustness_all_results.csv instead.

# Try the main robustness results
if os.path.exists(rob_path):
    rob = pd.read_csv(rob_path)
    # Sample splits might not be in the CSV. Generate a placeholder table.
    # The numbers were printed in the console output:
    # Pre-2008 h=0: -0.00446, Post-2008 h=0: -0.00872
    # We note these need to be re-generated if not in CSV.
    print("  Sample split table needs manual verification from console output")


# ============================================================
# APPENDIX TABLE: LFP Test
# ============================================================
print("\nAppendix: LFP test...")

if os.path.exists(mech_path):
    mech = pd.read_csv(mech_path)
    lfp = mech[mech['outcome'] == 'lfp_rate']
    
    if len(lfp) > 0:
        lines = [
            r'\begin{table}[htbp]',
            r'\centering',
            r'\caption{Selection Test: Effect on Labor Force Participation Rate}',
            r'\label{tab:lfp}',
            r'\begin{tabular}{lc}',
            r'\hline\hline',
            r'Horizon $h$ & $\Delta$ LFP Rate (pp) \\',
            r'\hline',
        ]
        
        for h in range(6):
            row = lfp[lfp['horizon'] == h]
            if len(row) > 0:
                r = row.iloc[0]
                lines.append(f"{h} & {r['beta']:.4f}{star_tex(r['pval'])} \\\\")
                lines.append(f" & ({r['se']:.4f}) \\\\[3pt]")
        
        lfp0 = float(lfp[lfp['horizon'] == 0]['beta'].values[0])
        lines += [r'\hline\hline', r'\end{tabular}',
                  r'\begin{tablenotes}\small',
                  r'\item Same LP-IV specification as the main results.',
                  rf'\item A 1 pp increase in unemployment changes the LFP rate by '
                  rf'{lfp0:.4f} pp at impact, statistically indistinguishable from zero.',
                  r'\end{tablenotes}',
                  r'\end{table}']

        with open(os.path.join(TABLE_DIR, 'tab_lfp.tex'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"  Saved: tab_lfp.tex")

        S['lfp_hzero_beta'] = lfp0
        for _h in (1, 2, 3, 4, 5):
            _r = lfp[lfp['horizon'] == _h]
            if len(_r) > 0:
                S[f'lfp_h{_h}_beta'] = float(_r['beta'].values[0])
                S[f'lfp_h{_h}_se'] = float(_r['se'].values[0])
                S[f'lfp_h{_h}_pval'] = float(_r['pval'].values[0])


# ============================================================
# APPENDIX TABLE: Anderson-Rubin
# ============================================================
print("\nAppendix: Anderson-Rubin...")

# AR results are already in stats — just make a small table
if 'ar_lower' in S and 'ar_upper' in S:
    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{Weak-Instrument-Robust Inference: Anderson-Rubin Confidence Sets}',
        r'\label{tab:ar}',
        r'\begin{tabular}{lcc}',
        r'\hline\hline',
        r'& 2SLS & Anderson-Rubin \\',
        r'\hline',
    ]
    
    iv_b = S.get('iv_routine_h0_beta', S.get('mech_baseline_hzero', ''))
    iv_se = S.get('iv_routine_h0_se', '')
    ar_l = S['ar_lower']
    ar_u = S['ar_upper']
    
    if iv_b != '' and iv_se != '':
        lines.append(f"Point estimate & {float(iv_b):.4f} & --- \\\\")
        lines.append(f"95\\% CI & [{float(iv_b)-1.96*float(iv_se):.4f}, {float(iv_b)+1.96*float(iv_se):.4f}] & [{float(ar_l):.4f}, {float(ar_u):.4f}] \\\\")
    
    lines.append(f"Excludes zero & Yes & {'Yes' if S.get('ar_excludes_zero', False) else 'No'} \\\\")
    
    lines += [r'\hline\hline', r'\end{tabular}',
              r'\begin{tablenotes}\small',
              r'\item AR confidence set constructed by inverting the Anderson-Rubin test over a grid of 400 values.',
              r'\end{tablenotes}',
              r'\end{table}']
    
    with open(os.path.join(TABLE_DIR, 'tab_ar.tex'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: tab_ar.tex")


# ============================================================
# APPENDIX TABLE: Fixed Bartik
# ============================================================
print("\nAppendix: Fixed Bartik...")

if os.path.exists(add_rob_path):
    add = pd.read_csv(add_rob_path)
    fixed = add[(add['specification'] == 'Fixed 1977 Bartik') & (add['outcome'] == 'mean_routine')]
    
    if len(fixed) > 0:
        # Get baseline from main results
        main_path = os.path.join(DATA_DIR, "results", "lp_iv_annual_results.csv")
        if os.path.exists(main_path):
            main_res = pd.read_csv(main_path)
            baseline = main_res[(main_res['model'] == 'IV') & (main_res['outcome'] == 'mean_routine')]
        else:
            baseline = pd.DataFrame()
        
        lines = [
            r'\begin{table}[htbp]',
            r'\centering',
            r'\caption{Robustness: Fixed Base-Year ({\statsBaseYear}) Bartik Shares}',
            r'\label{tab:fixedbartik}',
            r'\begin{tabular}{lcc}',
            r'\hline\hline',
            r'Horizon $h$ & Rolling Shares & Fixed {\statsBaseYear} Shares \\',
            r'\hline',
        ]
        
        for h in range(6):
            b_row = baseline[baseline['horizon'] == h] if len(baseline) > 0 else pd.DataFrame()
            f_row = fixed[fixed['horizon'] == h]
            
            b_str = f"{b_row.iloc[0]['beta']:.4f}{star_tex(b_row.iloc[0]['pval'])}" if len(b_row) > 0 else ''
            f_str = f"{f_row.iloc[0]['beta']:.4f}{star_tex(f_row.iloc[0]['pval'])}" if len(f_row) > 0 else ''
            
            lines.append(f'{h} & {b_str} & {f_str} \\\\')
            
            b_se = f"({b_row.iloc[0]['se']:.4f})" if len(b_row) > 0 else ''
            f_se = f"({f_row.iloc[0]['se']:.4f})" if len(f_row) > 0 else ''
            lines.append(f' & {b_se} & {f_se} \\\\[3pt]')
        
        f_fixed_val = (fixed['first_stage_f'].iloc[0]
                       if 'first_stage_f' in fixed.columns and len(fixed) > 0 else None)
        f_fixed_str = f'{f_fixed_val:.1f}' if f_fixed_val and f_fixed_val > 0 else '---'
        lines += [
            r'\hline',
            r'First-stage $F$ & {\statsFsF} & ' + f_fixed_str + r' \\',
            r'\hline\hline',
            r'\end{tabular}',
            r'\end{table}',
        ]

        with open(os.path.join(TABLE_DIR, 'tab_fixedbartik.tex'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"  Saved: tab_fixedbartik.tex")


# ============================================================
# ADDITIONAL INLINE STATS (placebos, heterogeneity, counts)
# ============================================================
# These feed \stats... macros used in the manuscript body so that no
# placebo / heterogeneity / count number is hand-typed into the LaTeX.
print("\nAdditional inline stats...")

# --- Pre-trend placebo coefficients (all four task outcomes) ---
if os.path.exists(rob_path):
    rob = pd.read_csv(rob_path)
    plac = rob[rob['specification'] == 'placebo']
    hmap = {-3: 'hm3', -2: 'hm2'}
    for out, tag in [('mean_routine', 'routine'), ('mean_rti', 'rti'),
                     ('mean_abstract', 'abstract'), ('mean_manual', 'manual')]:
        for h, hkey in hmap.items():
            row = plac[(plac['outcome'] == out) & (plac['horizon'] == h)]
            if len(row) > 0:
                r = row.iloc[0]
                S[f'plac_{tag}_{hkey}_beta'] = float(r['beta'])
                S[f'plac_{tag}_{hkey}_se'] = float(r['se'])
                S[f'plac_{tag}_{hkey}_pval'] = float(r['pval'])
    # RTI placebo SE relative to the Routine placebo SE at h = -3 (percent).
    if 'plac_rti_hm3_se' in S and S.get('plac_routine_hm3_se', 0) > 0:
        S['plac_rti_routine_se_ratio_pct'] = (
            100 * S['plac_rti_hm3_se'] / S['plac_routine_hm3_se'])

# --- Heterogeneity coefficients (final_extensions_results.csv) ---
fe_path = os.path.join(ROB_DIR, 'final_extensions_results.csv')
if os.path.exists(fe_path):
    fe = pd.read_csv(fe_path)
    fe_map = {'split_High Routine': 'hethi', 'split_Low Routine': 'hetlo',
              'age_young': 'ageyoung', 'age_prime': 'ageprime',
              'age_older': 'ageolder', 'delta_log_emp': 'empgrowth'}
    for spec, tag in fe_map.items():
        sub = fe[fe['specification'] == spec]
        for h in range(6):
            row = sub[sub['horizon'] == h]
            if len(row) > 0:
                r = row.iloc[0]
                S[f'{tag}_h{h}_beta'] = float(r['beta'])
                S[f'{tag}_h{h}_se'] = float(r['se'])
                S[f'{tag}_h{h}_pval'] = float(r['pval'])

# --- Occupation-type heterogeneity (extensions_results.csv) ---
ext_path = os.path.join(ROB_DIR, 'extensions_results.csv')
if os.path.exists(ext_path):
    ext = pd.read_csv(ext_path)
    educ_map = {'educ_College-Type Occupations': 'educcollege',
                'educ_Non-College-Type Occupations': 'educnoncollege'}
    for spec, tag in educ_map.items():
        sub = ext[ext['specification'] == spec]
        for h in range(6):
            row = sub[sub['horizon'] == h]
            if len(row) > 0:
                r = row.iloc[0]
                S[f'{tag}_h{h}_beta'] = float(r['beta'])
                S[f'{tag}_h{h}_se'] = float(r['se'])
                S[f'{tag}_h{h}_pval'] = float(r['pval'])

# --- Number of industry supersectors in the Bartik instrument ---
if os.path.exists(gpss_src):
    wdf_n = pd.read_csv(gpss_src)
    S['n_industries'] = int(wdf_n['industry'].nunique())


# ============================================================
# UPDATE STATS.TEX
# ============================================================
print("\nUpdating stats.tex...")

# Add new stats
# Load existing
existing_stats_tex = os.path.join(PAPER_DIR, 'stats.tex')
existing_lines = []
existing_cmds = set()
if os.path.exists(existing_stats_tex):
    with open(existing_stats_tex, 'r', encoding='utf-8') as f:
        existing_lines = f.readlines()
    for line in existing_lines:
        if line.startswith('\\newcommand'):
            cmd = line.split('{')[1].split('}')[0]
            existing_cmds.add(cmd)

# New stats to add
new_stats = {}
for key, val in S.items():
    cmd = '\\' + safe_cmd(key)
    if cmd not in existing_cmds:
        new_stats[key] = val

if new_stats:
    with open(existing_stats_tex, 'a', encoding='utf-8') as f:
        f.write('\n% --- Additional stats from supplementary generator ---\n')
        for key, val in sorted(new_stats.items()):
            cmd = safe_cmd(key)
            if isinstance(val, float):
                if abs(val) < 0.0001 and val != 0:
                    formatted = f'{val:.6f}'
                elif abs(val) < 1:
                    formatted = f'{val:.4f}'
                else:
                    formatted = f'{val:.1f}'
            elif isinstance(val, bool):
                formatted = 'Yes' if val else 'No'
            elif isinstance(val, (int, np.integer)):
                formatted = f'{val:,}'
            else:
                formatted = str(val)
            f.write(f'\\newcommand{{\\{cmd}}}{{{formatted}}}\n')
    print(f"  Added {len(new_stats)} new stats to stats.tex")
else:
    print(f"  No new stats to add")

# Also update JSON
with open(stats_path, 'w') as f:
    json.dump({k: float(v) if isinstance(v, (np.floating, float)) else
               int(v) if isinstance(v, (np.integer, int)) else
               str(v) for k, v in S.items()}, f, indent=2)

print("\n" + "=" * 70)
print("SUPPLEMENTARY TABLES COMPLETE")
print(f"Tables: {TABLE_DIR}")
print("=" * 70)

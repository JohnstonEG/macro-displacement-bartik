"""
Generate LaTeX tables for extensions results.
"""

import pandas as pd
import numpy as np
import os

from paths import DATA_DIR, TABLES_DIR as TABLE_DIR, ROBUSTNESS_DIR as ROB_DIR
os.makedirs(TABLE_DIR, exist_ok=True)

def star_tex(p):
    if p < 0.01: return '$^{***}$'
    if p < 0.05: return '$^{**}$'
    if p < 0.1: return '$^{*}$'
    return ''

ext = pd.read_csv(os.path.join(ROB_DIR, 'extensions_results.csv'))
print(f"Loaded: {len(ext)} rows")
print(f"Specifications: {ext['specification'].unique()}")

# Also load baseline from main results
main_path = os.path.join(DATA_DIR, "results", "lp_iv_annual_results.csv")
main = pd.read_csv(main_path)
baseline = main[(main['model'] == 'IV') & (main['outcome'] == 'mean_routine')]

# ============================================================
# TABLE: Leave-One-Industry-Out
# ============================================================
loo_specs = [s for s in ext['specification'].unique() if s.startswith('LOO')]
loo_labels = {s: s.replace('LOO_drop_', 'Drop ') for s in loo_specs}

lines = [
    r'\begin{table}[htbp]',
    r'\centering',
    r'\caption{Leave-One-Industry-Out Bartik: Routine Task Content}',
    r'\label{tab:loo}',
    r'\small',
    r'\begin{tabular}{l' + 'c' * (1 + len(loo_specs)) + '}',
    r'\hline\hline',
    r'Horizon & Baseline & ' + ' & '.join([loo_labels[s] for s in loo_specs]) + r' \\',
    r'\hline',
]

for h in range(6):
    vals = []
    # Baseline
    b_row = baseline[baseline['horizon'] == h]
    if len(b_row) > 0:
        r = b_row.iloc[0]
        vals.append(f"{r['beta']:.4f}{star_tex(r['pval'])}")
    else:
        vals.append('')
    
    # LOO specs
    for spec in loo_specs:
        d = ext[(ext['specification'] == spec) & (ext['horizon'] == h)]
        if len(d) > 0:
            r = d.iloc[0]
            vals.append(f"{r['beta']:.4f}{star_tex(r['pval'])}")
        else:
            vals.append('')
    
    lines.append(f'{h} & ' + ' & '.join(vals) + r' \\')
    
    # SE row
    ses = []
    b_row = baseline[baseline['horizon'] == h]
    ses.append(f"({b_row.iloc[0]['se']:.4f})" if len(b_row) > 0 else '')
    for spec in loo_specs:
        d = ext[(ext['specification'] == spec) & (ext['horizon'] == h)]
        ses.append(f"({d.iloc[0]['se']:.4f})" if len(d) > 0 else '')
    lines.append(' & ' + ' & '.join(ses) + r' \\[3pt]')

# First-stage F row: baseline pulls the main first-stage F macro; each
# leave-one-out column reads the F saved by extensions.py.
loo_f_cells = [r'{\statsFsF}']
for spec in loo_specs:
    d = ext[ext['specification'] == spec]
    fval = (d['first_stage_f'].iloc[0]
            if 'first_stage_f' in d.columns and len(d) > 0 else None)
    loo_f_cells.append(f'{fval:.1f}' if fval is not None and fval == fval and fval > 0 else '---')

lines += [
    r'\hline',
    r'First-stage $F$ & ' + ' & '.join(loo_f_cells) + r' \\',
    r'\hline\hline',
    r'\end{tabular}',
    r'\begin{tablenotes}\small',
    r'\item Each column drops the named industry from the Bartik instrument and renormalizes shares.',
    r'\item $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.1$.',
    r'\end{tablenotes}',
    r'\end{table}',
]

with open(os.path.join(TABLE_DIR, 'tab_loo.tex'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print("Saved: tab_loo.tex")


# ============================================================
# TABLE: Education Split
# ============================================================
educ_specs = [s for s in ext['specification'].unique() if s.startswith('educ')]

lines = [
    r'\begin{table}[htbp]',
    r'\centering',
    r'\caption{Occupation-Type Heterogeneity: Routine Task Content}',
    r'\label{tab:educ}',
    r'\begin{tabular}{lcc}',
    r'\hline\hline',
    r'Horizon & College-Type Occ. & Non-College-Type Occ. \\',
    r'\hline',
]

for h in range(6):
    vals = []
    for spec in educ_specs:
        d = ext[(ext['specification'] == spec) & (ext['horizon'] == h)]
        if len(d) > 0:
            r = d.iloc[0]
            vals.append(f"{r['beta']:.4f}{star_tex(r['pval'])}")
        else:
            vals.append('')
    lines.append(f'{h} & ' + ' & '.join(vals) + r' \\')
    
    ses = []
    for spec in educ_specs:
        d = ext[(ext['specification'] == spec) & (ext['horizon'] == h)]
        ses.append(f"({d.iloc[0]['se']:.4f})" if len(d) > 0 else '')
    lines.append(' & ' + ' & '.join(ses) + r' \\[3pt]')

lines += [
    r'\hline\hline',
    r'\end{tabular}',
    r'\begin{tablenotes}\small',
    r'\item College-type: OCC2010 $<$ 3600 (management, professional, technical).',
    r'\item Non-college-type: OCC2010 $\geq$ 3600 (service, production, operators).',
    r'\item $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.1$.',
    r'\end{tablenotes}',
    r'\end{table}',
]

with open(os.path.join(TABLE_DIR, 'tab_educ.tex'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print("Saved: tab_educ.tex")


# ============================================================
# TABLE: Region × Year Trends
# ============================================================
region = ext[ext['specification'] == 'region_year_trends']

lines = [
    r'\begin{table}[htbp]',
    r'\centering',
    r'\caption{Robustness: Census Region $\times$ Year Trends}',
    r'\label{tab:region}',
    r'\begin{tabular}{lcc}',
    r'\hline\hline',
    r'Horizon & Baseline & + Region $\times$ Year \\',
    r'\hline',
]

for h in range(6):
    b_row = baseline[baseline['horizon'] == h]
    r_row = region[region['horizon'] == h]
    
    b_str = f"{b_row.iloc[0]['beta']:.4f}{star_tex(b_row.iloc[0]['pval'])}" if len(b_row) > 0 else ''
    r_str = f"{r_row.iloc[0]['beta']:.4f}{star_tex(r_row.iloc[0]['pval'])}" if len(r_row) > 0 else ''
    
    lines.append(f'{h} & {b_str} & {r_str} \\\\')
    
    b_se = f"({b_row.iloc[0]['se']:.4f})" if len(b_row) > 0 else ''
    r_se = f"({r_row.iloc[0]['se']:.4f})" if len(r_row) > 0 else ''
    lines.append(f' & {b_se} & {r_se} \\\\[3pt]')

region_f_val = (region['first_stage_f'].iloc[0]
                if 'first_stage_f' in region.columns and len(region) > 0 else None)
region_f_str = (f'{region_f_val:.1f}'
                if region_f_val is not None and region_f_val == region_f_val
                and region_f_val > 0 else '---')
lines += [
    r'\hline',
    r'First-stage $F$ & {\statsFsF} & ' + region_f_str + r' \\',
    r'\hline\hline',
    r'\end{tabular}',
    r'\end{table}',
]

with open(os.path.join(TABLE_DIR, 'tab_region.tex'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print("Saved: tab_region.tex")

print("\nAll extension tables generated.")

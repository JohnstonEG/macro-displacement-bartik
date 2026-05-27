"""
Final Extensions
=================
1. Employment growth as alternative endogenous variable
2. High vs Low routine state heterogeneity
3. Age group heterogeneity (requires CPS re-read)

Author: Ethan Johnston
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS
from scipy.stats import norm
import os
import warnings
warnings.filterwarnings('ignore')

from paths import DATA_DIR, ROBUSTNESS_DIR as ROB_DIR, TABLES_DIR as TABLE_DIR
os.makedirs(ROB_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

MAX_H = 5

# ============================================================
# HELPERS
# ============================================================
def dd(x, s, t):
    d = pd.DataFrame({'x': x, 's': s, 't': t})
    gm = d['x'].mean()
    return (d['x'] - d.groupby('s')['x'].transform('mean')
            - d.groupby('t')['x'].transform('mean') + gm).values

def run_2sls(y, endog, instr, exog_list, states, times):
    y_dd = dd(y, states, times)
    end_dd = dd(endog, states, times)
    ins_dd = dd(instr, states, times)
    exog_dd = [dd(x, states, times) for x in exog_list]
    Z = np.column_stack([ins_dd] + [e.reshape(-1,1) for e in exog_dd]) if exog_dd else ins_dd.reshape(-1,1)
    b1, _, _, _ = np.linalg.lstsq(Z, end_dd, rcond=None)
    end_hat = Z @ b1
    X2 = np.column_stack([end_hat] + [e.reshape(-1,1) for e in exog_dd]) if exog_dd else end_hat.reshape(-1,1)
    b2, _, _, _ = np.linalg.lstsq(X2, y_dd, rcond=None)
    X2a = np.column_stack([end_dd] + [e.reshape(-1,1) for e in exog_dd]) if exog_dd else end_dd.reshape(-1,1)
    resid = y_dd - X2a @ b2
    n, k = len(y_dd), X2.shape[1]
    us = np.unique(states)
    G = len(us)
    XtXi = np.linalg.pinv(X2.T @ X2)
    B = np.zeros((k, k))
    for sv in us:
        m = states == sv
        Xe = X2[m].T @ resid[m]
        B += np.outer(Xe, Xe)
    cor = (G/(G-1)) * ((n-1)/(n-k))
    V = cor * XtXi @ B @ XtXi
    se = np.sqrt(np.maximum(np.diag(V), 0))
    return b2[0], se[0], n

def run_lp(df_in, outcomes, horizons, exog_cols, endog_col='delta_urate', bartik_col='bartik_shock'):
    results = {}
    for out in outcomes:
        rows = []
        for h in horizons:
            lp_var = f'{out}_lp{h}'
            if lp_var not in df_in.columns: continue
            cols = [lp_var, endog_col, bartik_col] + exog_cols
            valid = df_in[cols].notna().all(axis=1)
            if valid.sum() < 100: continue
            try:
                b, se, n = run_2sls(
                    df_in.loc[valid, lp_var].values,
                    df_in.loc[valid, endog_col].values,
                    df_in.loc[valid, bartik_col].values,
                    [df_in.loc[valid, c].values for c in exog_cols],
                    df_in.loc[valid, 'STATEFIP'].values,
                    df_in.loc[valid, 'YEAR'].values
                )
                t = b / se if se > 0 else 0
                p = 2 * (1 - norm.cdf(abs(t)))
                rows.append({'horizon': h, 'beta': b, 'se': se, 'pval': p, 'n': n,
                             'ci_lower': b - 1.96*se, 'ci_upper': b + 1.96*se})
            except: pass
        if rows: results[out] = pd.DataFrame(rows)
    return results

def first_stage_f(df_in, endog_col, bartik_col, exog_cols):
    df_p = df_in.set_index(['STATEFIP', 'YEAR'])
    y = df_p[endog_col]
    x = df_p[[bartik_col] + exog_cols]
    valid = y.notna() & x.notna().all(axis=1)
    if valid.sum() < 100: return 0, 0, 0
    try:
        mod = PanelOLS(y[valid], x[valid], entity_effects=True, time_effects=True,
                       drop_absorbed=True, check_rank=False)
        res = mod.fit(cov_type='clustered', cluster_entity=True)
        if bartik_col in res.params.index:
            pi = res.params[bartik_col]
            se = res.std_errors[bartik_col]
            return (pi/se)**2, pi, int(valid.sum())
    except: pass
    return 0, 0, 0

def star(p):
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.1: return '*'
    return ''

def star_tex(p):
    if p < 0.01: return '$^{***}$'
    if p < 0.05: return '$^{**}$'
    if p < 0.1: return '$^{*}$'
    return ''

def wmean(g, v, w='total_employed'):
    d = g[[v, w]].dropna()
    if len(d) == 0 or d[w].sum() == 0: return np.nan
    return np.average(d[v], weights=d[w])

# ============================================================
# LOAD BASE DATA
# ============================================================
print("=" * 70)
print("FINAL EXTENSIONS")
print("=" * 70)

sm = pd.read_csv(os.path.join(DATA_DIR, "state_month_panel.csv"))
sa = sm.groupby(['STATEFIP', 'YEAR']).apply(
    lambda g: pd.Series({v: wmean(g, v) for v in
        ['mean_routine', 'share_routine'] if v in g.columns})
).reset_index()
sa['STATEFIP'] = sa['STATEFIP'].astype(int)
sa['YEAR'] = sa['YEAR'].astype(int)

laus = pd.read_csv(os.path.join(DATA_DIR, "laus_state_unemployment.csv"))
laus_a = laus.groupby(['STATEFIP', 'year']).agg(
    urate=('unemployment_rate', 'mean'),
    emp_total=('employment_total', 'mean'),
).reset_index().rename(columns={'year': 'YEAR'})
laus_a['STATEFIP'] = laus_a['STATEFIP'].astype(int)

# Log employment and its change
laus_a['log_emp'] = np.log(laus_a['emp_total'].clip(lower=1))
laus_a = laus_a.sort_values(['STATEFIP', 'YEAR'])
laus_a['delta_log_emp'] = laus_a.groupby('STATEFIP')['log_emp'].diff()

bartik = pd.read_csv(os.path.join(DATA_DIR, "bartik_instrument.csv"))
bartik['STATEFIP'] = bartik['STATEFIP'].astype(int)

df = sa.merge(laus_a, on=['STATEFIP', 'YEAR'], how='inner')
df = df.merge(bartik, on=['STATEFIP', 'YEAR'], how='inner')
df = df.sort_values(['STATEFIP', 'YEAR']).reset_index(drop=True)

df['delta_urate'] = df.groupby('STATEFIP')['urate'].diff()

# Tech control
base_yr = df['YEAR'].min() + 1
base_sh = df[df['YEAR'] == base_yr].groupby('STATEFIP')['share_routine'].mean()
df = df.merge(base_sh.rename('rs_init'), on='STATEFIP', how='left')
df['time_trend'] = df['YEAR'] - df['YEAR'].min()
df['tc_routine'] = df['rs_init'] * df['time_trend']

# LP outcomes
for v in ['mean_routine']:
    df[f'{v}_L1'] = df.groupby('STATEFIP')[v].shift(1)
    for h in range(MAX_H + 1):
        df[f'{v}_F{h}'] = df.groupby('STATEFIP')[v].shift(-h)
        df[f'{v}_lp{h}'] = df[f'{v}_F{h}'] - df[f'{v}_L1']

print(f"Base panel: {len(df):,} obs, {df['STATEFIP'].nunique()} states")


# ============================================================
# 1. EMPLOYMENT GROWTH AS ALTERNATIVE TREATMENT
# ============================================================
print("\n" + "=" * 70)
print("1. EMPLOYMENT GROWTH AS ALTERNATIVE TREATMENT")
print("=" * 70)

# First stage: Bartik -> delta_log_emp
f_emp, pi_emp, n_emp = first_stage_f(df, 'delta_log_emp', 'bartik_shock', ['tc_routine'])
print(f"  First stage (Bartik -> Δlog(emp)): F={f_emp:.1f}, pi={pi_emp:.4f}, N={n_emp:,}")

# For comparison: first stage with unemployment
f_u, pi_u, n_u = first_stage_f(df, 'delta_urate', 'bartik_shock', ['tc_routine'])
print(f"  First stage (Bartik -> ΔU):       F={f_u:.1f}, pi={pi_u:.4f}, N={n_u:,}")

# LP-IV with employment growth
if f_emp > 5:
    emp_results = run_lp(df, ['mean_routine'], list(range(MAX_H+1)), ['tc_routine'],
                          endog_col='delta_log_emp')
    
    # Baseline with unemployment for comparison
    urate_results = run_lp(df, ['mean_routine'], list(range(MAX_H+1)), ['tc_routine'],
                            endog_col='delta_urate')
    
    print(f"\n  {'h':>3s}  {'ΔU (baseline)':>25s}    {'Δlog(emp)':>25s}")
    for h in range(MAX_H + 1):
        u_str = '---'
        e_str = '---'
        if 'mean_routine' in urate_results:
            r = urate_results['mean_routine']
            row = r[r['horizon'] == h]
            if len(row) > 0:
                u_str = f"{row.iloc[0]['beta']:+.5f} ({row.iloc[0]['se']:.5f}){star(row.iloc[0]['pval']):>4s}"
        if 'mean_routine' in emp_results:
            r = emp_results['mean_routine']
            row = r[r['horizon'] == h]
            if len(row) > 0:
                e_str = f"{row.iloc[0]['beta']:+.5f} ({row.iloc[0]['se']:.5f}){star(row.iloc[0]['pval']):>4s}"
        print(f"  {h:3d}  {u_str}    {e_str}")
    
    # Note on interpretation
    print(f"\n  Note: coefficients have different scales.")
    print(f"    ΔU: effect of 1pp unemployment increase")
    print(f"    Δlog(emp): effect of 1% employment decline (sign flipped)")
    print(f"    If both significant, the result is not an artifact of the unemployment measure.")
else:
    emp_results = {}
    print(f"  Weak instrument for Δlog(emp) — skipping")


# ============================================================
# 2. HIGH vs LOW ROUTINE STATE SPLIT
# ============================================================
print("\n" + "=" * 70)
print("2. HIGH vs LOW ROUTINE STATE HETEROGENEITY")
print("=" * 70)

median_rs = df.groupby('STATEFIP')['rs_init'].first().median()
print(f"  Median initial routine share: {median_rs:.4f}")
print(f"  High-routine states: {(df.groupby('STATEFIP')['rs_init'].first() >= median_rs).sum()}")
print(f"  Low-routine states: {(df.groupby('STATEFIP')['rs_init'].first() < median_rs).sum()}")

split_results = {}

for label, mask_fn in [('High Routine', lambda d: d['rs_init'] >= median_rs),
                         ('Low Routine', lambda d: d['rs_init'] < median_rs)]:
    sub = df[mask_fn(df)].copy()
    sub = sub.sort_values(['STATEFIP', 'YEAR'])
    
    # Recompute delta_urate within subsample
    sub['delta_urate'] = sub.groupby('STATEFIP')['urate'].diff()
    
    # Recompute LP outcomes
    for v in ['mean_routine']:
        sub[f'{v}_L1'] = sub.groupby('STATEFIP')[v].shift(1)
        for h in range(MAX_H + 1):
            sub[f'{v}_F{h}'] = sub.groupby('STATEFIP')[v].shift(-h)
            sub[f'{v}_lp{h}'] = sub[f'{v}_F{h}'] - sub[f'{v}_L1']
    
    f_val, pi_val, n_val = first_stage_f(sub, 'delta_urate', 'bartik_shock', ['tc_routine'])
    print(f"\n  {label}: N={len(sub):,}, states={sub['STATEFIP'].nunique()}, F={f_val:.1f}")
    
    if f_val > 5:
        res = run_lp(sub, ['mean_routine'], list(range(MAX_H+1)), ['tc_routine'])
        if 'mean_routine' in res:
            split_results[label] = res['mean_routine']
            print(f"    Routine LP-IV:")
            for _, r in res['mean_routine'].iterrows():
                print(f"      h={int(r['horizon'])}: {r['beta']:+.5f} ({r['se']:.5f}){star(r['pval']):>4s}")
    else:
        print(f"    Weak instrument — skipping")


# ============================================================
# 3. AGE HETEROGENEITY (requires CPS re-read)
# ============================================================
print("\n" + "=" * 70)
print("3. AGE HETEROGENEITY")
print("=" * 70)
print("  Reading CPS for age-stratified task composition...")

occ_task = pd.read_csv(os.path.join(DATA_DIR, "occ2010_task_scores.csv"))
occ_task['OCC2010'] = occ_task['OCC2010'].astype(int)

CPS_DAT = os.path.join(DATA_DIR, "cps_00002.dat")
CHUNK = 1_000_000

# Need: YEAR, MONTH, STATEFIP, WTFINL, AGE, EMPSTAT, OCC2010
colspecs_age = [
    (0, 4),      # YEAR
    (9, 11),     # MONTH
    (48, 50),    # STATEFIP
    (52, 66),    # WTFINL
    (110, 112),  # AGE
    (124, 126),  # EMPSTAT
    (131, 135),  # OCC2010
]
colnames_age = ['YEAR', 'MONTH', 'STATEFIP', 'WTFINL', 'AGE', 'EMPSTAT', 'OCC2010']

age_groups = {
    'young': (16, 29),
    'prime': (30, 54),
    'older': (55, 99),
}

accum = {k: [] for k in age_groups}

reader = pd.read_fwf(CPS_DAT, colspecs=colspecs_age, names=colnames_age,
                      chunksize=CHUNK, dtype=str)

total_read = 0
for i, chunk in enumerate(reader):
    total_read += len(chunk)
    
    for col in ['YEAR', 'MONTH', 'STATEFIP', 'AGE', 'EMPSTAT', 'OCC2010']:
        chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
    chunk['WTFINL'] = pd.to_numeric(chunk['WTFINL'], errors='coerce') / 10000
    
    mask = (
        chunk['EMPSTAT'].isin([10, 12]) &
        (chunk['OCC2010'] > 0) & (chunk['OCC2010'] < 9800) &
        (chunk['STATEFIP'] >= 1) & (chunk['STATEFIP'] <= 56) &
        (chunk['WTFINL'] > 0) &
        (chunk['AGE'] >= 16)
    )
    kept = chunk[mask].copy()
    kept['OCC2010'] = kept['OCC2010'].astype(int)
    kept = kept.merge(occ_task, on='OCC2010', how='left')
    kept = kept[kept['routine'].notna()]
    
    for grp, (age_lo, age_hi) in age_groups.items():
        sub = kept[(kept['AGE'] >= age_lo) & (kept['AGE'] <= age_hi)]
        if len(sub) == 0: continue
        agg = sub.groupby(['STATEFIP', 'YEAR']).apply(
            lambda g: pd.Series({
                'mean_routine': np.average(g['routine'], weights=g['WTFINL']),
                'total_emp': g['WTFINL'].sum(),
                'n_obs': len(g),
            })
        ).reset_index()
        accum[grp].append(agg)
    
    if (i + 1) % 20 == 0:
        print(f"    Chunk {i+1}: {total_read:,} read")

print(f"  Total read: {total_read:,}")

def combine(acc):
    if not acc: return pd.DataFrame()
    c = pd.concat(acc, ignore_index=True)
    p = c.groupby(['STATEFIP', 'YEAR']).apply(
        lambda g: pd.Series({
            'mean_routine': np.average(g['mean_routine'], weights=g['total_emp']),
            'total_emp': g['total_emp'].sum(),
        })
    ).reset_index()
    p['STATEFIP'] = p['STATEFIP'].astype(int)
    p['YEAR'] = p['YEAR'].astype(int)
    return p

age_panels = {k: combine(v) for k, v in accum.items()}

for grp, panel in age_panels.items():
    print(f"  {grp}: {len(panel):,} state-year obs")

# Save
for grp, panel in age_panels.items():
    panel.to_csv(os.path.join(ROB_DIR, f'task_age_{grp}.csv'), index=False)

# Run LP-IV for each age group
age_results = {}
age_labels = {'young': 'Young (16-29)', 'prime': 'Prime Age (30-54)', 'older': 'Older (55+)'}

for grp, panel in age_panels.items():
    label = age_labels[grp]
    print(f"\n  --- {label} ---")
    
    if len(panel) == 0:
        print(f"    No data — skipping")
        continue
    
    rn = panel.rename(columns={'mean_routine': f'{grp}_routine'})
    mg = rn.merge(laus_a[['STATEFIP', 'YEAR', 'urate']], on=['STATEFIP', 'YEAR'], how='inner')
    mg = mg.merge(bartik, on=['STATEFIP', 'YEAR'], how='inner')
    mg = mg.merge(df[['STATEFIP', 'YEAR', 'rs_init', 'tc_routine']].drop_duplicates(),
                  on=['STATEFIP', 'YEAR'], how='left')
    mg = mg.sort_values(['STATEFIP', 'YEAR']).reset_index(drop=True)
    mg['delta_urate'] = mg.groupby('STATEFIP')['urate'].diff()
    
    rout_col = f'{grp}_routine'
    mg[f'{rout_col}_L1'] = mg.groupby('STATEFIP')[rout_col].shift(1)
    for h in range(MAX_H + 1):
        mg[f'{rout_col}_F{h}'] = mg.groupby('STATEFIP')[rout_col].shift(-h)
        mg[f'{rout_col}_lp{h}'] = mg[f'{rout_col}_F{h}'] - mg[f'{rout_col}_L1']
    
    f_val, _, n_val = first_stage_f(mg, 'delta_urate', 'bartik_shock', ['tc_routine'])
    print(f"    First stage: F={f_val:.1f}, N={n_val:,}")
    
    if f_val > 5:
        res = run_lp(mg, [rout_col], list(range(MAX_H+1)), ['tc_routine'])
        if rout_col in res:
            age_results[grp] = res[rout_col]
            for _, r in res[rout_col].iterrows():
                print(f"      h={int(r['horizon'])}: {r['beta']:+.5f} ({r['se']:.5f}){star(r['pval']):>4s}")


# ============================================================
# 4. PLOTS
# ============================================================
print("\n" + "=" * 70)
print("4. PLOTS")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Panel A: Employment growth vs Unemployment
ax = axes[0]
if 'mean_routine' in urate_results:
    d = urate_results['mean_routine']
    ax.plot(d['horizon'], d['beta'], 'o-', color='#b2182b', linewidth=2.5,
            markersize=5, label=r'$\Delta U$ (baseline)', alpha=0.85)
    ax.errorbar(d['horizon'], d['beta'],
                yerr=[d['beta']-d['ci_lower'], d['ci_upper']-d['beta']],
                fmt='none', ecolor='#b2182b', capsize=4, elinewidth=1.5, zorder=2)
if emp_results and 'mean_routine' in emp_results:
    d = emp_results['mean_routine']
    ax.plot(d['horizon'], d['beta'], 's--', color='#4393c3', linewidth=2,
            markersize=5, label=r'$\Delta \log(emp)$', alpha=0.85)
    ax.errorbar(d['horizon'], d['beta'],
                yerr=[d['beta']-d['ci_lower'], d['ci_upper']-d['beta']],
                fmt='none', ecolor='#4393c3', capsize=3, elinewidth=1.2, alpha=0.85, zorder=2)
ax.axhline(0, color='gray', linewidth=0.8)
ax.set_xlabel('Horizon (years)'); ax.set_ylabel(r'$\hat{\beta}_h$')
ax.set_title('A: Alternative Treatment Variable', fontweight='bold')
ax.legend(fontsize=8); ax.grid(True, alpha=0.2); ax.set_xticks(range(MAX_H+1))

# Panel B: High vs Low routine
ax = axes[1]
split_colors = {'High Routine': '#b2182b', 'Low Routine': '#4393c3'}
for label, d in split_results.items():
    ax.plot(d['horizon'], d['beta'], 'o-', color=split_colors.get(label, 'gray'),
            linewidth=2, markersize=5, label=label, alpha=0.85)
    ax.errorbar(d['horizon'], d['beta'],
                yerr=[d['beta']-d['ci_lower'], d['ci_upper']-d['beta']],
                fmt='none', ecolor=split_colors.get(label, 'gray'),
                capsize=3, elinewidth=1.2, alpha=0.85, zorder=2)
ax.axhline(0, color='gray', linewidth=0.8)
ax.set_xlabel('Horizon (years)'); ax.set_ylabel(r'$\hat{\beta}_h$')
ax.set_title('B: High vs Low Routine States', fontweight='bold')
ax.legend(fontsize=8); ax.grid(True, alpha=0.2); ax.set_xticks(range(MAX_H+1))

# Panel C: Age groups
ax = axes[2]
age_colors = {'young': '#4393c3', 'prime': '#b2182b', 'older': '#1b7837'}
for grp, d in age_results.items():
    ax.plot(d['horizon'], d['beta'], 'o-', color=age_colors.get(grp, 'gray'),
            linewidth=2, markersize=5, label=age_labels[grp], alpha=0.85)
    ax.errorbar(d['horizon'], d['beta'],
                yerr=[d['beta']-d['ci_lower'], d['ci_upper']-d['beta']],
                fmt='none', ecolor=age_colors.get(grp, 'gray'),
                capsize=3, elinewidth=1.2, alpha=0.85, zorder=2)
ax.axhline(0, color='gray', linewidth=0.8)
ax.set_xlabel('Horizon (years)'); ax.set_ylabel(r'$\hat{\beta}_h$')
ax.set_title('C: Age Group Heterogeneity', fontweight='bold')
ax.legend(fontsize=8); ax.grid(True, alpha=0.2); ax.set_xticks(range(MAX_H+1))

plt.tight_layout()
path = os.path.join(ROB_DIR, 'fig_final_extensions.pdf')
fig.savefig(path, dpi=300, bbox_inches='tight')
print(f"  Saved: {path}")
plt.close()


# ============================================================
# 5. LATEX TABLES
# ============================================================
print("\n" + "=" * 70)
print("5. LATEX TABLES")
print("=" * 70)

# Table: Employment growth comparison
if emp_results and 'mean_routine' in emp_results:
    lines = [
        r'\begin{table}[htbp]', r'\centering',
        r'\caption{Alternative Treatment: $\Delta U$ vs $\Delta \log(\text{emp})$}',
        r'\label{tab:empgrowth}',
        r'\begin{tabular}{lcc}', r'\hline\hline',
        r'Horizon & $\Delta U$ (baseline) & $\Delta \log(\text{emp})$ \\', r'\hline',
    ]
    for h in range(MAX_H+1):
        vals, ses = [], []
        for src in [urate_results, emp_results]:
            if 'mean_routine' in src:
                row = src['mean_routine'][src['mean_routine']['horizon'] == h]
                if len(row) > 0:
                    r = row.iloc[0]
                    vals.append(f"{r['beta']:.4f}{star_tex(r['pval'])}")
                    ses.append(f"({r['se']:.4f})")
                else: vals.append(''); ses.append('')
            else: vals.append(''); ses.append('')
        lines.append(f'{h} & ' + ' & '.join(vals) + r' \\')
        lines.append(' & ' + ' & '.join(ses) + r' \\[3pt]')
    lines += [r'\hline',
              f'First-stage $F$ & {f_u:.1f} & {f_emp:.1f} \\\\',
              r'\hline\hline', r'\end{tabular}',
              r'\begin{tablenotes}\small',
              r'\item Coefficients have different scales: $\Delta U$ is per 1pp unemployment increase;',
              r'$\Delta \log(\text{emp})$ is per 1\% employment change.',
              r'\end{tablenotes}', r'\end{table}']
    with open(os.path.join(TABLE_DIR, 'tab_empgrowth.tex'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print("  Saved: tab_empgrowth.tex")

# Table: High vs Low routine
if split_results:
    lines = [
        r'\begin{table}[htbp]', r'\centering',
        r'\caption{Heterogeneity by Initial Routine Intensity}',
        r'\label{tab:routine_split}',
        r'\begin{tabular}{lcc}', r'\hline\hline',
        r'Horizon & High Routine & Low Routine \\', r'\hline',
    ]
    for h in range(MAX_H+1):
        vals, ses = [], []
        for label in ['High Routine', 'Low Routine']:
            if label in split_results:
                row = split_results[label][split_results[label]['horizon'] == h]
                if len(row) > 0:
                    r = row.iloc[0]
                    vals.append(f"{r['beta']:.4f}{star_tex(r['pval'])}")
                    ses.append(f"({r['se']:.4f})")
                else: vals.append(''); ses.append('')
            else: vals.append(''); ses.append('')
        lines.append(f'{h} & ' + ' & '.join(vals) + r' \\')
        lines.append(' & ' + ' & '.join(ses) + r' \\[3pt]')
    lines += [r'\hline\hline', r'\end{tabular}',
              r'\begin{tablenotes}\small',
              r'\item States split at median initial ({\statsBaseYear}) routine employment share.',
              r'\end{tablenotes}', r'\end{table}']
    with open(os.path.join(TABLE_DIR, 'tab_routine_split.tex'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print("  Saved: tab_routine_split.tex")

# Table: Age heterogeneity
if age_results:
    lines = [
        r'\begin{table}[htbp]', r'\centering',
        r'\caption{Age Group Heterogeneity: Routine Task Content}',
        r'\label{tab:age}',
        r'\begin{tabular}{l' + 'c' * len(age_results) + '}', r'\hline\hline',
        r'Horizon & ' + ' & '.join([age_labels[g] for g in age_results.keys()]) + r' \\', r'\hline',
    ]
    for h in range(MAX_H+1):
        vals, ses = [], []
        for grp in age_results:
            row = age_results[grp][age_results[grp]['horizon'] == h]
            if len(row) > 0:
                r = row.iloc[0]
                vals.append(f"{r['beta']:.4f}{star_tex(r['pval'])}")
                ses.append(f"({r['se']:.4f})")
            else: vals.append(''); ses.append('')
        lines.append(f'{h} & ' + ' & '.join(vals) + r' \\')
        lines.append(' & ' + ' & '.join(ses) + r' \\[3pt]')
    lines += [r'\hline\hline', r'\end{tabular}', r'\end{table}']
    with open(os.path.join(TABLE_DIR, 'tab_age.tex'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print("  Saved: tab_age.tex")


# ============================================================
# 6. EXPORT
# ============================================================
print("\n" + "-" * 50)
print("6. Export")
print("-" * 50)

all_rows = []
if emp_results and 'mean_routine' in emp_results:
    o = emp_results['mean_routine'].copy()
    o['specification'] = 'delta_log_emp'; o['outcome'] = 'mean_routine'
    all_rows.append(o)
for label, d in split_results.items():
    o = d.copy(); o['specification'] = f'split_{label}'; o['outcome'] = 'mean_routine'
    all_rows.append(o)
for grp, d in age_results.items():
    o = d.copy(); o['specification'] = f'age_{grp}'; o['outcome'] = 'mean_routine'
    all_rows.append(o)

if all_rows:
    out = pd.concat(all_rows, ignore_index=True)
    p = os.path.join(ROB_DIR, 'final_extensions_results.csv')
    out.to_csv(p, index=False, float_format='%.6f')
    print(f"  Saved: {p}")

print("\n" + "=" * 70)
print("FINAL EXTENSIONS COMPLETE")
print("=" * 70)

"""
High-Value Extensions
======================
1. Leave-One-Industry-Out Bartik (drop top Rotemberg-weight industries)
2. Demographic Heterogeneity (college vs non-college LP-IV)
3. Census-Region × Year Trends

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

from paths import DATA_DIR, ROBUSTNESS_DIR as ROB_DIR
os.makedirs(ROB_DIR, exist_ok=True)

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

def run_lp(df_in, outcomes, horizons, exog_cols, bartik_col='bartik_shock'):
    results = {}
    for out in outcomes:
        rows = []
        for h in horizons:
            lp_var = f'{out}_lp{h}'
            if lp_var not in df_in.columns: continue
            cols = [lp_var, 'delta_urate', bartik_col] + exog_cols
            valid = df_in[cols].notna().all(axis=1)
            if valid.sum() < 100: continue
            try:
                b, se, n = run_2sls(
                    df_in.loc[valid, lp_var].values,
                    df_in.loc[valid, 'delta_urate'].values,
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

def first_stage_f(df_in, bartik_col, exog_cols):
    df_p = df_in.set_index(['STATEFIP', 'YEAR'])
    y = df_p['delta_urate']
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

def wmean(g, v, w='total_employed'):
    d = g[[v, w]].dropna()
    if len(d) == 0 or d[w].sum() == 0: return np.nan
    return np.average(d[v], weights=d[w])

# IND1990 -> supersector
def ind1990_to_supersector(code):
    # NOTE: must stay synchronized with bartik_shares.py mapping.
    # Agriculture/forestry/fishing (IND1990 10-32) intentionally dropped
    # because CES does not cover farm employment — mapping them to
    # "Mining and logging" was a measurement-error source in earlier runs.
    if code <= 0: return None
    if 10 <= code <= 32: return None  # ag/forestry/fishing — drop
    if 40 <= code <= 50: return "Mining and logging"
    if 60 <= code <= 60: return "Construction"
    if 100 <= code <= 222: return "Manufacturing - Nondurable goods"
    if 230 <= code <= 392: return "Manufacturing - Durable goods"
    if 400 <= code <= 432: return "Transportation and warehousing"
    if 440 <= code <= 442: return "Information"
    if 450 <= code <= 472: return "Utilities"
    if 500 <= code <= 571: return "Wholesale trade"
    if 580 <= code <= 691: return "Retail trade"
    if 700 <= code <= 712: return "Financial activities"
    if 721 <= code <= 760: return "Professional and business services"
    if 761 <= code <= 791: return "Other services"
    if 800 <= code <= 810: return "Leisure and hospitality"
    if 812 <= code <= 860: return "Education and health services"
    if 861 <= code <= 893: return "Professional and business services"
    if 900 <= code <= 932: return "Government"
    return None

# FIPS -> Census Region
FIPS_TO_REGION = {
    9: 'Northeast', 23: 'Northeast', 25: 'Northeast', 33: 'Northeast',
    34: 'Northeast', 36: 'Northeast', 42: 'Northeast', 44: 'Northeast', 50: 'Northeast',
    17: 'Midwest', 18: 'Midwest', 19: 'Midwest', 20: 'Midwest',
    26: 'Midwest', 27: 'Midwest', 29: 'Midwest', 31: 'Midwest',
    38: 'Midwest', 39: 'Midwest', 46: 'Midwest', 55: 'Midwest',
    10: 'South', 11: 'South', 12: 'South', 13: 'South',
    21: 'South', 22: 'South', 24: 'South', 28: 'South',
    37: 'South', 40: 'South', 45: 'South', 47: 'South',
    48: 'South', 51: 'South', 54: 'South',  1: 'South',  5: 'South',
    2: 'West', 4: 'West', 6: 'West', 8: 'West',
    15: 'West', 16: 'West', 30: 'West', 32: 'West',
    35: 'West', 41: 'West', 49: 'West', 53: 'West', 56: 'West',
}


# ============================================================
# LOAD BASE DATA
# ============================================================
print("=" * 70)
print("HIGH-VALUE EXTENSIONS")
print("=" * 70)

# Main panel
sm = pd.read_csv(os.path.join(DATA_DIR, "state_month_panel.csv"))
sa = sm.groupby(['STATEFIP', 'YEAR']).apply(
    lambda g: pd.Series({v: wmean(g, v) for v in
        ['mean_routine', 'mean_abstract', 'mean_manual', 'mean_rti', 'share_routine']
        if v in g.columns})
).reset_index()
sa['STATEFIP'] = sa['STATEFIP'].astype(int)
sa['YEAR'] = sa['YEAR'].astype(int)

laus = pd.read_csv(os.path.join(DATA_DIR, "laus_state_unemployment.csv"))
laus_a = laus.groupby(['STATEFIP', 'year']).agg(urate=('unemployment_rate', 'mean')).reset_index()
laus_a.rename(columns={'year': 'YEAR'}, inplace=True)
laus_a['STATEFIP'] = laus_a['STATEFIP'].astype(int)

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
# 1. LEAVE-ONE-INDUSTRY-OUT BARTIK
# ============================================================
print("\n" + "=" * 70)
print("1. LEAVE-ONE-INDUSTRY-OUT BARTIK")
print("=" * 70)

ind_shares = pd.read_csv(os.path.join(DATA_DIR, "state_industry_shares.csv"))
ces = pd.read_csv(os.path.join(DATA_DIR, "ces_national_industry.csv"))

ces_a = ces.groupby(['industry', 'year'])['employment'].mean().reset_index()
ces_a = ces_a.sort_values(['industry', 'year'])
ces_a['emp_lag'] = ces_a.groupby('industry')['employment'].shift(1)
ces_a['growth'] = (ces_a['employment'] - ces_a['emp_lag']) / ces_a['emp_lag']
ces_a = ces_a.dropna(subset=['growth'])

# Top 3 Rotemberg-weight industries to drop. Data-driven from the GPSS
# decomposition so the leave-one-out table never drifts from tab_gpss.tex;
# falls back to the historical list only if the GPSS file is missing.
_gpss_path = os.path.join(ROB_DIR, 'gpss_rotemberg_weights.csv')
if os.path.exists(_gpss_path):
    _gw = pd.read_csv(_gpss_path).sort_values('rotemberg_pct', ascending=False)
    drop_industries = _gw['industry'].head(3).tolist()
else:
    drop_industries = ['Mining and logging', 'Construction',
                       'Manufacturing - Durable goods']
print(f"  Top-3 Rotemberg-weight industries (LOO drops): {drop_industries}")

loo_results = {}
loo_fstats = {}

for drop_ind in drop_industries:
    print(f"\n  Dropping: {drop_ind}")
    
    # Rebuild Bartik without this industry
    ind_shares_loo = ind_shares[ind_shares['industry'] != drop_ind].copy()
    
    # Recompute shares (renormalize)
    totals = ind_shares_loo.groupby(['STATEFIP', 'YEAR'])['emp_weighted'].sum().reset_index()
    totals.columns = ['STATEFIP', 'YEAR', 'total_loo']
    ind_shares_loo = ind_shares_loo.merge(totals, on=['STATEFIP', 'YEAR'])
    ind_shares_loo['emp_share_loo'] = ind_shares_loo['emp_weighted'] / ind_shares_loo['total_loo']
    
    ces_a_loo = ces_a[ces_a['industry'] != drop_ind]
    
    bartik_loo_rows = []
    for year in sorted(df['YEAR'].unique()):
        base_year = year - 1
        base_sh_loo = ind_shares_loo[ind_shares_loo['YEAR'] == base_year][
            ['STATEFIP', 'industry', 'emp_share_loo']]
        if len(base_sh_loo) == 0: continue
        growth = ces_a_loo[ces_a_loo['year'] == year][['industry', 'growth']]
        if len(growth) == 0: continue
        merged = base_sh_loo.merge(growth, on='industry', how='inner')
        b = merged.groupby('STATEFIP').apply(
            lambda g: (g['emp_share_loo'] * g['growth']).sum()
        ).reset_index()
        b.columns = ['STATEFIP', 'bartik_loo']
        b['YEAR'] = year
        bartik_loo_rows.append(b)
    
    if not bartik_loo_rows:
        print(f"    Could not construct — skipping")
        continue
    
    bartik_loo = pd.concat(bartik_loo_rows, ignore_index=True)
    
    # Merge into main panel
    df_loo = df.copy()
    df_loo = df_loo.merge(bartik_loo, on=['STATEFIP', 'YEAR'], how='left')
    
    # First stage
    f_val, pi_val, n_val = first_stage_f(df_loo, 'bartik_loo', ['tc_routine'])
    loo_fstats[drop_ind] = f_val
    print(f"    First stage: F={f_val:.1f}, pi={pi_val:.4f}, N={n_val:,}")
    
    if f_val < 5:
        print(f"    Weak instrument — skipping LP-IV")
        continue
    
    # LP-IV
    res = run_lp(df_loo, ['mean_routine'], list(range(MAX_H+1)), ['tc_routine'], 'bartik_loo')
    
    if 'mean_routine' in res:
        loo_results[drop_ind] = res['mean_routine']
        print(f"    Routine LP-IV:")
        for _, r in res['mean_routine'].iterrows():
            print(f"      h={int(r['horizon'])}: {r['beta']:+.5f} ({r['se']:.5f}){star(r['pval']):>4s}")

# Baseline for comparison
baseline = run_lp(df, ['mean_routine'], list(range(MAX_H+1)), ['tc_routine'])


# ============================================================
# 2. DEMOGRAPHIC HETEROGENEITY (College vs Non-College)
# ============================================================
print("\n" + "=" * 70)
print("2. DEMOGRAPHIC HETEROGENEITY (Education Split)")
print("=" * 70)
print("  Reading CPS microdata for education-stratified task composition...")

# Need: STATEFIP, YEAR, MONTH, WTFINL, EMPSTAT, OCC2010, EDUC
colspecs = [
    (0, 4),      # YEAR
    (9, 11),     # MONTH
    (48, 50),    # STATEFIP
    (52, 66),    # WTFINL
    (112, 113),  # SEX
    (124, 126),  # EMPSTAT
    (131, 135),  # OCC2010
]

# EDUC position from SAS syntax: not in current colspecs
# From the SAS file: RELATE 107-110, AGE 111-112, SEX 113, RACE 114-116,
# MARST 117, POPSTAT 118, ASIAN 119-120, VETSTAT 121, HISPAN 122-124,
# EMPSTAT 125-126, LABFORCE 127, OCC 128-131, OCC2010 132-135
# EDUC is not in cps_00002! Check if available.
# Looking at the SAS syntax from our conversations, EDUC was in cps_00001
# but cps_00002 has: RELATE, AGE, SEX, RACE, MARST, POPSTAT, ASIAN, 
# VETSTAT, HISPAN, EMPSTAT, LABFORCE, OCC, OCC2010, OCC1990, IND1990, etc.
# No EDUC variable in cps_00002.

# Alternative: use OCC2010 as a proxy for education level
# Occupations with OCC2010 < 3600 are roughly "college" occupations
# (management, professional, technical) and >= 3600 are "non-college"
# This is imperfect but standard when EDUC is unavailable

print("  Note: EDUC not in cps_00002 extract. Using occupation-based proxy:")
print("    OCC2010 < 3600 → 'College-type' occupations (mgmt, professional, technical)")
print("    OCC2010 >= 3600 → 'Non-college-type' occupations (service, production, etc.)")

occ_task = pd.read_csv(os.path.join(DATA_DIR, "occ2010_task_scores.csv"))
occ_task['OCC2010'] = occ_task['OCC2010'].astype(int)

CPS_DAT = os.path.join(DATA_DIR, "cps_00002.dat")
CHUNK = 1_000_000

colspecs_edu = [
    (0, 4),      # YEAR
    (9, 11),     # MONTH
    (48, 50),    # STATEFIP
    (52, 66),    # WTFINL
    (124, 126),  # EMPSTAT
    (131, 135),  # OCC2010
]
colnames_edu = ['YEAR', 'MONTH', 'STATEFIP', 'WTFINL', 'EMPSTAT', 'OCC2010']

accum_college = []
accum_noncollege = []

reader = pd.read_fwf(CPS_DAT, colspecs=colspecs_edu, names=colnames_edu,
                      chunksize=CHUNK, dtype=str)

total_read = 0
for i, chunk in enumerate(reader):
    total_read += len(chunk)
    
    for col in ['YEAR', 'MONTH', 'STATEFIP', 'EMPSTAT', 'OCC2010']:
        chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
    chunk['WTFINL'] = pd.to_numeric(chunk['WTFINL'], errors='coerce') / 10000
    
    mask = (
        chunk['EMPSTAT'].isin([10, 12]) &
        (chunk['OCC2010'] > 0) & (chunk['OCC2010'] < 9800) &
        (chunk['STATEFIP'] >= 1) & (chunk['STATEFIP'] <= 56) &
        (chunk['WTFINL'] > 0)
    )
    kept = chunk[mask].copy()
    kept['OCC2010'] = kept['OCC2010'].astype(int)
    kept = kept.merge(occ_task, on='OCC2010', how='left')
    kept = kept[kept['routine'].notna()]
    
    # Split by occupation type
    kept['is_college_occ'] = kept['OCC2010'] < 3600
    
    for is_college, acc in [(True, accum_college), (False, accum_noncollege)]:
        sub = kept[kept['is_college_occ'] == is_college]
        if len(sub) == 0: continue
        agg = sub.groupby(['STATEFIP', 'YEAR']).apply(
            lambda g: pd.Series({
                'mean_routine': np.average(g['routine'], weights=g['WTFINL']),
                'mean_abstract': np.average(g['abstract'], weights=g['WTFINL']),
                'mean_manual': np.average(g['manual'], weights=g['WTFINL']),
                'total_emp': g['WTFINL'].sum(),
                'n_obs': len(g),
            })
        ).reset_index()
        acc.append(agg)
    
    if (i + 1) % 20 == 0:
        print(f"    Chunk {i+1}: {total_read:,} read")

print(f"  Total read: {total_read:,}")

def combine_chunks(acc):
    if not acc: return pd.DataFrame()
    combined = pd.concat(acc, ignore_index=True)
    panel = combined.groupby(['STATEFIP', 'YEAR']).apply(
        lambda g: pd.Series({
            'mean_routine': np.average(g['mean_routine'], weights=g['total_emp']),
            'mean_abstract': np.average(g['mean_abstract'], weights=g['total_emp']),
            'mean_manual': np.average(g['mean_manual'], weights=g['total_emp']),
            'total_emp': g['total_emp'].sum(),
        })
    ).reset_index()
    panel['STATEFIP'] = panel['STATEFIP'].astype(int)
    panel['YEAR'] = panel['YEAR'].astype(int)
    return panel

college_panel = combine_chunks(accum_college)
noncollege_panel = combine_chunks(accum_noncollege)

print(f"  College-type panel: {len(college_panel):,} state-year obs")
print(f"  Non-college-type panel: {len(noncollege_panel):,} state-year obs")

# Save
college_panel.to_csv(os.path.join(ROB_DIR, 'task_college_occ.csv'), index=False)
noncollege_panel.to_csv(os.path.join(ROB_DIR, 'task_noncollege_occ.csv'), index=False)

# Run LP-IV for each group
educ_results = {}

for panel, label, prefix in [(college_panel, 'College-Type Occupations', 'col'),
                               (noncollege_panel, 'Non-College-Type Occupations', 'ncol')]:
    print(f"\n  --- {label} ---")
    if len(panel) == 0:
        print(f"    No data — skipping")
        continue
    
    # Rename
    rn = panel.rename(columns={'mean_routine': f'{prefix}_routine',
                                'mean_abstract': f'{prefix}_abstract'})
    
    # Merge with unemployment, Bartik, tech control
    mg = rn.merge(laus_a, on=['STATEFIP', 'YEAR'], how='inner')
    mg = mg.merge(bartik, on=['STATEFIP', 'YEAR'], how='inner')
    
    # Get initial routine share from main panel
    mg = mg.merge(df[['STATEFIP', 'YEAR', 'rs_init', 'time_trend', 'tc_routine']].drop_duplicates(),
                  on=['STATEFIP', 'YEAR'], how='left')
    
    mg = mg.sort_values(['STATEFIP', 'YEAR']).reset_index(drop=True)
    mg['delta_urate'] = mg.groupby('STATEFIP')['urate'].diff()
    
    # LP outcomes
    rout_col = f'{prefix}_routine'
    mg[f'{rout_col}_L1'] = mg.groupby('STATEFIP')[rout_col].shift(1)
    for h in range(MAX_H + 1):
        mg[f'{rout_col}_F{h}'] = mg.groupby('STATEFIP')[rout_col].shift(-h)
        mg[f'{rout_col}_lp{h}'] = mg[f'{rout_col}_F{h}'] - mg[f'{rout_col}_L1']
    
    # First stage
    f_val, pi_val, n_val = first_stage_f(mg, 'bartik_shock', ['tc_routine'])
    print(f"    First stage: F={f_val:.1f}, N={n_val:,}")
    
    if f_val < 5:
        print(f"    Weak instrument — skipping")
        continue
    
    res = run_lp(mg, [rout_col], list(range(MAX_H+1)), ['tc_routine'])
    if rout_col in res:
        educ_results[label] = res[rout_col]
        print(f"    Routine LP-IV:")
        for _, r in res[rout_col].iterrows():
            print(f"      h={int(r['horizon'])}: {r['beta']:+.5f} ({r['se']:.5f}){star(r['pval']):>4s}")


# ============================================================
# 3. CENSUS-REGION × YEAR TRENDS
# ============================================================
print("\n" + "=" * 70)
print("3. CENSUS-REGION × YEAR TRENDS")
print("=" * 70)

df['region'] = df['STATEFIP'].map(FIPS_TO_REGION)
print(f"  Region mapping: {df['region'].notna().sum()}/{len(df)} matched")
print(f"  Region distribution:")
for r in sorted(df['region'].dropna().unique()):
    n = (df['region'] == r).sum()
    print(f"    {r}: {n:,} obs")

# Create region × year dummies
regions = sorted(df['region'].dropna().unique())
years = sorted(df['YEAR'].unique())
omit_region = regions[0]  # omit first region
omit_year = years[0]

region_year_cols = []
for reg in regions:
    if reg == omit_region: continue
    for yr in years:
        if yr == omit_year: continue
        col = f'ry_{reg[:2]}_{yr}'
        df[col] = ((df['region'] == reg) & (df['YEAR'] == yr)).astype(float)
        if df[col].std() > 0:
            region_year_cols.append(col)

print(f"  Region × year controls: {len(region_year_cols)} dummies")

# First stage with region × year trends
exog_region = ['tc_routine'] + region_year_cols
f_val, pi_val, n_val = first_stage_f(df, 'bartik_shock', exog_region)
region_f = f_val
print(f"  First stage with region × year: F={f_val:.1f}, pi={pi_val:.4f}, N={n_val:,}")

if f_val > 5:
    region_results = run_lp(df, ['mean_routine'], list(range(MAX_H+1)), exog_region)
    if 'mean_routine' in region_results:
        print(f"\n  Routine LP-IV with region × year trends:")
        for _, r in region_results['mean_routine'].iterrows():
            print(f"    h={int(r['horizon'])}: {r['beta']:+.5f} ({r['se']:.5f}){star(r['pval']):>4s}")
else:
    region_results = {}
    print(f"  Weak instrument with region × year — skipping LP-IV")


# ============================================================
# 4. PLOTS
# ============================================================
print("\n" + "=" * 70)
print("4. PLOTS")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Panel A: Leave-one-industry-out
ax = axes[0]
if 'mean_routine' in baseline:
    d = baseline['mean_routine']
    ax.plot(d['horizon'], d['beta'], 'o-', color='#b2182b', linewidth=2.5,
            markersize=5, label='Baseline', alpha=0.85)
    ax.errorbar(d['horizon'], d['beta'],
                yerr=[d['beta']-d['ci_lower'], d['ci_upper']-d['beta']],
                fmt='none', ecolor='#b2182b', capsize=4, elinewidth=1.5, zorder=2)

loo_colors = {'Mining and logging': '#4393c3',
              'Construction': '#1b7837',
              'Manufacturing - Durable goods': '#762a83'}
loo_styles = {'Mining and logging': 's--',
              'Construction': '^:',
              'Manufacturing - Durable goods': 'D-.'}

for ind, d in loo_results.items():
    ax.plot(d['horizon'], d['beta'], loo_styles.get(ind, 's--'),
            color=loo_colors.get(ind, 'gray'), linewidth=1.5, markersize=4,
            label=f'Drop {ind}', alpha=0.8)

ax.axhline(0, color='gray', linewidth=0.8)
ax.set_xlabel('Horizon (years)')
ax.set_ylabel(r'$\hat{\beta}_h$')
ax.set_title('A: Leave-One-Industry-Out Bartik', fontweight='bold')
ax.legend(fontsize=7, loc='lower right')
ax.grid(True, alpha=0.2)
ax.set_xticks(range(MAX_H + 1))

# Panel B: Education split
ax = axes[1]
educ_colors = {'College-Type Occupations': '#4393c3',
               'Non-College-Type Occupations': '#b2182b'}
for label, d in educ_results.items():
    ax.plot(d['horizon'], d['beta'], 'o-', color=educ_colors.get(label, 'gray'),
            linewidth=2, markersize=5, label=label, alpha=0.85)
    ax.errorbar(d['horizon'], d['beta'],
                yerr=[d['beta']-d['ci_lower'], d['ci_upper']-d['beta']],
                fmt='none', ecolor=educ_colors.get(label, 'gray'),
                capsize=3, elinewidth=1.2, alpha=0.85, zorder=2)

ax.axhline(0, color='gray', linewidth=0.8)
ax.set_xlabel('Horizon (years)')
ax.set_ylabel(r'$\hat{\beta}_h$')
ax.set_title('B: Occupation-Type Heterogeneity', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)
ax.set_xticks(range(MAX_H + 1))

# Panel C: Region × year trends
ax = axes[2]
if 'mean_routine' in baseline:
    d = baseline['mean_routine']
    ax.plot(d['horizon'], d['beta'], 'o-', color='#b2182b', linewidth=2.5,
            markersize=5, label='Baseline', alpha=0.85)
    ax.errorbar(d['horizon'], d['beta'],
                yerr=[d['beta']-d['ci_lower'], d['ci_upper']-d['beta']],
                fmt='none', ecolor='#b2182b', capsize=4, elinewidth=1.5, zorder=2)

if region_results and 'mean_routine' in region_results:
    d = region_results['mean_routine']
    ax.plot(d['horizon'], d['beta'], 's--', color='#4393c3', linewidth=2,
            markersize=5, label='+ Region × Year', alpha=0.85)
    ax.errorbar(d['horizon'], d['beta'],
                yerr=[d['beta']-d['ci_lower'], d['ci_upper']-d['beta']],
                fmt='none', ecolor='#4393c3', capsize=3, elinewidth=1.2, alpha=0.85, zorder=2)

ax.axhline(0, color='gray', linewidth=0.8)
ax.set_xlabel('Horizon (years)')
ax.set_ylabel(r'$\hat{\beta}_h$')
ax.set_title('C: Census Region × Year Trends', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)
ax.set_xticks(range(MAX_H + 1))

plt.tight_layout()
path = os.path.join(ROB_DIR, 'fig_extensions.pdf')
fig.savefig(path, dpi=300, bbox_inches='tight')
print(f"  Saved: {path}")
plt.close()


# ============================================================
# 5. EXPORT
# ============================================================
print("\n" + "-" * 50)
print("5. Export")
print("-" * 50)

all_rows = []

# LOO results
for ind, d in loo_results.items():
    o = d.copy()
    o['specification'] = f'LOO_drop_{ind}'
    o['outcome'] = 'mean_routine'
    o['first_stage_f'] = loo_fstats.get(ind, np.nan)
    all_rows.append(o)

# Education results
for label, d in educ_results.items():
    o = d.copy()
    o['specification'] = f'educ_{label}'
    o['outcome'] = 'mean_routine'
    all_rows.append(o)

# Region results
if region_results and 'mean_routine' in region_results:
    o = region_results['mean_routine'].copy()
    o['specification'] = 'region_year_trends'
    o['outcome'] = 'mean_routine'
    o['first_stage_f'] = region_f
    all_rows.append(o)

if all_rows:
    out_df = pd.concat(all_rows, ignore_index=True)
    p = os.path.join(ROB_DIR, 'extensions_results.csv')
    out_df.to_csv(p, index=False, float_format='%.6f')
    print(f"  Saved: {p}")

print("\n" + "=" * 70)
print("EXTENSIONS COMPLETE")
print("=" * 70)

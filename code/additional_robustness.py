"""
Additional Robustness Checks
==============================
1. Multiple trend controls (routine + abstract + manual share × trend)
2. Fixed base-year (1977) Bartik vs rolling shares
3. Industry decomposition: Bartik-contributing vs non-Bartik industries

Reads same data as lp_iv_annual.py
Output: results/robustness/

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

OUTCOMES = ['mean_rti', 'mean_abstract', 'mean_routine', 'mean_manual']
OLABELS = {'mean_rti': 'RTI', 'mean_abstract': 'Abstract',
           'mean_routine': 'Routine', 'mean_manual': 'Manual'}
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

def first_stage_f(df_in, exog_cols):
    """Compute first-stage F for Bartik -> delta_urate."""
    df_p = df_in.set_index(['STATEFIP', 'YEAR'])
    y = df_p['delta_urate']
    x = df_p[['bartik_shock'] + exog_cols]
    valid = y.notna() & x.notna().all(axis=1)
    if valid.sum() < 100:
        return 0, 0, 0
    try:
        mod = PanelOLS(y[valid], x[valid], entity_effects=True, time_effects=True,
                       drop_absorbed=True, check_rank=False)
        res = mod.fit(cov_type='clustered', cluster_entity=True)
        if 'bartik_shock' in res.params.index:
            pi = res.params['bartik_shock']
            se = res.std_errors['bartik_shock']
            f = (pi / se) ** 2
            return f, pi, int(valid.sum())
    except:
        pass
    return 0, 0, 0

def run_lp(df_in, outcomes, horizons, exog_cols):
    """Run LP-IV for given outcomes and horizons."""
    results = {}
    for out in outcomes:
        rows = []
        for h in horizons:
            lp_var = f'{out}_lp{h}'
            if lp_var not in df_in.columns:
                continue
            cols = [lp_var, 'delta_urate', 'bartik_shock'] + exog_cols
            valid = df_in[cols].notna().all(axis=1)
            if valid.sum() < 100:
                continue
            try:
                b, se, n = run_2sls(
                    df_in.loc[valid, lp_var].values,
                    df_in.loc[valid, 'delta_urate'].values,
                    df_in.loc[valid, 'bartik_shock'].values,
                    [df_in.loc[valid, c].values for c in exog_cols],
                    df_in.loc[valid, 'STATEFIP'].values,
                    df_in.loc[valid, 'YEAR'].values
                )
                t = b / se if se > 0 else 0
                p = 2 * (1 - norm.cdf(abs(t)))
                rows.append({'horizon': h, 'beta': b, 'se': se, 'pval': p, 'n': n,
                             'ci_lower': b - 1.96*se, 'ci_upper': b + 1.96*se})
            except:
                pass
        if rows:
            results[out] = pd.DataFrame(rows)
    return results

def star(p):
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.1: return '*'
    return ''

def wmean(g, v, w='total_employed'):
    d = g[[v, w]].dropna()
    if len(d) == 0 or d[w].sum() == 0:
        return np.nan
    return np.average(d[v], weights=d[w])


# ============================================================
# LOAD DATA
# ============================================================
print("=" * 70)
print("ADDITIONAL ROBUSTNESS CHECKS")
print("=" * 70)

sm = pd.read_csv(os.path.join(DATA_DIR, "state_month_panel.csv"))
all_vars = OUTCOMES + ['share_routine', 'share_NRC', 'share_NRM']
sa = sm.groupby(['STATEFIP', 'YEAR']).apply(
    lambda g: pd.Series({v: wmean(g, v) for v in all_vars if v in g.columns})
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

print(f"Panel: {len(df):,} obs, {df['STATEFIP'].nunique()} states, "
      f"{df['YEAR'].min()}-{df['YEAR'].max()}")

# Base year for initial shares
base_yr = df['YEAR'].min() + 1

# LP outcomes
for v in OUTCOMES:
    df[f'{v}_L1'] = df.groupby('STATEFIP')[v].shift(1)
    for h in range(MAX_H + 1):
        df[f'{v}_F{h}'] = df.groupby('STATEFIP')[v].shift(-h)
        df[f'{v}_lp{h}'] = df[f'{v}_F{h}'] - df[f'{v}_L1']


# ============================================================
# 1. MULTIPLE TREND CONTROLS
# ============================================================
print("\n" + "=" * 70)
print("1. MULTIPLE TREND CONTROLS")
print("=" * 70)

# Compute initial shares for each task dimension
for share_var, label in [('share_routine', 'routine'), ('mean_abstract', 'abstract'),
                          ('mean_manual', 'manual'), ('mean_routine', 'routine_task')]:
    if share_var in df.columns:
        base_vals = df[df['YEAR'] == base_yr].groupby('STATEFIP')[share_var].mean()
        df = df.merge(base_vals.rename(f'init_{label}'), on='STATEFIP', how='left')

df['time_trend'] = df['YEAR'] - df['YEAR'].min()

# Specification A: Routine share only (baseline)
df['tc_routine'] = df['init_routine'] * df['time_trend']
exog_A = ['tc_routine']

# Specification B: Routine + Abstract + Manual
df['tc_abstract'] = df['init_abstract'] * df['time_trend']
df['tc_manual'] = df['init_manual'] * df['time_trend']
exog_B = ['tc_routine', 'tc_abstract', 'tc_manual']

# Specification C: All three task-level trends
df['tc_routine_task'] = df['init_routine_task'] * df['time_trend']
exog_C = ['tc_routine', 'tc_abstract', 'tc_manual', 'tc_routine_task']
# Check for multicollinearity — drop if correlation too high
exog_C_valid = []
for c in exog_C:
    if c in df.columns and df[c].notna().any() and df[c].std() > 0:
        exog_C_valid.append(c)
exog_C = exog_C_valid

specs = [
    ('A: Routine trend only', exog_A),
    ('B: Routine + Abstract + Manual trends', exog_B),
]
if len(exog_C) > len(exog_B):
    specs.append(('C: All task trends', exog_C))

for spec_name, exog_cols in specs:
    # Check valid columns
    exog_valid = [c for c in exog_cols if c in df.columns and df[c].std() > 0]
    
    f_val, pi_val, n_val = first_stage_f(df, exog_valid)
    print(f"\n  {spec_name}:")
    print(f"    First stage: F={f_val:.1f}, pi={pi_val:.4f}, N={n_val:,}")
    
    if f_val > 5:
        results = run_lp(df, ['mean_routine'], list(range(MAX_H + 1)), exog_valid)
        if 'mean_routine' in results:
            print(f"    Routine LP-IV:")
            for _, r in results['mean_routine'].iterrows():
                print(f"      h={int(r['horizon'])}: {r['beta']:+.5f} ({r['se']:.5f}){star(r['pval']):>4s}")
    else:
        print(f"    First stage too weak — skipping LP-IV")

# Correlation matrix of trend controls
print(f"\n  Correlation between trend controls:")
trend_cols = [c for c in ['tc_routine', 'tc_abstract', 'tc_manual'] if c in df.columns]
if trend_cols:
    corr = df[trend_cols].corr()
    for i, c1 in enumerate(trend_cols):
        for c2 in trend_cols[i+1:]:
            print(f"    {c1} × {c2}: r = {corr.loc[c1, c2]:.3f}")


# ============================================================
# 2. FIXED BASE-YEAR BARTIK
# ============================================================
print("\n" + "=" * 70)
print("2. FIXED BASE-YEAR BARTIK (1977 shares)")
print("=" * 70)

# Load industry shares
shares = pd.read_csv(os.path.join(DATA_DIR, "state_industry_shares.csv"))
ces = pd.read_csv(os.path.join(DATA_DIR, "ces_national_industry.csv"))

# Fixed base year shares (1977 or earliest available)
base_shares = shares[shares['YEAR'] == base_yr][['STATEFIP', 'industry', 'emp_share']].copy()
print(f"  Base year: {base_yr}")
print(f"  States with base shares: {base_shares['STATEFIP'].nunique()}")
print(f"  Industries: {base_shares['industry'].nunique()}")

# National growth rates (annual)
ces_a = ces.groupby(['industry', 'year'])['employment'].mean().reset_index()
ces_a = ces_a.sort_values(['industry', 'year'])
ces_a['emp_lag'] = ces_a.groupby('industry')['employment'].shift(1)
ces_a['growth'] = (ces_a['employment'] - ces_a['emp_lag']) / ces_a['emp_lag']
ces_a = ces_a.dropna(subset=['growth'])

# Build fixed Bartik
bartik_fixed_rows = []
for year in sorted(df['YEAR'].unique()):
    growth = ces_a[ces_a['year'] == year][['industry', 'growth']]
    if len(growth) == 0:
        continue
    merged = base_shares.merge(growth, on='industry', how='inner')
    b = merged.groupby('STATEFIP').apply(
        lambda g: (g['emp_share'] * g['growth']).sum()
    ).reset_index()
    b.columns = ['STATEFIP', 'bartik_fixed']
    b['YEAR'] = year
    bartik_fixed_rows.append(b)

if bartik_fixed_rows:
    bartik_fixed = pd.concat(bartik_fixed_rows, ignore_index=True)
    df = df.merge(bartik_fixed, on=['STATEFIP', 'YEAR'], how='left')
    
    print(f"  Fixed Bartik obs: {df['bartik_fixed'].notna().sum():,}")
    print(f"  Correlation with rolling Bartik: "
          f"{df[['bartik_shock', 'bartik_fixed']].dropna().corr().iloc[0,1]:.3f}")
    
    # First stage with fixed Bartik
    df_p = df.set_index(['STATEFIP', 'YEAR'])
    y_fs = df_p['delta_urate']
    x_fs = df_p[['bartik_fixed', 'tc_routine']].dropna()
    valid_fs = y_fs.notna() & df_p[['bartik_fixed', 'tc_routine']].notna().all(axis=1)
    
    try:
        mod = PanelOLS(y_fs[valid_fs], df_p.loc[valid_fs, ['bartik_fixed', 'tc_routine']],
                       entity_effects=True, time_effects=True,
                       drop_absorbed=True, check_rank=False)
        res = mod.fit(cov_type='clustered', cluster_entity=True)
        pi_fixed = res.params['bartik_fixed']
        se_fixed = res.std_errors['bartik_fixed']
        f_fixed = (pi_fixed / se_fixed) ** 2
        print(f"  First stage (fixed): F={f_fixed:.1f}, pi={pi_fixed:.4f}")
    except Exception as e:
        print(f"  First stage failed: {e}")
        f_fixed = 0
    
    # LP-IV with fixed Bartik
    if f_fixed > 5:
        # Temporarily swap instruments
        df_backup = df['bartik_shock'].copy()
        df['bartik_shock'] = df['bartik_fixed']
        
        fixed_results = run_lp(df, ['mean_routine'], list(range(MAX_H + 1)), ['tc_routine'])
        
        df['bartik_shock'] = df_backup  # restore
        
        if 'mean_routine' in fixed_results:
            print(f"\n  Routine LP-IV (fixed base-year Bartik):")
            for _, r in fixed_results['mean_routine'].iterrows():
                print(f"    h={int(r['horizon'])}: {r['beta']:+.5f} ({r['se']:.5f}){star(r['pval']):>4s}")
    else:
        fixed_results = {}
        print(f"  Fixed Bartik first stage too weak — skipping")
else:
    fixed_results = {}
    print("  Could not construct fixed Bartik")


# ============================================================
# 3. BARTIK vs NON-BARTIK INDUSTRY DECOMPOSITION
# ============================================================
print("\n" + "=" * 70)
print("3. INDUSTRY DECOMPOSITION (Exclusion Restriction Test)")
print("=" * 70)

# Identify high-variance (Bartik-driving) vs low-variance industries
growth_var = ces_a.groupby('industry')['growth'].var().reset_index()
growth_var.columns = ['industry', 'var_growth']
growth_var = growth_var.sort_values('var_growth', ascending=False)

median_var = growth_var['var_growth'].median()
high_var_industries = set(growth_var[growth_var['var_growth'] >= median_var]['industry'])
low_var_industries = set(growth_var[growth_var['var_growth'] < median_var]['industry'])

print(f"  High-variance (Bartik-driving) industries ({len(high_var_industries)}):")
for ind in sorted(high_var_industries):
    v = growth_var[growth_var['industry'] == ind]['var_growth'].values[0]
    print(f"    {ind}: var(growth) = {v:.5f}")

print(f"\n  Low-variance (non-Bartik) industries ({len(low_var_industries)}):")
for ind in sorted(low_var_industries):
    v = growth_var[growth_var['industry'] == ind]['var_growth'].values[0]
    print(f"    {ind}: var(growth) = {v:.5f}")

# Need to go back to CPS microdata to compute task composition separately
# for Bartik vs non-Bartik industries. We have state_industry_shares.csv
# but not task scores by industry. We need the IND1990 mapping.

# Check if we can compute this from existing data
ind_shares_path = os.path.join(DATA_DIR, "state_industry_shares.csv")
if os.path.exists(ind_shares_path):
    ind_sh = pd.read_csv(ind_shares_path)
    
    # We can at least check: do Bartik-contributing industries have different
    # employment dynamics than non-Bartik industries?
    ind_sh['is_bartik'] = ind_sh['industry'].isin(high_var_industries)
    
    # Aggregate employment share of Bartik vs non-Bartik industries by state-year
    bartik_ind_share = ind_sh.groupby(['STATEFIP', 'YEAR', 'is_bartik'])['emp_share'].sum().reset_index()
    bartik_ind_share = bartik_ind_share.pivot_table(
        index=['STATEFIP', 'YEAR'], columns='is_bartik', values='emp_share'
    ).reset_index()
    bartik_ind_share.columns = ['STATEFIP', 'YEAR', 'share_nonbartik_ind', 'share_bartik_ind']
    
    df = df.merge(bartik_ind_share, on=['STATEFIP', 'YEAR'], how='left')
    
    # LP-IV with Bartik industry share and non-Bartik industry share as outcomes
    for share_col, label in [('share_bartik_ind', 'Bartik Industries'),
                              ('share_nonbartik_ind', 'Non-Bartik Industries')]:
        if share_col not in df.columns:
            continue
        
        df[f'{share_col}_L1'] = df.groupby('STATEFIP')[share_col].shift(1)
        for h in range(MAX_H + 1):
            df[f'{share_col}_F{h}'] = df.groupby('STATEFIP')[share_col].shift(-h)
            df[f'{share_col}_lp{h}'] = df[f'{share_col}_F{h}'] - df[f'{share_col}_L1']
    
    ind_results = run_lp(df, ['share_bartik_ind', 'share_nonbartik_ind'],
                          list(range(MAX_H + 1)), ['tc_routine'])
    
    print(f"\n  LP-IV: Employment share response by industry type")
    for out_key, label in [('share_bartik_ind', 'Bartik Industries'),
                            ('share_nonbartik_ind', 'Non-Bartik Industries')]:
        if out_key in ind_results:
            print(f"\n    {label}:")
            for _, r in ind_results[out_key].iterrows():
                print(f"      h={int(r['horizon'])}: {r['beta']:+.5f} ({r['se']:.5f}){star(r['pval']):>4s}")
        else:
            print(f"\n    {label}: no results")
    
    print(f"\n  Interpretation:")
    print(f"    If non-Bartik industries also show employment shifts,")
    print(f"    the demand shock propagates broadly (supports exclusion restriction).")
    print(f"    If only Bartik industries respond, the effect may be mechanical.")


# ============================================================
# 4. COMPARISON PLOT
# ============================================================
print("\n" + "=" * 70)
print("4. COMPARISON PLOTS")
print("=" * 70)

# Run baseline for comparison
baseline = run_lp(df, ['mean_routine'], list(range(MAX_H + 1)), ['tc_routine'])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel A: Multiple trend controls
ax = axes[0]
specs_plot = [('Routine only', exog_A), ('Routine+Abstract+Manual', exog_B)]
colors = ['#b2182b', '#4393c3', '#1b7837']
for i, (name, exog) in enumerate(specs_plot):
    exog_valid = [c for c in exog if c in df.columns and df[c].std() > 0]
    res = run_lp(df, ['mean_routine'], list(range(MAX_H + 1)), exog_valid)
    if 'mean_routine' in res:
        d = res['mean_routine']
        ax.plot(d['horizon'], d['beta'], 'o-', color=colors[i], linewidth=2,
                markersize=5, label=name, alpha=0.85)
        ax.errorbar(d['horizon'], d['beta'],
                    yerr=[d['beta']-d['ci_lower'], d['ci_upper']-d['beta']],
                    fmt='none', ecolor=colors[i], capsize=3, elinewidth=1.2,
                    alpha=0.85, zorder=2)
ax.axhline(0, color='gray', linewidth=0.8)
ax.set_xlabel('Horizon (years)')
ax.set_ylabel(r'$\hat{\beta}_h$')
ax.set_title('A: Multiple Trend Controls', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)
ax.set_xticks(range(MAX_H + 1))

# Panel B: Rolling vs Fixed Bartik
ax = axes[1]
if 'mean_routine' in baseline:
    d = baseline['mean_routine']
    ax.plot(d['horizon'], d['beta'], 'o-', color='#b2182b', linewidth=2,
            markersize=5, label='Rolling shares', alpha=0.85)
    ax.errorbar(d['horizon'], d['beta'],
                yerr=[d['beta']-d['ci_lower'], d['ci_upper']-d['beta']],
                fmt='none', ecolor='#b2182b', capsize=3, elinewidth=1.2,
                alpha=0.85, zorder=2)
if fixed_results and 'mean_routine' in fixed_results:
    d = fixed_results['mean_routine']
    ax.plot(d['horizon'], d['beta'], 's--', color='#4393c3', linewidth=2,
            markersize=5, label='Fixed 1977 shares', alpha=0.85)
    ax.errorbar(d['horizon'], d['beta'],
                yerr=[d['beta']-d['ci_lower'], d['ci_upper']-d['beta']],
                fmt='none', ecolor='#4393c3', capsize=3, elinewidth=1.2,
                alpha=0.85, zorder=2)
ax.axhline(0, color='gray', linewidth=0.8)
ax.set_xlabel('Horizon (years)')
ax.set_ylabel(r'$\hat{\beta}_h$')
ax.set_title('B: Rolling vs Fixed Bartik', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)
ax.set_xticks(range(MAX_H + 1))

# Panel C: Industry decomposition
ax = axes[2]
ind_colors = {'share_bartik_ind': '#b2182b', 'share_nonbartik_ind': '#4393c3'}
ind_labels = {'share_bartik_ind': 'Bartik industries', 'share_nonbartik_ind': 'Non-Bartik industries'}
for out_key in ['share_bartik_ind', 'share_nonbartik_ind']:
    if out_key in ind_results:
        d = ind_results[out_key]
        ax.plot(d['horizon'], d['beta'], 'o-', color=ind_colors[out_key],
                linewidth=2, markersize=5, label=ind_labels[out_key], alpha=0.85)
        ax.errorbar(d['horizon'], d['beta'],
                    yerr=[d['beta']-d['ci_lower'], d['ci_upper']-d['beta']],
                    fmt='none', ecolor=ind_colors[out_key],
                    capsize=3, elinewidth=1.2, alpha=0.85, zorder=2)
ax.axhline(0, color='gray', linewidth=0.8)
ax.set_xlabel('Horizon (years)')
ax.set_ylabel(r'$\hat{\beta}_h$')
ax.set_title('C: Industry Decomposition', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)
ax.set_xticks(range(MAX_H + 1))

plt.tight_layout()
path = os.path.join(ROB_DIR, 'fig_additional_robustness.pdf')
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

# Multiple trends
for spec_name, exog in specs:
    exog_valid = [c for c in exog if c in df.columns and df[c].std() > 0]
    res = run_lp(df, ['mean_routine'], list(range(MAX_H + 1)), exog_valid)
    if 'mean_routine' in res:
        o = res['mean_routine'].copy()
        o['specification'] = spec_name
        o['outcome'] = 'mean_routine'
        all_rows.append(o)

# Fixed Bartik
if fixed_results and 'mean_routine' in fixed_results:
    o = fixed_results['mean_routine'].copy()
    o['specification'] = 'Fixed 1977 Bartik'
    o['outcome'] = 'mean_routine'
    o['first_stage_f'] = f_fixed
    all_rows.append(o)

# Industry decomposition
for out_key in ['share_bartik_ind', 'share_nonbartik_ind']:
    if out_key in ind_results:
        o = ind_results[out_key].copy()
        o['specification'] = 'Industry decomposition'
        o['outcome'] = out_key
        all_rows.append(o)

if all_rows:
    out_df = pd.concat(all_rows, ignore_index=True)
    p = os.path.join(ROB_DIR, 'additional_robustness_results.csv')
    out_df.to_csv(p, index=False, float_format='%.6f')
    print(f"  Saved: {p}")


print("\n" + "=" * 70)
print("ADDITIONAL ROBUSTNESS COMPLETE")
print("=" * 70)

"""
Annual LP-IV
=============
Same specification as quarterly LP-IV but at annual frequency
where the Bartik instrument should have stronger first stage.

T_{s,t+h} - T_{s,t-1} = β_h · ΔÛ_st + δ_h · TechControl + α_s + θ_t + ε

First stage: ΔU_st = π · Bartik_st + α_s + θ_t + v_st

Reads:
  state_month_panel.csv → aggregated to annual
  laus_state_unemployment.csv → annual
  bartik_instrument.csv → already annual

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

from paths import DATA_DIR, RESULTS_DIR
os.makedirs(RESULTS_DIR, exist_ok=True)

MAX_H = 5  # years ahead
OUTCOMES = ['mean_rti', 'mean_abstract', 'mean_routine', 'mean_manual']
OLABELS = {'mean_rti': 'RTI', 'mean_abstract': 'Abstract',
           'mean_routine': 'Routine', 'mean_manual': 'Manual'}

# ============================================================
# STEP 1: Load and Build Annual Panel
# ============================================================
print("=" * 70)
print("ANNUAL LP-IV")
print("=" * 70)

# Task composition — aggregate monthly to annual
sm = pd.read_csv(os.path.join(DATA_DIR, "state_month_panel.csv"))

def wmean(g, v, w='total_employed'):
    d = g[[v, w]].dropna()
    if len(d) == 0 or d[w].sum() == 0:
        return np.nan
    return np.average(d[v], weights=d[w])

sa = sm.groupby(['STATEFIP', 'YEAR']).apply(
    lambda g: pd.Series({
        **{v: wmean(g, v) for v in OUTCOMES},
        'total_employed': g['total_employed'].sum(),
        'n_obs': g['n_obs'].sum(),
        'share_routine': wmean(g, 'share_routine') if 'share_routine' in g.columns else np.nan,
    })
).reset_index()
sa['STATEFIP'] = sa['STATEFIP'].astype(int)
sa['YEAR'] = sa['YEAR'].astype(int)
print(f"Task composition: {len(sa):,} state-year obs")

# Unemployment — annual average
laus = pd.read_csv(os.path.join(DATA_DIR, "laus_state_unemployment.csv"))
laus_a = laus.groupby(['STATEFIP', 'year']).agg(
    urate=('unemployment_rate', 'mean'),
).reset_index().rename(columns={'year': 'YEAR'})
laus_a['STATEFIP'] = laus_a['STATEFIP'].astype(int)
print(f"Unemployment: {len(laus_a):,} state-year obs")

# Bartik — already annual
bartik = pd.read_csv(os.path.join(DATA_DIR, "bartik_instrument.csv"))
bartik['STATEFIP'] = bartik['STATEFIP'].astype(int)
bartik.rename(columns={c: c for c in bartik.columns}, inplace=True)
print(f"Bartik: {len(bartik):,} state-year obs")

# Merge
df = sa.merge(laus_a, on=['STATEFIP', 'YEAR'], how='inner')
df = df.merge(bartik, on=['STATEFIP', 'YEAR'], how='inner')
df = df.sort_values(['STATEFIP', 'YEAR']).reset_index(drop=True)
print(f"\nMerged: {len(df):,} obs, {df['STATEFIP'].nunique()} states, "
      f"{df['YEAR'].min()}-{df['YEAR'].max()}")

# ΔU
df['delta_urate'] = df.groupby('STATEFIP')['urate'].diff()

# Tech control
base_yr = df['YEAR'].min() + 1
base = df[df['YEAR'] == base_yr].groupby('STATEFIP')['share_routine'].mean()
df = df.merge(base.rename('rs_init'), on='STATEFIP', how='left')
df['time_trend'] = df['YEAR'] - df['YEAR'].min()
df['tech_control'] = df['rs_init'] * df['time_trend']
has_tech = df['tech_control'].notna().all() and df['tech_control'].std() > 0
print(f"Tech control: {'yes' if has_tech else 'no'}")

# LP dependent variables
for out in OUTCOMES:
    df[f'{out}_L1'] = df.groupby('STATEFIP')[out].shift(1)
    for h in range(MAX_H + 1):
        df[f'{out}_F{h}'] = df.groupby('STATEFIP')[out].shift(-h)
        df[f'{out}_lp{h}'] = df[f'{out}_F{h}'] - df[f'{out}_L1']

print(f"LP outcomes: h = 0,...,{MAX_H}")
print(f"Valid at h=0: {df[f'{OUTCOMES[0]}_lp0'].notna().sum():,}")
print(f"Valid at h={MAX_H}: {df[f'{OUTCOMES[0]}_lp{MAX_H}'].notna().sum():,}")


# ============================================================
# STEP 2: First Stage
# ============================================================
print("\n" + "-" * 50)
print("STEP 2: First Stage")
print("-" * 50)

df_p = df.set_index(['STATEFIP', 'YEAR'])

fs_exog = ['bartik_shock']
if has_tech:
    fs_exog.append('tech_control')

y_fs = df_p['delta_urate']
x_fs = df_p[fs_exog]
valid_fs = y_fs.notna() & x_fs.notna().all(axis=1)

try:
    fs_mod = PanelOLS(y_fs[valid_fs], x_fs[valid_fs],
                      entity_effects=True, time_effects=True,
                      drop_absorbed=True, check_rank=False)
    fs_res = fs_mod.fit(cov_type='clustered', cluster_entity=True)
    
    pi = fs_res.params['bartik_shock']
    se_pi = fs_res.std_errors['bartik_shock']
    f_stat = (pi / se_pi) ** 2
    
    print(f"  pi = {pi:.4f} ({se_pi:.4f})")
    print(f"  F = {f_stat:.1f}")
    print(f"  N = {valid_fs.sum():,}")
    print(f"  Within-R2 = {fs_res.rsquared_within:.4f}")
    
    if f_stat >= 10:
        print("  STRONG instrument")
    else:
        print("  WEAK instrument (F < 10)")
except Exception as e:
    print(f"  Failed: {e}")
    f_stat = 0
    pi = 0


# ============================================================
# Helper: Double-demean + 2SLS + Clustered SE
# ============================================================
def dd(x, s, t):
    """Double-demean."""
    d = pd.DataFrame({'x': x, 's': s, 't': t})
    gm = d['x'].mean()
    return (d['x'] - d.groupby('s')['x'].transform('mean')
            - d.groupby('t')['x'].transform('mean') + gm).values

def run_2sls(y, endog, instr, exog_list, states, times):
    """Manual 2SLS with clustered SE."""
    y_dd = dd(y, states, times)
    end_dd = dd(endog, states, times)
    ins_dd = dd(instr, states, times)
    
    exog_dd = [dd(x, states, times) for x in exog_list]
    
    # Stage 1
    Z = np.column_stack([ins_dd] + [e.reshape(-1,1) for e in exog_dd]) if exog_dd else ins_dd.reshape(-1,1)
    b1, _, _, _ = np.linalg.lstsq(Z, end_dd, rcond=None)
    end_hat = Z @ b1
    
    # Stage 2
    X2 = np.column_stack([end_hat] + [e.reshape(-1,1) for e in exog_dd]) if exog_dd else end_hat.reshape(-1,1)
    b2, _, _, _ = np.linalg.lstsq(X2, y_dd, rcond=None)
    
    # Residuals with actual endog
    X2a = np.column_stack([end_dd] + [e.reshape(-1,1) for e in exog_dd]) if exog_dd else end_dd.reshape(-1,1)
    resid = y_dd - X2a @ b2
    
    # Clustered SE
    n = len(y_dd)
    k = X2.shape[1]
    us = np.unique(states)
    G = len(us)
    XtXi = np.linalg.pinv(X2.T @ X2)
    B = np.zeros((k, k))
    for sv in us:
        m = states == sv
        Xe = X2[m].T @ resid[m]
        B += np.outer(Xe, Xe)
    cor = (G / (G - 1)) * ((n - 1) / (n - k))
    V = cor * XtXi @ B @ XtXi
    se = np.sqrt(np.maximum(np.diag(V), 0))
    
    return b2[0], se[0], n

def run_ols(y, x, exog_list, states, times):
    """OLS with clustered SE."""
    y_dd = dd(y, states, times)
    x_dd = dd(x, states, times)
    exog_dd = [dd(e, states, times) for e in exog_list]
    
    X = np.column_stack([x_dd] + [e.reshape(-1,1) for e in exog_dd]) if exog_dd else x_dd.reshape(-1,1)
    b, _, _, _ = np.linalg.lstsq(X, y_dd, rcond=None)
    resid = y_dd - X @ b
    
    n = len(y_dd)
    k = X.shape[1]
    us = np.unique(states)
    G = len(us)
    XtXi = np.linalg.pinv(X.T @ X)
    B = np.zeros((k, k))
    for sv in us:
        m = states == sv
        Xe = X[m].T @ resid[m]
        B += np.outer(Xe, Xe)
    cor = (G / (G - 1)) * ((n - 1) / (n - k))
    V = cor * XtXi @ B @ XtXi
    se = np.sqrt(np.maximum(np.diag(V), 0))
    
    return b[0], se[0], n


# ============================================================
# STEP 3: LP-IV
# ============================================================
print("\n" + "-" * 50)
print("STEP 3: LP-IV (Annual)")
print("-" * 50)

iv_results = {}
ols_results = {}

for out in OUTCOMES:
    print(f"\n  --- {OLABELS[out]} ---")
    print(f"  {'h':>3s}  {'IV beta':>10s} {'IV se':>8s} {'':>4s}  {'OLS beta':>10s} {'OLS se':>8s}")
    
    iv_rows = []
    ols_rows = []
    
    for h in range(MAX_H + 1):
        lp_var = f'{out}_lp{h}'
        
        # Get valid mask
        cols_needed = [lp_var, 'delta_urate', 'bartik_shock']
        if has_tech:
            cols_needed.append('tech_control')
        
        valid = df[cols_needed].notna().all(axis=1)
        if valid.sum() < 100:
            continue
        
        y = df.loc[valid, lp_var].values
        endog = df.loc[valid, 'delta_urate'].values
        instr = df.loc[valid, 'bartik_shock'].values
        states = df.loc[valid, 'STATEFIP'].values
        times = df.loc[valid, 'YEAR'].values
        
        exog_list = []
        if has_tech:
            exog_list.append(df.loc[valid, 'tech_control'].values)
        
        # IV
        try:
            b_iv, se_iv, n_iv = run_2sls(y, endog, instr, exog_list, states, times)
            t_iv = b_iv / se_iv if se_iv > 0 else 0
            p_iv = 2 * (1 - norm.cdf(abs(t_iv)))
            iv_rows.append({
                'horizon': h, 'beta': b_iv, 'se': se_iv, 'pval': p_iv,
                'ci_lower': b_iv - 1.96 * se_iv,
                'ci_upper': b_iv + 1.96 * se_iv, 'n': n_iv,
            })
        except Exception as e:
            b_iv, se_iv, p_iv = np.nan, np.nan, np.nan
            print(f"  {h:3d}  IV failed: {e}")
        
        # OLS
        try:
            b_ols, se_ols, n_ols = run_ols(y, endog, exog_list, states, times)
            t_ols = b_ols / se_ols if se_ols > 0 else 0
            p_ols = 2 * (1 - norm.cdf(abs(t_ols)))
            ols_rows.append({
                'horizon': h, 'beta': b_ols, 'se': se_ols, 'pval': p_ols,
                'ci_lower': b_ols - 1.96 * se_ols,
                'ci_upper': b_ols + 1.96 * se_ols, 'n': n_ols,
            })
        except Exception as e:
            b_ols, se_ols, p_ols = np.nan, np.nan, np.nan
        
        # Print
        def star(p):
            if p < 0.01: return '***'
            if p < 0.05: return '**'
            if p < 0.1: return '*'
            return ''
        
        iv_s = f"{b_iv:+.5f} ({se_iv:.5f}){star(p_iv):>4s}" if not np.isnan(b_iv) else "failed"
        ols_s = f"{b_ols:+.5f} ({se_ols:.5f}){star(p_ols):>4s}" if not np.isnan(b_ols) else "failed"
        print(f"  {h:3d}  {iv_s}  {ols_s}")
    
    if iv_rows:
        iv_results[out] = pd.DataFrame(iv_rows)
    if ols_rows:
        ols_results[out] = pd.DataFrame(ols_rows)


# ============================================================
# STEP 4: Plots
# ============================================================
print("\n" + "-" * 50)
print("STEP 4: Plots")
print("-" * 50)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i, out in enumerate(OUTCOMES[:4]):
    ax = axes.flatten()[i]
    
    if out in iv_results:
        d = iv_results[out]
        ax.plot(d['horizon'], d['beta'], 'o-', color='#b2182b',
                linewidth=2.5, markersize=5, label='IV (Bartik)', zorder=3)
        ax.errorbar(d['horizon'], d['beta'],
                    yerr=[d['beta']-d['ci_lower'], d['ci_upper']-d['beta']],
                    fmt='none', ecolor='#b2182b', capsize=4, elinewidth=1.5, zorder=2)
    
    if out in ols_results:
        d = ols_results[out]
        ax.plot(d['horizon'], d['beta'], 's--', color='#4393c3',
                linewidth=1.5, markersize=4, label='OLS', alpha=0.7, zorder=2)
        ax.errorbar(d['horizon'], d['beta'],
                    yerr=[d['beta']-d['ci_lower'], d['ci_upper']-d['beta']],
                    fmt='none', ecolor='#4393c3', capsize=3, elinewidth=1.2,
                    alpha=0.7, zorder=2)
    
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.set_xlabel('Horizon (years)')
    ax.set_ylabel(r'$\hat{\beta}_h$')
    ax.set_title(OLABELS[out], fontweight='bold')
    ax.set_xticks(range(MAX_H + 1))
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=9)

fig.suptitle('Annual LP-IV: Effect of Unemployment on Task Composition\n'
             r'$T_{s,t+h} - T_{s,t-1} = \beta_h \cdot \widehat{\Delta U}_{st} + \delta \cdot Tech + \alpha_s + \theta_t + \varepsilon$',
             fontsize=13, fontweight='bold', y=1.03)
plt.tight_layout()
path = os.path.join(RESULTS_DIR, 'fig_lp_iv_annual.pdf')
fig.savefig(path, dpi=300, bbox_inches='tight')
print(f"  Saved: {path}")
plt.close()


# ============================================================
# STEP 5: Export
# ============================================================
print("\n" + "-" * 50)
print("STEP 5: Export")
print("-" * 50)

rows = []
for src, model in [(iv_results, 'IV'), (ols_results, 'OLS')]:
    for out, d in src.items():
        o = d.copy()
        o['outcome'] = out
        o['model'] = model
        rows.append(o)

if rows:
    out_df = pd.concat(rows, ignore_index=True)
    p = os.path.join(RESULTS_DIR, 'lp_iv_annual_results.csv')
    out_df.to_csv(p, index=False, float_format='%.6f')
    print(f"  Saved: {p}")


# ============================================================
# STEP 6: Summary
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\n  First stage F: {f_stat:.1f} ({'strong' if f_stat >= 10 else 'WEAK'})")
print(f"  pi = {pi:.2f} (Bartik -> Delta U)")

for out in OUTCOMES:
    if out in iv_results:
        d = iv_results[out]
        b0 = d.loc[d['horizon'] == 0, 'beta'].values
        bmax = d.loc[d['horizon'] == MAX_H, 'beta'].values
        p0 = d.loc[d['horizon'] == 0, 'pval'].values
        if len(b0) > 0 and len(bmax) > 0:
            sig = '***' if p0[0] < 0.01 else ('**' if p0[0] < 0.05 else ('*' if p0[0] < 0.1 else 'ns'))
            revert = abs(bmax[0]) < abs(b0[0]) * 0.5
            print(f"  {OLABELS[out]:>8s}: b(0)={b0[0]:+.5f} [{sig}], "
                  f"b({MAX_H})={bmax[0]:+.5f}, {'REVERTS' if revert else 'PERSISTS'}")

print(f"\n  Results: {RESULTS_DIR}")

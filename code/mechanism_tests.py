"""
Mechanism Tests
================
1. Control for industry composition change in second stage
   (Tests: is the routine effect within-industry or purely between-industry?)
2. LFP rate as outcome
   (Tests: is labor force selection driving the composition shift?)

Author: Ethan Johnston
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

def run_2sls_multi(y, endog, instr, exog_list, states, times):
    """2SLS with multiple exogenous controls, clustered SE."""
    y_dd = dd(y, states, times)
    end_dd = dd(endog, states, times)
    ins_dd = dd(instr, states, times)
    exog_dd = [dd(x, states, times) for x in exog_list]
    
    # Stage 1: endog on instrument + exog
    Z = np.column_stack([ins_dd] + [e.reshape(-1,1) for e in exog_dd])
    b1, _, _, _ = np.linalg.lstsq(Z, end_dd, rcond=None)
    end_hat = Z @ b1
    
    # Stage 2: y on fitted_endog + exog
    X2 = np.column_stack([end_hat] + [e.reshape(-1,1) for e in exog_dd])
    b2, _, _, _ = np.linalg.lstsq(X2, y_dd, rcond=None)
    
    # Residuals with actual endog
    X2a = np.column_stack([end_dd] + [e.reshape(-1,1) for e in exog_dd])
    resid = y_dd - X2a @ b2
    
    # Clustered SE
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
    
    return b2, se, n

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
# LOAD AND BUILD PANEL
# ============================================================
print("=" * 70)
print("MECHANISM TESTS")
print("=" * 70)

# Task composition
sm = pd.read_csv(os.path.join(DATA_DIR, "state_month_panel.csv"))
task_vars = ['mean_rti', 'mean_abstract', 'mean_routine', 'mean_manual',
             'share_routine', 'share_NRC', 'share_NRM']
sa = sm.groupby(['STATEFIP', 'YEAR']).apply(
    lambda g: pd.Series({v: wmean(g, v) for v in task_vars if v in g.columns})
).reset_index()
sa['STATEFIP'] = sa['STATEFIP'].astype(int)
sa['YEAR'] = sa['YEAR'].astype(int)

# Unemployment
laus = pd.read_csv(os.path.join(DATA_DIR, "laus_state_unemployment.csv"))
laus_a = laus.groupby(['STATEFIP', 'year']).agg(
    urate=('unemployment_rate', 'mean'),
    labor_force=('labor_force', 'mean'),
    employment_total=('employment_total', 'mean'),
    population=('population', 'mean'),
).reset_index().rename(columns={'year': 'YEAR'})
laus_a['STATEFIP'] = laus_a['STATEFIP'].astype(int)

# Compute LFP rate
laus_a['lfp_rate'] = laus_a['labor_force'] / laus_a['population'] * 100

# Bartik
bartik = pd.read_csv(os.path.join(DATA_DIR, "bartik_instrument.csv"))
bartik['STATEFIP'] = bartik['STATEFIP'].astype(int)

# Industry shares (for Bartik industry employment share)
ind_shares = pd.read_csv(os.path.join(DATA_DIR, "state_industry_shares.csv"))
ces = pd.read_csv(os.path.join(DATA_DIR, "ces_national_industry.csv"))

# Classify industries
ces_a = ces.groupby(['industry', 'year'])['employment'].mean().reset_index()
ces_a = ces_a.sort_values(['industry', 'year'])
ces_a['emp_lag'] = ces_a.groupby('industry')['employment'].shift(1)
ces_a['growth'] = (ces_a['employment'] - ces_a['emp_lag']) / ces_a['emp_lag']
growth_var = ces_a.groupby('industry')['growth'].var().reset_index()
growth_var.columns = ['industry', 'var_growth']
median_var = growth_var['var_growth'].median()
high_var_ind = set(growth_var[growth_var['var_growth'] >= median_var]['industry'])

# Compute state-year Bartik-industry employment share
ind_shares['is_bartik'] = ind_shares['industry'].isin(high_var_ind)
bartik_emp = ind_shares[ind_shares['is_bartik']].groupby(
    ['STATEFIP', 'YEAR']
)['emp_share'].sum().reset_index()
bartik_emp.columns = ['STATEFIP', 'YEAR', 'bartik_ind_share']

# Merge everything
df = sa.merge(laus_a, on=['STATEFIP', 'YEAR'], how='inner')
df = df.merge(bartik, on=['STATEFIP', 'YEAR'], how='inner')
df = df.merge(bartik_emp, on=['STATEFIP', 'YEAR'], how='left')
df = df.sort_values(['STATEFIP', 'YEAR']).reset_index(drop=True)

df['delta_urate'] = df.groupby('STATEFIP')['urate'].diff()

# Tech control
base_yr = df['YEAR'].min() + 1
base_sh = df[df['YEAR'] == base_yr].groupby('STATEFIP')['share_routine'].mean()
df = df.merge(base_sh.rename('rs_init'), on='STATEFIP', how='left')
df['time_trend'] = df['YEAR'] - df['YEAR'].min()
df['tc_routine'] = df['rs_init'] * df['time_trend']

# Change in Bartik industry share (industry composition control)
df['delta_bartik_ind_share'] = df.groupby('STATEFIP')['bartik_ind_share'].diff()

# LP outcomes for task measures
for v in ['mean_routine', 'mean_abstract', 'mean_manual', 'mean_rti']:
    df[f'{v}_L1'] = df.groupby('STATEFIP')[v].shift(1)
    for h in range(MAX_H + 1):
        df[f'{v}_F{h}'] = df.groupby('STATEFIP')[v].shift(-h)
        df[f'{v}_lp{h}'] = df[f'{v}_F{h}'] - df[f'{v}_L1']

# LP outcomes for LFP
df['lfp_L1'] = df.groupby('STATEFIP')['lfp_rate'].shift(1)
for h in range(MAX_H + 1):
    df[f'lfp_F{h}'] = df.groupby('STATEFIP')['lfp_rate'].shift(-h)
    df[f'lfp_lp{h}'] = df[f'lfp_F{h}'] - df['lfp_L1']

# LP outcomes for Bartik industry share (to see how composition shifts)
df['bis_L1'] = df.groupby('STATEFIP')['bartik_ind_share'].shift(1)
for h in range(MAX_H + 1):
    df[f'bis_F{h}'] = df.groupby('STATEFIP')['bartik_ind_share'].shift(-h)
    df[f'bis_lp{h}'] = df[f'bis_F{h}'] - df['bis_L1']

print(f"Panel: {len(df):,} obs, {df['STATEFIP'].nunique()} states")
print(f"Valid delta_bartik_ind_share: {df['delta_bartik_ind_share'].notna().sum():,}")
print(f"Valid lfp_rate: {df['lfp_rate'].notna().sum():,}")


# ============================================================
# TEST 1: Controlling for Industry Composition Change
# ============================================================
print("\n" + "=" * 70)
print("TEST 1: Routine LP-IV Controlling for Industry Composition Change")
print("=" * 70)

print("\n  Comparison:")
print(f"  {'':>3s}  {'Baseline':>25s}    {'+ Ind. Comp. Control':>25s}")
print(f"  {'h':>3s}  {'beta':>10s} {'se':>8s} {'':>4s}    {'beta':>10s} {'se':>8s} {'':>4s}")

baseline_rows = []
controlled_rows = []

for h in range(MAX_H + 1):
    lp_var = f'mean_routine_lp{h}'
    
    # Required columns
    base_cols = [lp_var, 'delta_urate', 'bartik_shock', 'tc_routine']
    ctrl_cols = base_cols + ['delta_bartik_ind_share']
    
    # --- Baseline (no industry control) ---
    valid_b = df[base_cols].notna().all(axis=1)
    b_beta, b_se, b_p = np.nan, np.nan, np.nan
    if valid_b.sum() > 100:
        try:
            betas, ses, n = run_2sls_multi(
                df.loc[valid_b, lp_var].values,
                df.loc[valid_b, 'delta_urate'].values,
                df.loc[valid_b, 'bartik_shock'].values,
                [df.loc[valid_b, 'tc_routine'].values],
                df.loc[valid_b, 'STATEFIP'].values,
                df.loc[valid_b, 'YEAR'].values
            )
            b_beta, b_se = betas[0], ses[0]
            b_t = b_beta / b_se if b_se > 0 else 0
            b_p = 2 * (1 - norm.cdf(abs(b_t)))
            baseline_rows.append({'horizon': h, 'beta': b_beta, 'se': b_se, 'pval': b_p, 'n': n,
                                  'ci_lower': b_beta - 1.96*b_se, 'ci_upper': b_beta + 1.96*b_se})
        except:
            pass
    
    # --- With industry composition control ---
    valid_c = df[ctrl_cols].notna().all(axis=1)
    c_beta, c_se, c_p = np.nan, np.nan, np.nan
    if valid_c.sum() > 100:
        try:
            # Exog list: tc_routine + delta_bartik_ind_share
            # Also need LP of delta_bartik_ind_share at horizon h
            # Actually, we want to control for the CUMULATIVE change in industry composition
            # through horizon h, not just the contemporaneous change
            # Use bis_lp{h} = bartik_ind_share_{t+h} - bartik_ind_share_{t-1}
            bis_var = f'bis_lp{h}'
            if bis_var in df.columns:
                ctrl_cols_h = [lp_var, 'delta_urate', 'bartik_shock', 'tc_routine', bis_var]
                valid_ch = df[ctrl_cols_h].notna().all(axis=1)
                
                if valid_ch.sum() > 100:
                    betas, ses, n = run_2sls_multi(
                        df.loc[valid_ch, lp_var].values,
                        df.loc[valid_ch, 'delta_urate'].values,
                        df.loc[valid_ch, 'bartik_shock'].values,
                        [df.loc[valid_ch, 'tc_routine'].values,
                         df.loc[valid_ch, bis_var].values],
                        df.loc[valid_ch, 'STATEFIP'].values,
                        df.loc[valid_ch, 'YEAR'].values
                    )
                    c_beta, c_se = betas[0], ses[0]
                    c_t = c_beta / c_se if c_se > 0 else 0
                    c_p = 2 * (1 - norm.cdf(abs(c_t)))
                    
                    # Also get the industry composition coefficient
                    ind_beta = betas[2] if len(betas) > 2 else np.nan
                    ind_se = ses[2] if len(ses) > 2 else np.nan
                    
                    controlled_rows.append({
                        'horizon': h, 'beta': c_beta, 'se': c_se, 'pval': c_p, 'n': n,
                        'ci_lower': c_beta - 1.96*c_se, 'ci_upper': c_beta + 1.96*c_se,
                        'ind_comp_beta': ind_beta, 'ind_comp_se': ind_se,
                    })
        except:
            pass
    
    # Print comparison
    b_str = f"{b_beta:+.5f} ({b_se:.5f}){star(b_p):>4s}" if not np.isnan(b_beta) else "  ---"
    c_str = f"{c_beta:+.5f} ({c_se:.5f}){star(c_p):>4s}" if not np.isnan(c_beta) else "  ---"
    print(f"  {h:3d}  {b_str}    {c_str}")

# Summary
if baseline_rows and controlled_rows:
    bl = pd.DataFrame(baseline_rows)
    cl = pd.DataFrame(controlled_rows)
    
    b0_base = bl.loc[bl['horizon']==0, 'beta'].values
    b0_ctrl = cl.loc[cl['horizon']==0, 'beta'].values
    
    if len(b0_base) > 0 and len(b0_ctrl) > 0:
        attenuation = (1 - abs(b0_ctrl[0]) / abs(b0_base[0])) * 100 if abs(b0_base[0]) > 0 else 0
        print(f"\n  At h=0: baseline = {b0_base[0]:.5f}, controlled = {b0_ctrl[0]:.5f}")
        print(f"  Attenuation: {attenuation:.1f}%")
        
        if attenuation < 30:
            print(f"  Interpretation: <30% attenuation — effect is NOT purely between-industry")
            print(f"  The task composition shift includes a within-industry component")
        elif attenuation < 60:
            print(f"  Interpretation: moderate attenuation — partially between-industry")
        else:
            print(f"  Interpretation: large attenuation — mostly between-industry composition")

# Industry composition coefficient
if controlled_rows:
    print(f"\n  Industry composition control coefficients (Δ Bartik ind. share):")
    for r in controlled_rows:
        if not np.isnan(r.get('ind_comp_beta', np.nan)):
            ind_t = r['ind_comp_beta'] / r['ind_comp_se'] if r['ind_comp_se'] > 0 else 0
            ind_p = 2 * (1 - norm.cdf(abs(ind_t)))
            print(f"    h={r['horizon']}: {r['ind_comp_beta']:+.5f} ({r['ind_comp_se']:.5f}){star(ind_p):>4s}")


# ============================================================
# TEST 2: LFP as Outcome (Selection Test)
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: Labor Force Participation as Outcome (Selection)")
print("=" * 70)

lfp_rows = []
for h in range(MAX_H + 1):
    lp_var = f'lfp_lp{h}'
    cols = [lp_var, 'delta_urate', 'bartik_shock', 'tc_routine']
    valid = df[cols].notna().all(axis=1)
    if valid.sum() < 100:
        continue
    
    try:
        betas, ses, n = run_2sls_multi(
            df.loc[valid, lp_var].values,
            df.loc[valid, 'delta_urate'].values,
            df.loc[valid, 'bartik_shock'].values,
            [df.loc[valid, 'tc_routine'].values],
            df.loc[valid, 'STATEFIP'].values,
            df.loc[valid, 'YEAR'].values
        )
        b, se = betas[0], ses[0]
        t = b / se if se > 0 else 0
        p = 2 * (1 - norm.cdf(abs(t)))
        lfp_rows.append({'horizon': h, 'beta': b, 'se': se, 'pval': p, 'n': n,
                         'ci_lower': b - 1.96*se, 'ci_upper': b + 1.96*se})
        print(f"  h={h}: {b:+.4f} ({se:.4f}){star(p):>4s}  N={n:,}")
    except Exception as e:
        print(f"  h={h}: failed — {e}")

if lfp_rows:
    lfp_df = pd.DataFrame(lfp_rows)
    print(f"\n  Interpretation:")
    b0 = lfp_df.loc[lfp_df['horizon']==0, 'beta'].values
    if len(b0) > 0:
        print(f"    h=0 effect on LFP: {b0[0]:+.4f} pp per 1pp unemployment increase")
        print(f"    Compare to routine task effect: -0.0062")
        print(f"    If LFP response is small relative to task composition effect,")
        print(f"    selection into/out of labor force is unlikely to drive the main result.")


# ============================================================
# TEST 3: Bartik Industry Share as Outcome (Composition Channel)
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: Bartik Industry Employment Share as Outcome")
print("=" * 70)

bis_rows = []
for h in range(MAX_H + 1):
    lp_var = f'bis_lp{h}'
    cols = [lp_var, 'delta_urate', 'bartik_shock', 'tc_routine']
    valid = df[cols].notna().all(axis=1)
    if valid.sum() < 100:
        continue
    
    try:
        betas, ses, n = run_2sls_multi(
            df.loc[valid, lp_var].values,
            df.loc[valid, 'delta_urate'].values,
            df.loc[valid, 'bartik_shock'].values,
            [df.loc[valid, 'tc_routine'].values],
            df.loc[valid, 'STATEFIP'].values,
            df.loc[valid, 'YEAR'].values
        )
        b, se = betas[0], ses[0]
        t = b / se if se > 0 else 0
        p = 2 * (1 - norm.cdf(abs(t)))
        bis_rows.append({'horizon': h, 'beta': b, 'se': se, 'pval': p, 'n': n,
                         'ci_lower': b - 1.96*se, 'ci_upper': b + 1.96*se})
        print(f"  h={h}: {b:+.5f} ({se:.5f}){star(p):>4s}  N={n:,}")
    except Exception as e:
        print(f"  h={h}: failed — {e}")

if bis_rows:
    bis_df = pd.DataFrame(bis_rows)
    print(f"\n  Interpretation:")
    print(f"    Shows how much industry composition shifts after a demand shock.")
    print(f"    This is the 'between-industry' channel that the composition control absorbs.")


# ============================================================
# PLOTS
# ============================================================
print("\n" + "=" * 70)
print("PLOTS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Baseline vs Controlled routine
ax = axes[0, 0]
if baseline_rows:
    d = pd.DataFrame(baseline_rows)
    ax.plot(d['horizon'], d['beta'], 'o-', color='#b2182b', linewidth=2.5,
            markersize=5, label='Baseline', alpha=0.85)
    ax.errorbar(d['horizon'], d['beta'],
                yerr=[d['beta']-d['ci_lower'], d['ci_upper']-d['beta']],
                fmt='none', ecolor='#b2182b', capsize=4, elinewidth=1.5, zorder=2)
if controlled_rows:
    d = pd.DataFrame(controlled_rows)
    ax.plot(d['horizon'], d['beta'], 's--', color='#4393c3', linewidth=2,
            markersize=5, label='+ Ind. composition control', alpha=0.85)
    ax.errorbar(d['horizon'], d['beta'],
                yerr=[d['beta']-d['ci_lower'], d['ci_upper']-d['beta']],
                fmt='none', ecolor='#4393c3', capsize=3, elinewidth=1.2, alpha=0.85, zorder=2)
ax.axhline(0, color='gray', linewidth=0.8)
ax.set_xlabel('Horizon (years)')
ax.set_ylabel(r'$\hat{\beta}_h$')
ax.set_title('A: Routine — Baseline vs Ind. Comp. Control', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)
ax.set_xticks(range(MAX_H + 1))

# Panel B: LFP response
ax = axes[0, 1]
if lfp_rows:
    d = pd.DataFrame(lfp_rows)
    ax.plot(d['horizon'], d['beta'], 'o-', color='#1b7837', linewidth=2.5,
            markersize=5, alpha=0.85)
    ax.errorbar(d['horizon'], d['beta'],
                yerr=[d['beta']-d['ci_lower'], d['ci_upper']-d['beta']],
                fmt='none', ecolor='#1b7837', capsize=4, elinewidth=1.5, zorder=2)
ax.axhline(0, color='gray', linewidth=0.8)
ax.set_xlabel('Horizon (years)')
ax.set_ylabel(r'$\hat{\beta}_h$ (pp)')
ax.set_title('B: Labor Force Participation Rate', fontweight='bold')
ax.grid(True, alpha=0.2)
ax.set_xticks(range(MAX_H + 1))

# Panel C: Bartik industry share response
ax = axes[1, 0]
if bis_rows:
    d = pd.DataFrame(bis_rows)
    ax.plot(d['horizon'], d['beta'], 'o-', color='#762a83', linewidth=2.5,
            markersize=5, alpha=0.85)
    ax.errorbar(d['horizon'], d['beta'],
                yerr=[d['beta']-d['ci_lower'], d['ci_upper']-d['beta']],
                fmt='none', ecolor='#762a83', capsize=4, elinewidth=1.5, zorder=2)
ax.axhline(0, color='gray', linewidth=0.8)
ax.set_xlabel('Horizon (years)')
ax.set_ylabel(r'$\hat{\beta}_h$')
ax.set_title('C: Bartik Industry Employment Share', fontweight='bold')
ax.grid(True, alpha=0.2)
ax.set_xticks(range(MAX_H + 1))

# Panel D: Attenuation comparison
ax = axes[1, 1]
if baseline_rows and controlled_rows:
    bl = pd.DataFrame(baseline_rows)
    cl = pd.DataFrame(controlled_rows)
    merged = bl[['horizon', 'beta']].merge(
        cl[['horizon', 'beta']], on='horizon', suffixes=('_base', '_ctrl')
    )
    merged['attenuation_pct'] = (1 - merged['beta_ctrl'].abs() / merged['beta_base'].abs()) * 100
    merged.loc[merged['beta_base'].abs() < 0.0001, 'attenuation_pct'] = 0
    
    ax.bar(merged['horizon'], merged['attenuation_pct'], color='#d6604d', alpha=0.7)
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.axhline(50, color='black', linewidth=1, linestyle='--', alpha=0.3, label='50% threshold')
    ax.set_xlabel('Horizon (years)')
    ax.set_ylabel('Attenuation (%)')
    ax.set_title('D: % Attenuation from Ind. Comp. Control', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xticks(range(MAX_H + 1))
    ax.set_ylim(-20, 100)

plt.tight_layout()
path = os.path.join(ROB_DIR, 'fig_mechanism_tests.pdf')
fig.savefig(path, dpi=300, bbox_inches='tight')
print(f"  Saved: {path}")
plt.close()


# ============================================================
# EXPORT
# ============================================================
print("\n" + "-" * 50)
print("Export")
print("-" * 50)

all_rows = []
if baseline_rows:
    d = pd.DataFrame(baseline_rows)
    d['specification'] = 'baseline'
    d['outcome'] = 'mean_routine'
    all_rows.append(d)
if controlled_rows:
    d = pd.DataFrame(controlled_rows)
    d['specification'] = 'ind_comp_control'
    d['outcome'] = 'mean_routine'
    all_rows.append(d)
if lfp_rows:
    d = pd.DataFrame(lfp_rows)
    d['specification'] = 'baseline'
    d['outcome'] = 'lfp_rate'
    all_rows.append(d)
if bis_rows:
    d = pd.DataFrame(bis_rows)
    d['specification'] = 'baseline'
    d['outcome'] = 'bartik_ind_share'
    all_rows.append(d)

if all_rows:
    out = pd.concat(all_rows, ignore_index=True)
    p = os.path.join(ROB_DIR, 'mechanism_tests_results.csv')
    out.to_csv(p, index=False, float_format='%.6f')
    print(f"  Saved: {p}")

print("\n" + "=" * 70)
print("MECHANISM TESTS COMPLETE")
print("=" * 70)

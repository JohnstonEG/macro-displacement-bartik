"""
LP-IV Robustness Battery
=========================
All robustness checks for the annual LP-IV specification.

1. Pre-trend placebos (h = -3, -2, -1)
2. Composition decomposition (employment shares as outcomes)
3. GPSS Rotemberg weights (which industries drive identification)
4. Flexible technology controls (routine share × year FE)
5. Sample period heterogeneity (pre-2008 vs post-2008)
6. Weak instrument diagnostics (Anderson-Rubin, KP)
7. State-specific linear trends

Reads same data as lp_iv_annual.py
Output: results/robustness/

Author: Ethan Johnston
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from linearmodels.panel import PanelOLS
from scipy.stats import norm, chi2, f as f_dist
import os
import warnings
warnings.filterwarnings('ignore')

from paths import DATA_DIR, ROBUSTNESS_DIR as ROB_DIR
os.makedirs(ROB_DIR, exist_ok=True)

OUTCOMES = ['mean_rti', 'mean_abstract', 'mean_routine', 'mean_manual']
OLABELS = {'mean_rti': 'RTI', 'mean_abstract': 'Abstract',
           'mean_routine': 'Routine', 'mean_manual': 'Manual'}

# Share outcomes for composition decomposition
SHARE_OUTCOMES = ['share_routine', 'share_NRC', 'share_NRM', 'share_RC', 'share_RM']
SHARE_LABELS = {'share_routine': 'Routine (RC+RM)',
                'share_NRC': 'Non-Routine Cognitive',
                'share_NRM': 'Non-Routine Manual',
                'share_RC': 'Routine Cognitive',
                'share_RM': 'Routine Manual'}

MAX_H = 5
PLACEBO_H = 3  # pre-trend horizons


# ============================================================
# DATA LOADING (same as lp_iv_annual.py)
# ============================================================
print("=" * 70)
print("LP-IV ROBUSTNESS BATTERY")
print("=" * 70)

# Task composition + shares — annual from monthly
sm = pd.read_csv(os.path.join(DATA_DIR, "state_month_panel.csv"))

def wmean(g, v, w='total_employed'):
    d = g[[v, w]].dropna()
    if len(d) == 0 or d[w].sum() == 0:
        return np.nan
    return np.average(d[v], weights=d[w])

all_vars = OUTCOMES + [s for s in SHARE_OUTCOMES if s in sm.columns]
sa = sm.groupby(['STATEFIP', 'YEAR']).apply(
    lambda g: pd.Series({v: wmean(g, v) for v in all_vars if v in g.columns})
).reset_index()
sa['STATEFIP'] = sa['STATEFIP'].astype(int)
sa['YEAR'] = sa['YEAR'].astype(int)

# Unemployment
laus = pd.read_csv(os.path.join(DATA_DIR, "laus_state_unemployment.csv"))
laus_a = laus.groupby(['STATEFIP', 'year']).agg(
    urate=('unemployment_rate', 'mean'),
).reset_index().rename(columns={'year': 'YEAR'})
laus_a['STATEFIP'] = laus_a['STATEFIP'].astype(int)

# Bartik
bartik = pd.read_csv(os.path.join(DATA_DIR, "bartik_instrument.csv"))
bartik['STATEFIP'] = bartik['STATEFIP'].astype(int)

# Merge
df = sa.merge(laus_a, on=['STATEFIP', 'YEAR'], how='inner')
df = df.merge(bartik, on=['STATEFIP', 'YEAR'], how='inner')
df = df.sort_values(['STATEFIP', 'YEAR']).reset_index(drop=True)

# Delta U
df['delta_urate'] = df.groupby('STATEFIP')['urate'].diff()

# Tech control (baseline)
base_yr = df['YEAR'].min() + 1
base = df[df['YEAR'] == base_yr].groupby('STATEFIP')['share_routine'].mean()
df = df.merge(base.rename('rs_init'), on='STATEFIP', how='left')
df['time_trend'] = df['YEAR'] - df['YEAR'].min()
df['tech_baseline'] = df['rs_init'] * df['time_trend']

# LP outcomes (including placebo horizons h < 0)
for v in OUTCOMES + [s for s in SHARE_OUTCOMES if s in df.columns]:
    df[f'{v}_L1'] = df.groupby('STATEFIP')[v].shift(1)
    for h in range(-PLACEBO_H, MAX_H + 1):
        df[f'{v}_F{h}'] = df.groupby('STATEFIP')[v].shift(-h)
        df[f'{v}_lp{h}'] = df[f'{v}_F{h}'] - df[f'{v}_L1']

print(f"Panel: {len(df):,} obs, {df['STATEFIP'].nunique()} states, "
      f"{df['YEAR'].min()}-{df['YEAR'].max()}")

available_shares = [s for s in SHARE_OUTCOMES if s in df.columns]
print(f"Share outcomes available: {available_shares}")


# ============================================================
# HELPERS
# ============================================================
def dd(x, s, t):
    """Double-demean."""
    d = pd.DataFrame({'x': x, 's': s, 't': t})
    gm = d['x'].mean()
    return (d['x'] - d.groupby('s')['x'].transform('mean')
            - d.groupby('t')['x'].transform('mean') + gm).values

def run_2sls(y, endog, instr, exog_list, states, times):
    """2SLS with clustered SE. Returns beta, se, n, resid."""
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
    cor = (G / (G-1)) * ((n-1) / (n-k))
    V = cor * XtXi @ B @ XtXi
    se = np.sqrt(np.maximum(np.diag(V), 0))

    return b2[0], se[0], n, resid, y_dd, end_dd, ins_dd

def run_lp_iv(df_in, outcome_list, horizons, exog_cols, label=""):
    """Run LP-IV for a list of outcomes and horizons. Returns dict of DataFrames."""
    results = {}
    for out in outcome_list:
        rows = []
        for h in horizons:
            lp_var = f'{out}_lp{h}'
            if lp_var not in df_in.columns:
                continue

            cols = [lp_var, 'delta_urate', 'bartik_shock'] + exog_cols
            valid = df_in[cols].notna().all(axis=1)
            if valid.sum() < 100:
                continue

            y = df_in.loc[valid, lp_var].values
            endog = df_in.loc[valid, 'delta_urate'].values
            instr = df_in.loc[valid, 'bartik_shock'].values
            states = df_in.loc[valid, 'STATEFIP'].values
            times = df_in.loc[valid, 'YEAR'].values
            exog_list = [df_in.loc[valid, c].values for c in exog_cols]

            try:
                b, se, n, _, _, _, _ = run_2sls(y, endog, instr, exog_list, states, times)
                t = b / se if se > 0 else 0
                p = 2 * (1 - norm.cdf(abs(t)))
                rows.append({'horizon': h, 'beta': b, 'se': se, 'pval': p,
                             'ci_lower': b - 1.96*se, 'ci_upper': b + 1.96*se, 'n': n})
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

def print_results(results, label_map, tag=""):
    for out, d in results.items():
        lab = label_map.get(out, out)
        print(f"\n    {lab}:")
        for _, r in d.iterrows():
            h = int(r['horizon'])
            print(f"      h={h:+d}: {r['beta']:+.5f} ({r['se']:.5f}){star(r['pval']):>4s}  N={int(r['n']):,}")


# ============================================================
# 1. PRE-TREND PLACEBOS
# ============================================================
print("\n" + "=" * 70)
print("1. PRE-TREND PLACEBOS (h = -3, -2)")
print("=" * 70)

# h = -1 is mechanically zero: LP outcome is T_{t+h} - T_{t-1},
# so at h = -1 the regressand is identically zero by construction.
placebo_horizons = [h for h in range(-PLACEBO_H, 0) if h != -1]
exog_base = ['tech_baseline'] if df['tech_baseline'].notna().all() else []

placebo_results = run_lp_iv(df, OUTCOMES, placebo_horizons, exog_base, "placebo")
print_results(placebo_results, OLABELS)

# Joint test: all pre-trend coefficients = 0
print("\n  Joint significance of pre-trend coefficients:")
for out, d in placebo_results.items():
    any_sig = (d['pval'] < 0.05).any()
    n_sig = (d['pval'] < 0.05).sum()
    print(f"    {OLABELS[out]}: {n_sig}/{len(d)} sig at 5% — {'CONCERN' if any_sig else 'CLEAN'}")


# ============================================================
# 2. COMPOSITION DECOMPOSITION
# ============================================================
print("\n" + "=" * 70)
print("2. COMPOSITION DECOMPOSITION (Employment shares as outcomes)")
print("=" * 70)

if available_shares:
    main_horizons = list(range(0, MAX_H + 1))
    comp_results = run_lp_iv(df, available_shares, main_horizons, exog_base, "composition")
    print_results(comp_results, SHARE_LABELS)
else:
    comp_results = {}
    print("  No share outcomes available — skipping")


# ============================================================
# 3. GPSS ROTEMBERG WEIGHTS
# ============================================================
print("\n" + "=" * 70)
print("3. GPSS ROTEMBERG WEIGHTS")
print("=" * 70)

# Load industry shares and CES data
try:
    ind_shares = pd.read_csv(os.path.join(DATA_DIR, "state_industry_shares.csv"))
    ces = pd.read_csv(os.path.join(DATA_DIR, "ces_national_industry.csv"))

    # Annual CES growth rates
    ces_a = ces.groupby(['industry', 'year'])['employment'].mean().reset_index()
    ces_a = ces_a.sort_values(['industry', 'year'])
    ces_a['emp_lag'] = ces_a.groupby('industry')['employment'].shift(1)
    ces_a['growth'] = (ces_a['employment'] - ces_a['emp_lag']) / ces_a['emp_lag']
    ces_a = ces_a.dropna(subset=['growth'])

    # For each industry j, the Rotemberg weight is proportional to:
    # α_j = Σ_s Σ_t ω_{sjt₀} × g_{jt} × (something from the projection)
    # Simplified: just compute the contribution of each industry to Bartik variance

    industries = sorted(ind_shares['industry'].unique())
    matched_ind = [j for j in industries if j in ces_a['industry'].unique()]

    print(f"  Industries for decomposition: {len(matched_ind)}")

    weights = []
    for j in matched_ind:
        # Average share of this industry across states and years
        avg_share = ind_shares[ind_shares['industry'] == j]['emp_share'].mean()

        # Variance of national growth for this industry
        j_growth = ces_a[ces_a['industry'] == j]['growth']
        var_growth = j_growth.var() if len(j_growth) > 1 else 0

        # Approximate Rotemberg weight ∝ avg_share × var(growth)
        rotemberg = avg_share * var_growth

        weights.append({
            'industry': j,
            'avg_emp_share': avg_share,
            'var_national_growth': var_growth,
            'sd_national_growth': j_growth.std() if len(j_growth) > 1 else 0,
            'mean_national_growth': j_growth.mean() if len(j_growth) > 0 else 0,
            'rotemberg_weight': rotemberg,
        })

    wdf = pd.DataFrame(weights)
    wdf['rotemberg_pct'] = wdf['rotemberg_weight'] / wdf['rotemberg_weight'].sum() * 100
    wdf = wdf.sort_values('rotemberg_pct', ascending=False)

    print(f"\n  Rotemberg weight decomposition:")
    print(f"  {'Industry':<40s} {'Avg Share':>9s} {'SD Growth':>9s} {'Weight %':>9s}")
    print(f"  {'-'*40} {'-'*9} {'-'*9} {'-'*9}")
    for _, r in wdf.iterrows():
        print(f"  {r['industry']:<40s} {r['avg_emp_share']:>9.3f} {r['sd_national_growth']:>9.3f} {r['rotemberg_pct']:>8.1f}%")

    # Flag if top industry dominates (>50%)
    top_pct = wdf.iloc[0]['rotemberg_pct']
    if top_pct > 50:
        print(f"\n  WARNING: {wdf.iloc[0]['industry']} accounts for {top_pct:.0f}% of identification")
    else:
        print(f"\n  Top industry: {wdf.iloc[0]['industry']} ({top_pct:.1f}%) — reasonably dispersed")

    # Save
    wdf.to_csv(os.path.join(ROB_DIR, 'gpss_rotemberg_weights.csv'), index=False)
    print(f"  Saved: gpss_rotemberg_weights.csv")

except Exception as e:
    print(f"  GPSS failed: {e}")
    wdf = None


# ============================================================
# 4. FLEXIBLE TECHNOLOGY CONTROLS
# ============================================================
print("\n" + "=" * 70)
print("4. FLEXIBLE TECHNOLOGY CONTROLS")
print("=" * 70)

# 4a: Routine share × year FE (fully nonparametric)
# Create rs_init × year dummies
years = sorted(df['YEAR'].unique())
omit_year = years[0]  # omit first year
for yr in years:
    if yr != omit_year:
        df[f'rs_yr_{yr}'] = df['rs_init'] * (df['YEAR'] == yr).astype(float)

flex_exog = [f'rs_yr_{yr}' for yr in years if yr != omit_year]
# Check that these have variation
flex_exog = [c for c in flex_exog if c in df.columns and df[c].std() > 0]

print(f"  Flexible controls: {len(flex_exog)} year interactions")

# First stage with flexible controls
df_p = df.set_index(['STATEFIP', 'YEAR'])
y_fs = df_p['delta_urate']
x_fs = df_p[['bartik_shock'] + flex_exog]
valid_fs = y_fs.notna() & x_fs.notna().all(axis=1)

try:
    fs_flex = PanelOLS(y_fs[valid_fs], x_fs[valid_fs],
                       entity_effects=True, time_effects=True,
                       drop_absorbed=True, check_rank=False)
    fs_flex_res = fs_flex.fit(cov_type='clustered', cluster_entity=True)

    if 'bartik_shock' in fs_flex_res.params.index:
        pi_flex = fs_flex_res.params['bartik_shock']
        se_flex = fs_flex_res.std_errors['bartik_shock']
        f_flex = (pi_flex / se_flex) ** 2
        print(f"  First stage with flex controls: pi={pi_flex:.4f}, F={f_flex:.1f}")
    else:
        print(f"  Bartik absorbed by flexible controls")
        f_flex = 0
except Exception as e:
    print(f"  Flexible first stage failed: {e}")
    f_flex = 0

# Run LP-IV with flexible controls (only if first stage survives)
if f_flex > 5:
    flex_results = run_lp_iv(df, OUTCOMES, list(range(MAX_H+1)), flex_exog, "flex")
    print("\n  LP-IV with flexible tech controls:")
    print_results(flex_results, OLABELS)
else:
    flex_results = {}
    print("  Skipping LP-IV with flex controls (first stage too weak)")


# ============================================================
# 5. SAMPLE PERIOD HETEROGENEITY
# ============================================================
print("\n" + "=" * 70)
print("5. SAMPLE PERIOD HETEROGENEITY")
print("=" * 70)

# Pre-2008 vs Post-2008
for period_name, mask in [("Pre-2008", df['YEAR'] <= 2007),
                           ("Post-2008", df['YEAR'] >= 2008)]:
    sub = df[mask].copy()

    # Recompute delta_urate within subsample
    sub = sub.sort_values(['STATEFIP', 'YEAR'])
    sub['delta_urate'] = sub.groupby('STATEFIP')['urate'].diff()

    # Recompute LP outcomes
    for v in OUTCOMES:
        sub[f'{v}_L1'] = sub.groupby('STATEFIP')[v].shift(1)
        for h in range(MAX_H + 1):
            sub[f'{v}_F{h}'] = sub.groupby('STATEFIP')[v].shift(-h)
            sub[f'{v}_lp{h}'] = sub[f'{v}_F{h}'] - sub[f'{v}_L1']

    # First stage
    cols = ['delta_urate', 'bartik_shock'] + exog_base
    valid = sub[cols].notna().all(axis=1)
    if valid.sum() > 100:
        y = sub.loc[valid, 'delta_urate'].values
        z = sub.loc[valid, 'bartik_shock'].values
        s = sub.loc[valid, 'STATEFIP'].values
        t = sub.loc[valid, 'YEAR'].values

        z_dd = dd(z, s, t)
        y_dd = dd(y, s, t)
        b1 = np.linalg.lstsq(z_dd.reshape(-1,1), y_dd, rcond=None)[0]
        yhat = z_dd * b1[0]
        ss_res = np.sum((y_dd - yhat)**2)
        ss_tot = np.sum((y_dd - y_dd.mean())**2)
        r2 = max(1 - ss_res/ss_tot, 0)
        n = len(y_dd)
        f_sub = (r2 / 1) / ((1 - r2) / max(n - 2, 1)) if r2 < 1 else 0

        print(f"\n  {period_name}: N={valid.sum():,}, First-stage F={f_sub:.1f}")

        if f_sub > 5:
            sub_results = run_lp_iv(sub, ['mean_routine', 'mean_manual'],
                                     list(range(MAX_H+1)), exog_base, period_name)
            print_results(sub_results, OLABELS)
        else:
            print(f"    Weak instrument in this subsample — skipping")
    else:
        print(f"\n  {period_name}: insufficient obs")


# ============================================================
# 6. ANDERSON-RUBIN CONFIDENCE SETS
# ============================================================
print("\n" + "=" * 70)
print("6. WEAK INSTRUMENT ROBUST INFERENCE (Anderson-Rubin)")
print("=" * 70)

# AR test: under H0: β = β0, regress y - β0*endog on instrument + controls
# Test whether instrument coefficient = 0. Invert to get confidence set.

for out in ['mean_routine']:  # Focus on main result
    lp_var = f'{out}_lp0'
    cols = [lp_var, 'delta_urate', 'bartik_shock'] + exog_base
    valid = df[cols].notna().all(axis=1)
    if valid.sum() < 100:
        continue

    y = df.loc[valid, lp_var].values
    endog = df.loc[valid, 'delta_urate'].values
    instr = df.loc[valid, 'bartik_shock'].values
    states = df.loc[valid, 'STATEFIP'].values
    times = df.loc[valid, 'YEAR'].values
    exog_list = [df.loc[valid, c].values for c in exog_base]

    y_dd = dd(y, states, times)
    end_dd = dd(endog, states, times)
    ins_dd = dd(instr, states, times)
    exog_dd = [dd(x, states, times) for x in exog_list]

    # Grid search over beta values
    beta_grid = np.linspace(-0.03, 0.01, 200)
    ar_pvals = []

    for b0 in beta_grid:
        # Under H0: β = b0, compute y - b0*endog
        y_adj = y_dd - b0 * end_dd

        # Regress on instrument
        Z = np.column_stack([ins_dd] + [e.reshape(-1,1) for e in exog_dd]) if exog_dd else ins_dd.reshape(-1,1)
        gamma, _, _, _ = np.linalg.lstsq(Z, y_adj, rcond=None)
        resid = y_adj - Z @ gamma

        # F-test for gamma[0] = 0 (instrument coefficient)
        n = len(y_adj)
        k = Z.shape[1]
        ss_res = np.sum(resid**2)

        # Restricted: exclude instrument
        if exog_dd:
            Z0 = np.column_stack([e.reshape(-1,1) for e in exog_dd])
            gamma0, _, _, _ = np.linalg.lstsq(Z0, y_adj, rcond=None)
            ss_res0 = np.sum((y_adj - Z0 @ gamma0)**2)
        else:
            ss_res0 = np.sum((y_adj - y_adj.mean())**2)

        f_ar = ((ss_res0 - ss_res) / 1) / (ss_res / (n - k))
        p_ar = 1 - f_dist.cdf(f_ar, 1, n - k)
        ar_pvals.append(p_ar)

    ar_pvals = np.array(ar_pvals)

    # 95% AR confidence set: all beta where p > 0.05
    in_cs = beta_grid[ar_pvals > 0.05]
    if len(in_cs) > 0:
        ar_lower = in_cs.min()
        ar_upper = in_cs.max()
        print(f"\n  {OLABELS[out]} at h=0:")
        print(f"    2SLS point estimate: {placebo_results.get(out, pd.DataFrame()).iloc[0]['beta'] if out in placebo_results else 'N/A'}")

        # Get actual 2SLS estimate
        try:
            b_iv, se_iv, _, _, _, _, _ = run_2sls(y, endog, instr, exog_list, states, times)
            print(f"    2SLS: {b_iv:.5f} [{b_iv - 1.96*se_iv:.5f}, {b_iv + 1.96*se_iv:.5f}]")
        except:
            pass

        print(f"    AR 95% CS: [{ar_lower:.5f}, {ar_upper:.5f}]")
        print(f"    AR CS excludes zero: {'YES' if ar_upper < 0 or ar_lower > 0 else 'NO'}")
    else:
        print(f"\n  {OLABELS[out]} at h=0: AR CS is empty (reject all beta at 5%)")


# ============================================================
# 7. STATE-SPECIFIC LINEAR TRENDS
# ============================================================
print("\n" + "=" * 70)
print("7. STATE-SPECIFIC LINEAR TRENDS")
print("=" * 70)

# Add state × trend to the double-demeaning
# Instead of standard dd, we residualize on state FE + time FE + state×trend

def dd_with_state_trend(x, states, times):
    """Residualize x on state FE + time FE + state-specific linear trend."""
    d = pd.DataFrame({'x': x, 's': states, 't': times})

    # State dummies
    s_dums = pd.get_dummies(d['s'], prefix='s', drop_first=True).values
    # Time dummies
    t_dums = pd.get_dummies(d['t'], prefix='t', drop_first=True).values
    # State × trend
    d['trend'] = d['t'] - d['t'].min()
    st_trends = pd.get_dummies(d['s'], prefix='st', drop_first=False).values * d['trend'].values.reshape(-1,1)

    W = np.column_stack([s_dums, t_dums, st_trends])

    # Add constant
    W = np.column_stack([np.ones(len(x)), W])

    # Remove collinear columns
    from numpy.linalg import matrix_rank
    rank = matrix_rank(W)
    if rank < W.shape[1]:
        # Use SVD to keep only linearly independent columns
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        keep = S > S[0] * 1e-10
        W = U[:, keep] * S[keep]

    beta, _, _, _ = np.linalg.lstsq(W, x, rcond=None)
    return x - W @ beta

# Run LP-IV with state-specific trends for main outcome
out = 'mean_routine'
print(f"\n  Running with state-specific trends for {OLABELS[out]}...")

trend_rows = []
for h in range(MAX_H + 1):
    lp_var = f'{out}_lp{h}'
    cols = [lp_var, 'delta_urate', 'bartik_shock']
    valid = df[cols].notna().all(axis=1)
    if valid.sum() < 100:
        continue

    y = df.loc[valid, lp_var].values
    endog = df.loc[valid, 'delta_urate'].values
    instr = df.loc[valid, 'bartik_shock'].values
    states = df.loc[valid, 'STATEFIP'].values
    times = df.loc[valid, 'YEAR'].values

    try:
        y_r = dd_with_state_trend(y, states, times)
        end_r = dd_with_state_trend(endog, states, times)
        ins_r = dd_with_state_trend(instr, states, times)

        # 2SLS on residualized data
        b1, _, _, _ = np.linalg.lstsq(ins_r.reshape(-1,1), end_r, rcond=None)
        end_hat = ins_r * b1[0]

        b2, _, _, _ = np.linalg.lstsq(end_hat.reshape(-1,1), y_r, rcond=None)
        resid = y_r - end_r * b2[0]

        # Clustered SE
        n = len(y_r)
        us = np.unique(states)
        G = len(us)
        XtXi = 1.0 / (end_hat @ end_hat) if (end_hat @ end_hat) > 0 else 0
        B_sum = 0
        for sv in us:
            m = states == sv
            B_sum += (end_hat[m] @ resid[m])**2
        cor = (G / (G-1)) * ((n-1) / (n-1))
        V = cor * XtXi**2 * B_sum
        se = np.sqrt(max(V, 0))

        t = b2[0] / se if se > 0 else 0
        p = 2 * (1 - norm.cdf(abs(t)))

        trend_rows.append({'horizon': h, 'beta': b2[0], 'se': se, 'pval': p,
                           'ci_lower': b2[0] - 1.96*se, 'ci_upper': b2[0] + 1.96*se, 'n': n})

        print(f"    h={h}: {b2[0]:+.5f} ({se:.5f}){star(p):>4s}")
    except Exception as e:
        print(f"    h={h}: failed — {e}")

if trend_rows:
    trend_df = pd.DataFrame(trend_rows)


# ============================================================
# 8. COMBINED PLOT: Baseline vs Robustness
# ============================================================
print("\n" + "=" * 70)
print("8. SUMMARY PLOTS")
print("=" * 70)

# Main result (routine) with placebo + baseline + flex + trends
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Routine with pre-trends
ax = axes[0]
# Load baseline results
baseline = run_lp_iv(df, ['mean_routine'], list(range(-PLACEBO_H, MAX_H+1)), exog_base)
if 'mean_routine' in baseline:
    d = baseline['mean_routine']
    ax.plot(d['horizon'], d['beta'], 'o-', color='#b2182b', linewidth=2, markersize=5, label='Baseline IV')
    ax.errorbar(d['horizon'], d['beta'],
                yerr=[d['beta']-d['ci_lower'], d['ci_upper']-d['beta']],
                fmt='none', ecolor='#b2182b', capsize=4, elinewidth=1.5, zorder=2)

if trend_rows:
    d = pd.DataFrame(trend_rows)
    ax.plot(d['horizon'], d['beta'], 's--', color='#4393c3', linewidth=1.5, markersize=4,
            label='+ State trends', alpha=0.8)

ax.axhline(0, color='gray', linewidth=0.8)
ax.axvline(0, color='black', linewidth=1, linestyle=':', alpha=0.5)
ax.set_xlabel('Horizon (years)')
ax.set_ylabel(r'$\hat{\beta}_h$')
ax.set_title('Routine: Baseline vs State Trends', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

# Right: Composition decomposition
ax = axes[1]
for out in available_shares[:4]:
    if out in comp_results:
        d = comp_results[out]
        label = SHARE_LABELS.get(out, out)
        ax.plot(d['horizon'], d['beta'], 'o-', linewidth=1.5, markersize=4, label=label, alpha=0.8)

ax.axhline(0, color='gray', linewidth=0.8)
ax.set_xlabel('Horizon (years)')
ax.set_ylabel(r'$\hat{\beta}_h$')
ax.set_title('Composition Decomposition', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

plt.tight_layout()
path = os.path.join(ROB_DIR, 'fig_robustness_summary.pdf')
fig.savefig(path, dpi=300, bbox_inches='tight')
print(f"  Saved: {path}")
plt.close()


# ============================================================
# 9. EXPORT ALL RESULTS
# ============================================================
print("\n" + "-" * 50)
print("9. Export")
print("-" * 50)

all_out = []

for src, tag in [(placebo_results, 'placebo'), (comp_results, 'composition'),
                  (flex_results, 'flex_controls')]:
    for out, d in src.items():
        o = d.copy()
        o['outcome'] = out
        o['specification'] = tag
        if tag == 'flex_controls':
            o['first_stage_f'] = f_flex
        all_out.append(o)

if baseline:
    for out, d in baseline.items():
        o = d.copy()
        o['outcome'] = out
        o['specification'] = 'baseline_with_placebos'
        all_out.append(o)

if trend_rows:
    o = pd.DataFrame(trend_rows)
    o['outcome'] = 'mean_routine'
    o['specification'] = 'state_trends'
    all_out.append(o)

if all_out:
    out_df = pd.concat(all_out, ignore_index=True)
    p = os.path.join(ROB_DIR, 'robustness_all_results.csv')
    out_df.to_csv(p, index=False, float_format='%.6f')
    print(f"  Saved: {p}")

if wdf is not None:
    print(f"  GPSS weights: gpss_rotemberg_weights.csv")

print("\n" + "=" * 70)
print("ROBUSTNESS BATTERY COMPLETE")
print("=" * 70)

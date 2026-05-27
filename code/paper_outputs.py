"""
Paper Output Generator — Macro Task Displacement LP-IV
=======================================================
Generates all tables, figures, and inline statistics for the paper.
All numbers are computed from data — nothing hardcoded.

Outputs to: D:\Data\MacroDisplacement\paper\
  - tables/    (LaTeX .tex files)
  - figures/   (PDF figures)
  - stats.json (inline statistics for \input in LaTeX)
  - stats.tex  (LaTeX \newcommand definitions for all stats)

Author: Ethan Johnston
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from linearmodels.panel import PanelOLS
from scipy.stats import norm, f as f_dist
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PATHS
# ============================================================
from paths import DATA_DIR, PAPER_DIR, TABLES_DIR as TABLE_DIR, FIGURES_DIR as FIG_DIR
for d in [PAPER_DIR, TABLE_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# Stats dictionary — every number used in the paper
S = {}

# ============================================================
# HELPERS
# ============================================================
OUTCOMES = ['mean_rti', 'mean_abstract', 'mean_routine', 'mean_manual']
OLABELS = {'mean_rti': 'RTI', 'mean_abstract': 'Abstract',
           'mean_routine': 'Routine', 'mean_manual': 'Manual'}
SHARE_OUTCOMES = ['share_routine', 'share_NRC', 'share_NRM', 'share_RC', 'share_RM']
SHARE_LABELS = {'share_routine': 'Routine (RC+RM)', 'share_NRC': 'Non-Routine Cognitive',
                'share_NRM': 'Non-Routine Manual', 'share_RC': 'Routine Cognitive',
                'share_RM': 'Routine Manual'}
MAX_H = 5
PLACEBO_H = 3

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

def run_ols(y, x, exog_list, states, times):
    y_dd = dd(y, states, times)
    x_dd = dd(x, states, times)
    exog_dd = [dd(e, states, times) for e in exog_list]
    X = np.column_stack([x_dd] + [e.reshape(-1,1) for e in exog_dd]) if exog_dd else x_dd.reshape(-1,1)
    b, _, _, _ = np.linalg.lstsq(X, y_dd, rcond=None)
    resid = y_dd - X @ b
    n, k = len(y_dd), X.shape[1]
    us = np.unique(states)
    G = len(us)
    XtXi = np.linalg.pinv(X.T @ X)
    B = np.zeros((k, k))
    for sv in us:
        m = states == sv
        Xe = X[m].T @ resid[m]
        B += np.outer(Xe, Xe)
    cor = (G/(G-1)) * ((n-1)/(n-k))
    V = cor * XtXi @ B @ XtXi
    se = np.sqrt(np.maximum(np.diag(V), 0))
    return b[0], se[0], n

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
# LOAD DATA
# ============================================================
print("=" * 70)
print("PAPER OUTPUT GENERATOR")
print("=" * 70)

sm = pd.read_csv(os.path.join(DATA_DIR, "state_month_panel.csv"))
all_vars = OUTCOMES + [s for s in SHARE_OUTCOMES if s in sm.columns]
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

base_yr = df['YEAR'].min() + 1
# Base year for the initial routine-share trend control. This is the
# second panel year (the first year is lost to the delta_urate lag), NOT a
# hardcoded calendar year. Exported so the paper text never drifts from it.
S['base_year'] = int(base_yr)
base = df[df['YEAR'] == base_yr].groupby('STATEFIP')['share_routine'].mean()
df = df.merge(base.rename('rs_init'), on='STATEFIP', how='left')
df['time_trend'] = df['YEAR'] - df['YEAR'].min()
df['tech_baseline'] = df['rs_init'] * df['time_trend']
exog_base = ['tech_baseline']

# LP outcomes
for v in all_vars:
    if v in df.columns:
        df[f'{v}_L1'] = df.groupby('STATEFIP')[v].shift(1)
        for h in range(-PLACEBO_H, MAX_H + 1):
            df[f'{v}_F{h}'] = df.groupby('STATEFIP')[v].shift(-h)
            df[f'{v}_lp{h}'] = df[f'{v}_F{h}'] - df[f'{v}_L1']

# Panel stats
S['n_obs'] = len(df)
S['n_states'] = df['STATEFIP'].nunique()
S['year_min'] = int(df['YEAR'].min())
S['year_max'] = int(df['YEAR'].max())
S['n_years'] = S['year_max'] - S['year_min'] + 1

print(f"Panel: {S['n_obs']:,} obs, {S['n_states']} states, {S['year_min']}-{S['year_max']}")


# ============================================================
# TABLE 1: SUMMARY STATISTICS
# ============================================================
print("\n" + "-" * 50)
print("Table 1: Summary Statistics")
print("-" * 50)

summ_vars = {
    'mean_abstract': 'Mean Abstract Task Content',
    'mean_routine': 'Mean Routine Task Content',
    'mean_manual': 'Mean Manual Task Content',
    'mean_rti': 'Mean RTI',
    'share_routine': 'Routine Employment Share',
    'share_NRC': 'Non-Routine Cognitive Share',
    'share_NRM': 'Non-Routine Manual Share',
    'urate': 'Unemployment Rate (\\%)',
    'delta_urate': '$\\Delta$ Unemployment Rate',
    'bartik_shock': 'Bartik Predicted Shock',
}

lines = [
    r'\begin{table}[htbp]',
    r'\centering',
    r'\caption{Summary Statistics: State-Year Panel}',
    r'\label{tab:summary}',
    r'\begin{tabular}{lcccccc}',
    r'\hline\hline',
    r'Variable & Mean & SD & p10 & Median & p90 & N \\',
    r'\hline',
]

for var, label in summ_vars.items():
    if var not in df.columns:
        continue
    s = df[var].dropna()
    S[f'summ_{var}_mean'] = s.mean()
    S[f'summ_{var}_sd'] = s.std()
    S[f'summ_{var}_p10'] = s.quantile(0.1)
    S[f'summ_{var}_p50'] = s.median()
    S[f'summ_{var}_p90'] = s.quantile(0.9)
    S[f'summ_{var}_n'] = int(len(s))
    
    lines.append(
        f'{label} & {s.mean():.4f} & {s.std():.4f} & {s.quantile(0.1):.4f} & '
        f'{s.median():.4f} & {s.quantile(0.9):.4f} & {len(s):,} \\\\'
    )

lines += [r'\hline\hline', r'\end{tabular}',
          r'\begin{tablenotes}\small',
          f'\\item Panel of {{\\statsNStates}} states observed annually from '
          f'{{\\statsYearMin}} to {{\\statsYearMax}}.',
          r'\end{tablenotes}',
          r'\end{table}']

with open(os.path.join(TABLE_DIR, 'tab_summary.tex'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f"  Saved: tab_summary.tex")


# ============================================================
# TABLE 2: FIRST STAGE
# ============================================================
print("\n" + "-" * 50)
print("Table 2: First Stage")
print("-" * 50)

df_p = df.set_index(['STATEFIP', 'YEAR'])
y_fs = df_p['delta_urate']
x_fs = df_p[['bartik_shock', 'tech_baseline']]
valid_fs = y_fs.notna() & x_fs.notna().all(axis=1)

fs_mod = PanelOLS(y_fs[valid_fs], x_fs[valid_fs],
                  entity_effects=True, time_effects=True,
                  drop_absorbed=True, check_rank=False)
fs_res = fs_mod.fit(cov_type='clustered', cluster_entity=True)

pi = fs_res.params['bartik_shock']
se_pi = fs_res.std_errors['bartik_shock']
p_pi = fs_res.pvalues['bartik_shock']
f_stat = (pi / se_pi) ** 2
n_fs = int(valid_fs.sum())
r2w_fs = fs_res.rsquared_within

S['fs_pi'] = pi
S['fs_se'] = se_pi
S['fs_f'] = f_stat
S['fs_n'] = n_fs
S['fs_r2w'] = r2w_fs

print(f"  pi={pi:.4f} ({se_pi:.4f}), F={f_stat:.1f}, N={n_fs:,}, R2w={r2w_fs:.4f}")

lines = [
    r'\begin{table}[htbp]',
    r'\centering',
    r'\caption{First Stage: Bartik Instrument and State Unemployment}',
    r'\label{tab:firststage}',
    r'\begin{tabular}{lc}',
    r'\hline\hline',
    r'& $\Delta U_{st}$ \\',
    r'\hline',
    f'Bartik Shock & {pi:.4f}{star_tex(p_pi)} \\\\',
    f'& ({se_pi:.4f}) \\\\[6pt]',
    r'\hline',
    f'State FE & Yes \\\\',
    f'Year FE & Yes \\\\',
    f'Tech Control & Yes \\\\',
    f'Observations & {n_fs:,} \\\\',
    f'Within $R^2$ & {r2w_fs:.4f} \\\\',
    f'$F$-statistic & {f_stat:.1f} \\\\',
    r'\hline\hline',
    r'\end{tabular}',
    r'\begin{tablenotes}\small',
    r'\item Standard errors clustered by state in parentheses.',
    r'\item Tech control is initial ({\statsBaseYear}) routine employment share $\times$ linear trend.',
    r'\item $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.1$.',
    r'\end{tablenotes}',
    r'\end{table}',
]

with open(os.path.join(TABLE_DIR, 'tab_firststage.tex'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f"  Saved: tab_firststage.tex")


# ============================================================
# RUN ALL LP-IV SPECIFICATIONS
# ============================================================
print("\n" + "-" * 50)
print("Running all LP-IV specifications...")
print("-" * 50)

def run_lp_battery(df_in, outcome_list, horizons, exog_cols, run_ols_too=True):
    """Run LP-IV (and optionally OLS) for all outcomes and horizons."""
    iv_res, ols_res = {}, {}
    for out in outcome_list:
        iv_rows, ols_rows = [], []
        for h in horizons:
            lp_var = f'{out}_lp{h}'
            if lp_var not in df_in.columns: continue
            cols = [lp_var, 'delta_urate', 'bartik_shock'] + exog_cols
            valid = df_in[cols].notna().all(axis=1)
            if valid.sum() < 100: continue
            y = df_in.loc[valid, lp_var].values
            endog = df_in.loc[valid, 'delta_urate'].values
            instr = df_in.loc[valid, 'bartik_shock'].values
            states = df_in.loc[valid, 'STATEFIP'].values
            times = df_in.loc[valid, 'YEAR'].values
            exog_list = [df_in.loc[valid, c].values for c in exog_cols]
            try:
                b, se, n = run_2sls(y, endog, instr, exog_list, states, times)
                t = b/se if se > 0 else 0
                p = 2*(1-norm.cdf(abs(t)))
                iv_rows.append({'horizon': h, 'beta': b, 'se': se, 'pval': p, 'n': n,
                                'ci_lower': b-1.96*se, 'ci_upper': b+1.96*se})
            except: pass
            if run_ols_too:
                try:
                    b, se, n = run_ols(y, endog, exog_list, states, times)
                    t = b/se if se > 0 else 0
                    p = 2*(1-norm.cdf(abs(t)))
                    ols_rows.append({'horizon': h, 'beta': b, 'se': se, 'pval': p, 'n': n,
                                     'ci_lower': b-1.96*se, 'ci_upper': b+1.96*se})
                except: pass
        if iv_rows: iv_res[out] = pd.DataFrame(iv_rows)
        if ols_rows: ols_res[out] = pd.DataFrame(ols_rows)
    return iv_res, ols_res

# Main LP-IV
main_iv, main_ols = run_lp_battery(df, OUTCOMES, list(range(MAX_H+1)), exog_base)
print("  Main LP-IV: done")

# Placebos
# Note: we exclude h = -1 because the LP outcome is T_{t+h} - T_{t-1},
# so at h = -1 the regressand is identically zero by construction.
# Including it printed a row of zeros with zero SE, which is misleading.
plac_horizons = [h for h in range(-PLACEBO_H, 0) if h != -1]
plac_iv, _ = run_lp_battery(df, OUTCOMES, plac_horizons, exog_base, False)
print("  Placebos: done")

# Full (placebo + main) for routine — exclude h = -1 (mechanically zero)
full_iv_horizons = [h for h in range(-PLACEBO_H, MAX_H+1) if h != -1]
full_iv, _ = run_lp_battery(df, ['mean_routine'], full_iv_horizons, exog_base, False)
print("  Full routine path: done")

# Composition
comp_iv, _ = run_lp_battery(df, [s for s in SHARE_OUTCOMES if s in df.columns],
                             list(range(MAX_H+1)), exog_base, False)
print("  Composition: done")

# Flexible controls
years = sorted(df['YEAR'].unique())
omit_yr = years[0]
for yr in years:
    if yr != omit_yr:
        df[f'rs_yr_{yr}'] = df['rs_init'] * (df['YEAR'] == yr).astype(float)
flex_exog = [f'rs_yr_{yr}' for yr in years if yr != omit_yr and f'rs_yr_{yr}' in df.columns and df[f'rs_yr_{yr}'].std() > 0]
flex_iv, _ = run_lp_battery(df, OUTCOMES, list(range(MAX_H+1)), flex_exog, False)
print("  Flex controls: done")

# State trends: use residualization approach
def dd_with_trend(x, states, times):
    d = pd.DataFrame({'x': x, 's': states, 't': times})
    s_dums = pd.get_dummies(d['s'], prefix='s', drop_first=True).values
    t_dums = pd.get_dummies(d['t'], prefix='t', drop_first=True).values
    d['trend'] = d['t'] - d['t'].min()
    st = pd.get_dummies(d['s'], prefix='st', drop_first=False).values * d['trend'].values.reshape(-1,1)
    W = np.column_stack([np.ones(len(x)), s_dums, t_dums, st])
    U, Sv, Vt = np.linalg.svd(W, full_matrices=False)
    keep = Sv > Sv[0]*1e-10
    W = U[:, keep] * Sv[keep]
    b, _, _, _ = np.linalg.lstsq(W, x, rcond=None)
    return x - W @ b

trend_iv = {}
for out in ['mean_routine', 'mean_manual']:
    rows = []
    for h in range(MAX_H+1):
        lp_var = f'{out}_lp{h}'
        cols = [lp_var, 'delta_urate', 'bartik_shock']
        valid = df[cols].notna().all(axis=1)
        if valid.sum() < 100: continue
        y = df.loc[valid, lp_var].values
        endog = df.loc[valid, 'delta_urate'].values
        instr = df.loc[valid, 'bartik_shock'].values
        states = df.loc[valid, 'STATEFIP'].values
        times = df.loc[valid, 'YEAR'].values
        try:
            yr = dd_with_trend(y, states, times)
            er = dd_with_trend(endog, states, times)
            ir = dd_with_trend(instr, states, times)
            b1, _, _, _ = np.linalg.lstsq(ir.reshape(-1,1), er, rcond=None)
            eh = ir*b1[0]
            b2, _, _, _ = np.linalg.lstsq(eh.reshape(-1,1), yr, rcond=None)
            resid = yr - er*b2[0]
            n = len(yr); us = np.unique(states); G = len(us)
            XtXi = 1.0/(eh@eh) if (eh@eh) > 0 else 0
            Bs = sum((eh[states==sv]@resid[states==sv])**2 for sv in us)
            V = (G/(G-1))*((n-1)/(n-1))*XtXi**2*Bs
            se = np.sqrt(max(V, 0))
            t = b2[0]/se if se > 0 else 0
            p = 2*(1-norm.cdf(abs(t)))
            rows.append({'horizon': h, 'beta': b2[0], 'se': se, 'pval': p, 'n': n,
                         'ci_lower': b2[0]-1.96*se, 'ci_upper': b2[0]+1.96*se})
        except: pass
    if rows: trend_iv[out] = pd.DataFrame(rows)
print("  State trends: done")

# Sample splits
split_iv = {}
sample_specs = [
    ('pre2008',   df['YEAR'] <= 2007),
    ('post2008',  df['YEAR'] >= 2008),
    ('drop2020',  ~df['YEAR'].isin([2020, 2021])),  # drop COVID years
]
for name, mask in sample_specs:
    sub = df[mask].copy().sort_values(['STATEFIP', 'YEAR'])
    sub['delta_urate'] = sub.groupby('STATEFIP')['urate'].diff()
    for v in ['mean_routine']:
        sub[f'{v}_L1'] = sub.groupby('STATEFIP')[v].shift(1)
        for h in range(MAX_H+1):
            sub[f'{v}_F{h}'] = sub.groupby('STATEFIP')[v].shift(-h)
            sub[f'{v}_lp{h}'] = sub[f'{v}_F{h}'] - sub[f'{v}_L1']
    iv, _ = run_lp_battery(sub, ['mean_routine'], list(range(MAX_H+1)), exog_base, False)
    if 'mean_routine' in iv:
        split_iv[name] = iv['mean_routine']
print("  Sample splits: done")

# Anderson-Rubin
print("  Anderson-Rubin...", end='')
out = 'mean_routine'
lp_var = f'{out}_lp0'
cols = [lp_var, 'delta_urate', 'bartik_shock'] + exog_base
valid = df[cols].notna().all(axis=1)
y = df.loc[valid, lp_var].values
endog = df.loc[valid, 'delta_urate'].values
instr = df.loc[valid, 'bartik_shock'].values
states = df.loc[valid, 'STATEFIP'].values
times = df.loc[valid, 'YEAR'].values
exog_list = [df.loc[valid, c].values for c in exog_base]
y_dd = dd(y, states, times); end_dd = dd(endog, states, times)
ins_dd = dd(instr, states, times); exog_dd = [dd(x, states, times) for x in exog_list]

beta_grid = np.linspace(-0.03, 0.01, 400)
ar_pvals = []
for b0 in beta_grid:
    y_adj = y_dd - b0*end_dd
    Z = np.column_stack([ins_dd] + [e.reshape(-1,1) for e in exog_dd]) if exog_dd else ins_dd.reshape(-1,1)
    g, _, _, _ = np.linalg.lstsq(Z, y_adj, rcond=None)
    r = y_adj - Z@g; n=len(y_adj); k=Z.shape[1]; ss1=np.sum(r**2)
    if exog_dd:
        Z0 = np.column_stack([e.reshape(-1,1) for e in exog_dd])
        g0, _, _, _ = np.linalg.lstsq(Z0, y_adj, rcond=None)
        ss0 = np.sum((y_adj - Z0@g0)**2)
    else:
        ss0 = np.sum((y_adj - y_adj.mean())**2)
    f_ar = ((ss0-ss1)/1)/(ss1/(n-k))
    ar_pvals.append(1 - f_dist.cdf(f_ar, 1, n-k))
ar_pvals = np.array(ar_pvals)
in_cs = beta_grid[ar_pvals > 0.05]
S['ar_lower'] = float(in_cs.min()) if len(in_cs) > 0 else np.nan
S['ar_upper'] = float(in_cs.max()) if len(in_cs) > 0 else np.nan
S['ar_excludes_zero'] = bool(S['ar_upper'] < 0 or S['ar_lower'] > 0)
print(f" [{S['ar_lower']:.5f}, {S['ar_upper']:.5f}], excludes zero: {S['ar_excludes_zero']}")

# Store key coefficients in S
for out in OUTCOMES:
    if out in main_iv:
        d = main_iv[out]
        for h in range(MAX_H+1):
            row = d[d['horizon']==h]
            if len(row) > 0:
                r = row.iloc[0]
                tag = OLABELS[out].lower()
                S[f'iv_{tag}_h{h}_beta'] = r['beta']
                S[f'iv_{tag}_h{h}_se'] = r['se']
                S[f'iv_{tag}_h{h}_pval'] = r['pval']
    if out in main_ols:
        d = main_ols[out]
        for h in range(MAX_H+1):
            row = d[d['horizon']==h]
            if len(row) > 0:
                r = row.iloc[0]
                tag = OLABELS[out].lower()
                S[f'ols_{tag}_h{h}_beta'] = r['beta']
                S[f'ols_{tag}_h{h}_se'] = r['se']
                S[f'ols_{tag}_h{h}_pval'] = r['pval']

# Reversion ratio for routine
if 'mean_routine' in main_iv:
    d = main_iv['mean_routine']
    b0 = d.loc[d['horizon']==0, 'beta'].values
    b1 = d.loc[d['horizon']==1, 'beta'].values
    b5 = d.loc[d['horizon']==MAX_H, 'beta'].values
    if len(b0) > 0 and len(b1) > 0 and len(b5) > 0:
        peak = min(b0[0], b1[0])  # most negative
        S['routine_peak'] = peak
        S['routine_h5'] = b5[0]
        S['routine_reversion_pct'] = (1 - abs(b5[0])/abs(peak)) * 100 if abs(peak) > 0 else 0

# Composition (employment-share) coefficients for inline use in the text.
# Short tags so the macros read \statsIvRmHZeroBeta, \statsIvNrmHOneBeta, etc.
SHARE_TAGS = {'share_routine': 'rgroup', 'share_NRC': 'nrc',
              'share_NRM': 'nrm', 'share_RC': 'rc', 'share_RM': 'rm'}
for out, tag in SHARE_TAGS.items():
    if out in comp_iv:
        d = comp_iv[out]
        for h in range(MAX_H + 1):
            row = d[d['horizon'] == h]
            if len(row) > 0:
                r = row.iloc[0]
                S[f'iv_{tag}_h{h}_beta'] = r['beta']
                S[f'iv_{tag}_h{h}_se'] = r['se']
                S[f'iv_{tag}_h{h}_pval'] = r['pval']

# Inline magnitude statistics, all derived from numbers already in S so
# the paper never carries a hand-computed ratio.
S['gr_shock_pp'] = 5  # stated size of a Great-Recession-scale unemployment rise
if 'iv_routine_h0_beta' in S and S.get('ols_routine_h0_beta', 0) != 0:
    S['iv_ols_ratio_routine_h0'] = int(round(
        abs(S['iv_routine_h0_beta'] / S['ols_routine_h0_beta'])))
if 'iv_routine_h0_beta' in S and S.get('summ_mean_routine_sd', 0) > 0:
    _sd = S['summ_mean_routine_sd']
    S['routine_h0_pct_sd'] = int(round(100 * abs(S['iv_routine_h0_beta']) / _sd))
    S['routine_gr_scaled_pct_sd'] = int(round(
        100 * S['gr_shock_pp'] * abs(S['iv_routine_h0_beta']) / _sd))


# ============================================================
# TABLE 3: MAIN LP-IV RESULTS
# ============================================================
print("\n" + "-" * 50)
print("Table 3: Main LP-IV")
print("-" * 50)

lines = [
    r'\begin{table}[htbp]',
    r'\centering',
    r'\caption{Local Projection IV: Effect of Unemployment on Task Composition}',
    r'\label{tab:lpiv}',
    r'\small',
    r'\begin{tabular}{l' + 'cc' * len(OUTCOMES) + '}',
    r'\hline\hline',
    r'& ' + ' & '.join([f'\\multicolumn{{2}}{{c}}{{{OLABELS[o]}}}' for o in OUTCOMES]) + r' \\',
    r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}',
    r'Horizon $h$ & ' + ' & '.join(['IV & OLS'] * len(OUTCOMES)) + r' \\',
    r'\hline',
]

for h in range(MAX_H + 1):
    vals = []
    for out in OUTCOMES:
        for src in [main_iv, main_ols]:
            if out in src:
                row = src[out][src[out]['horizon']==h]
                if len(row) > 0:
                    r = row.iloc[0]
                    vals.append(f"{r['beta']:.4f}{star_tex(r['pval'])}")
                else:
                    vals.append('')
            else:
                vals.append('')
    lines.append(f'{h} & ' + ' & '.join(vals) + r' \\')
    
    # SE row
    ses = []
    for out in OUTCOMES:
        for src in [main_iv, main_ols]:
            if out in src:
                row = src[out][src[out]['horizon']==h]
                if len(row) > 0:
                    ses.append(f"({row.iloc[0]['se']:.4f})")
                else:
                    ses.append('')
            else:
                ses.append('')
    lines.append(' & ' + ' & '.join(ses) + r' \\[3pt]')

lines += [
    r'\hline',
    f'First-stage $F$ & \\multicolumn{{{2*len(OUTCOMES)}}}{{c}}{{{f_stat:.1f}}} \\\\',
    r'\hline\hline',
    r'\end{tabular}',
    r'\begin{tablenotes}\small',
    r'\item Each cell reports $\hat{\beta}_h$ from the LP regression '
    r'$T_{s,t+h} - T_{s,t-1} = \beta_h \Delta \hat{U}_{st} + \delta \cdot Tech_{st} + \alpha_s + \theta_t + \varepsilon_{s,t+h}$.',
    r'\item All specifications include state fixed effects, year fixed effects, '
    r'and the technology control $\text{Tech}_{st} = \bar{\omega}^{R}_{s,\statsBaseYear} \times t$ '
    r'(initial routine employment share interacted with a linear trend).',
    r'\item IV instruments $\Delta U_{st}$ with a Bartik shift-share predictor.',
    r'\item Standard errors clustered by state. $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.1$.',
    r'\end{tablenotes}',
    r'\end{table}',
]

with open(os.path.join(TABLE_DIR, 'tab_lpiv_main.tex'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f"  Saved: tab_lpiv_main.tex")


# ============================================================
# TABLE 4: COMPOSITION DECOMPOSITION
# ============================================================
print("\n" + "-" * 50)
print("Table 4: Composition")
print("-" * 50)

comp_outcomes = [s for s in SHARE_OUTCOMES if s in comp_iv]
if comp_outcomes:
    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{LP-IV: Composition Decomposition (Employment Shares)}',
        r'\label{tab:composition}',
        r'\small',
        r'\begin{tabular}{l' + 'c' * len(comp_outcomes) + '}',
        r'\hline\hline',
        r'Horizon & ' + ' & '.join([SHARE_LABELS.get(o, o) for o in comp_outcomes]) + r' \\',
        r'\hline',
    ]
    
    for h in range(MAX_H+1):
        vals = []
        for out in comp_outcomes:
            if out in comp_iv:
                row = comp_iv[out][comp_iv[out]['horizon']==h]
                if len(row) > 0:
                    r = row.iloc[0]
                    vals.append(f"{r['beta']:.4f}{star_tex(r['pval'])}")
                else:
                    vals.append('')
            else:
                vals.append('')
        lines.append(f'{h} & ' + ' & '.join(vals) + r' \\')
        ses = []
        for out in comp_outcomes:
            if out in comp_iv:
                row = comp_iv[out][comp_iv[out]['horizon']==h]
                if len(row) > 0:
                    ses.append(f"({row.iloc[0]['se']:.4f})")
                else:
                    ses.append('')
            else:
                ses.append('')
        lines.append(' & ' + ' & '.join(ses) + r' \\[3pt]')
    
    lines += [r'\hline\hline', r'\end{tabular}',
              r'\begin{tablenotes}\small',
              r'\item Same LP-IV specification as Table \ref{tab:lpiv}.',
              r'\end{tablenotes}',
              r'\end{table}']
    
    with open(os.path.join(TABLE_DIR, 'tab_composition.tex'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: tab_composition.tex")


# ============================================================
# TABLE 5: ROBUSTNESS
# ============================================================
print("\n" + "-" * 50)
print("Table 5: Robustness (Routine)")
print("-" * 50)

rob_specs = [
    ('Baseline', main_iv.get('mean_routine')),
    ('Flex. Tech Controls', flex_iv.get('mean_routine')),
    ('State Trends', trend_iv.get('mean_routine')),
    ('Pre-2008', split_iv.get('pre2008')),
    ('Post-2008', split_iv.get('post2008')),
    ('Drop 2020-21', split_iv.get('drop2020')),
]
rob_specs = [(n, d) for n, d in rob_specs if d is not None]

if rob_specs:
    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{Robustness: LP-IV for Routine Task Content}',
        r'\label{tab:robustness}',
        r'\small',
        r'\begin{tabular}{l' + 'c' * len(rob_specs) + '}',
        r'\hline\hline',
        r'Horizon & ' + ' & '.join([n for n, _ in rob_specs]) + r' \\',
        r'\hline',
    ]
    
    for h in range(MAX_H+1):
        vals, ses = [], []
        for _, d in rob_specs:
            row = d[d['horizon']==h]
            if len(row) > 0:
                r = row.iloc[0]
                vals.append(f"{r['beta']:.4f}{star_tex(r['pval'])}")
                ses.append(f"({r['se']:.4f})")
            else:
                vals.append(''); ses.append('')
        lines.append(f'{h} & ' + ' & '.join(vals) + r' \\')
        lines.append(' & ' + ' & '.join(ses) + r' \\[3pt]')
    
    lines += [r'\hline\hline', r'\end{tabular}',
              r'\begin{tablenotes}\small',
              r'\item All specifications instrument $\Delta U_{st}$ with Bartik shift-share.',
              r'\item $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.1$.',
              r'\end{tablenotes}',
              r'\end{table}']
    
    with open(os.path.join(TABLE_DIR, 'tab_robustness.tex'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: tab_robustness.tex")


# ============================================================
# TABLE 6: PRE-TREND PLACEBOS
# ============================================================
print("\n" + "-" * 50)
print("Table 6: Pre-Trend Placebos")
print("-" * 50)

if plac_iv:
    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{Pre-Trend Placebos: LP-IV at Negative Horizons}',
        r'\label{tab:placebos}',
        r'\begin{tabular}{l' + 'c' * len(OUTCOMES) + '}',
        r'\hline\hline',
        r'Horizon & ' + ' & '.join([OLABELS[o] for o in OUTCOMES]) + r' \\',
        r'\hline',
    ]
    
    for h in [hh for hh in range(-PLACEBO_H, 0) if hh != -1]:
        vals, ses = [], []
        for out in OUTCOMES:
            if out in plac_iv:
                row = plac_iv[out][plac_iv[out]['horizon']==h]
                if len(row) > 0:
                    r = row.iloc[0]
                    vals.append(f"{r['beta']:.4f}{star_tex(r['pval'])}")
                    ses.append(f"({r['se']:.4f})")
                else:
                    vals.append(''); ses.append('')
            else:
                vals.append(''); ses.append('')
        lines.append(f'{h} & ' + ' & '.join(vals) + r' \\')
        lines.append(' & ' + ' & '.join(ses) + r' \\[3pt]')
    
    lines += [r'\hline\hline', r'\end{tabular}',
              r'\begin{tablenotes}\small',
              r'\item Estimates of $\hat{\beta}_h$ for $h < 0$ (pre-treatment).',
              r'\item Insignificant coefficients support the exclusion restriction.',
              r'\end{tablenotes}',
              r'\end{table}']
    
    with open(os.path.join(TABLE_DIR, 'tab_placebos.tex'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: tab_placebos.tex")


# ============================================================
# TABLE 7: GPSS WEIGHTS
# ============================================================
print("\n" + "-" * 50)
print("Table 7: GPSS Weights")
print("-" * 50)

gpss_path = os.path.join(DATA_DIR, "results", "robustness", "gpss_rotemberg_weights.csv")
if os.path.exists(gpss_path):
    wdf = pd.read_csv(gpss_path)
    
    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{GPSS Rotemberg Weight Decomposition}',
        r'\label{tab:gpss}',
        r'\begin{tabular}{lccc}',
        r'\hline\hline',
        r'Industry & Avg. Emp. Share & SD(Nat. Growth) & Weight (\%) \\',
        r'\hline',
    ]
    
    for _, r in wdf.iterrows():
        lines.append(f"{r['industry']} & {r['avg_emp_share']:.3f} & "
                     f"{r['sd_national_growth']:.3f} & {r['rotemberg_pct']:.1f} \\\\")
    
    S['gpss_top_industry'] = wdf.iloc[0]['industry']
    S['gpss_top_pct'] = wdf.iloc[0]['rotemberg_pct']
    S['gpss_top3_pct'] = wdf.iloc[:3]['rotemberg_pct'].sum()
    
    lines += [r'\hline\hline', r'\end{tabular}',
              r'\begin{tablenotes}\small',
              r'\item Approximate Rotemberg weights following Goldsmith-Pinkham, Sorkin, and Swift (2020).',
              r'\end{tablenotes}',
              r'\end{table}']
    
    with open(os.path.join(TABLE_DIR, 'tab_gpss.tex'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: tab_gpss.tex")


# ============================================================
# FIGURES
# ============================================================
print("\n" + "-" * 50)
print("Figures")
print("-" * 50)

plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})

# --- Figure 1: National time series (from national panel) ---
nat = pd.read_csv(os.path.join(DATA_DIR, "national_month_panel.csv"))
nat['date'] = pd.to_datetime(nat['date'])

RECESSIONS_SHADE = [('1980-01', '1980-07'), ('1981-07', '1982-11'),
                     ('1990-07', '1991-03'), ('2001-03', '2001-11'),
                     ('2007-12', '2009-06'), ('2020-02', '2020-04')]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for i, out in enumerate(OUTCOMES):
    ax = axes.flatten()[i]
    if out in nat.columns:
        ax.plot(nat['date'], nat[out], color='#2166ac', linewidth=1.2)
    for s, e in RECESSIONS_SHADE:
        ax.axvspan(pd.to_datetime(s), pd.to_datetime(e), alpha=0.12, color='gray')
    ax.set_title(OLABELS[out], fontweight='bold')
    ax.set_ylabel('Weighted Mean')
    ax.grid(True, alpha=0.2)
fig.suptitle('Aggregate Task Composition of Employed Workers, 1976-2025',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig_timeseries.pdf'), dpi=300, bbox_inches='tight')
print(f"  Saved: fig_timeseries.pdf")
plt.close()

# --- Figure 2: Main LP-IV impulse response ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for i, out in enumerate(OUTCOMES):
    ax = axes.flatten()[i]
    if out in main_iv:
        d = main_iv[out]
        ax.plot(d['horizon'], d['beta'], 'o-', color='#b2182b', linewidth=2.5, markersize=5, label='IV')
        ax.errorbar(d['horizon'], d['beta'],
                    yerr=[d['beta']-d['ci_lower'], d['ci_upper']-d['beta']],
                    fmt='none', ecolor='#b2182b', capsize=4, elinewidth=1.5, zorder=2)
    if out in main_ols:
        d = main_ols[out]
        ax.plot(d['horizon'], d['beta'], 's--', color='#4393c3', linewidth=1.5, markersize=4, label='OLS', alpha=0.7)
        ax.errorbar(d['horizon'], d['beta'],
                    yerr=[d['beta']-d['ci_lower'], d['ci_upper']-d['beta']],
                    fmt='none', ecolor='#4393c3', capsize=3, elinewidth=1.2, alpha=0.7, zorder=2)
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.set_xlabel('Horizon (years)'); ax.set_ylabel(r'$\hat{\beta}_h$')
    ax.set_title(OLABELS[out], fontweight='bold'); ax.set_xticks(range(MAX_H+1))
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
fig.suptitle(r'LP-IV: Effect of $\Delta U$ on Task Composition', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig_lpiv_main.pdf'), dpi=300, bbox_inches='tight')
print(f"  Saved: fig_lpiv_main.pdf")
plt.close()

# --- Figure 3: Routine with pre-trends ---
fig, ax = plt.subplots(figsize=(10, 6))
if 'mean_routine' in full_iv:
    d = full_iv['mean_routine']
    pre = d[d['horizon'] < 0]
    post = d[d['horizon'] >= 0]
    ax.plot(pre['horizon'], pre['beta'], 'o--', color='#4393c3', linewidth=1.5, markersize=5, label='Pre-trend (placebo)')
    ax.errorbar(pre['horizon'], pre['beta'],
                yerr=[pre['beta']-pre['ci_lower'], pre['ci_upper']-pre['beta']],
                fmt='none', ecolor='#4393c3', capsize=3, elinewidth=1.2, zorder=2)
    ax.plot(post['horizon'], post['beta'], 'o-', color='#b2182b', linewidth=2.5, markersize=6, label='Post-shock')
    ax.errorbar(post['horizon'], post['beta'],
                yerr=[post['beta']-post['ci_lower'], post['ci_upper']-post['beta']],
                fmt='none', ecolor='#b2182b', capsize=4, elinewidth=1.5, zorder=2)
    # Add reference at h=-1
    ax.plot(-1, 0, 'D', color='black', markersize=8, zorder=5, label='Reference (h=-1)')
ax.axhline(0, color='gray', linewidth=0.8)
ax.axvline(0, color='black', linewidth=1, linestyle=':', alpha=0.5)
ax.set_xlabel('Horizon (years)', fontsize=12); ax.set_ylabel(r'$\hat{\beta}_h$', fontsize=12)
ax.set_title('Routine Task Content: LP-IV with Pre-Trend Placebos', fontsize=14, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.2)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig_routine_pretrend.pdf'), dpi=300, bbox_inches='tight')
print(f"  Saved: fig_routine_pretrend.pdf")
plt.close()

# --- Figure 4: Composition decomposition ---
fig, ax = plt.subplots(figsize=(10, 6))
comp_colors = {'share_routine': '#b2182b', 'share_NRC': '#2166ac', 'share_NRM': '#1b7837',
               'share_RC': '#d6604d', 'share_RM': '#762a83'}
for out in comp_outcomes:
    if out in comp_iv:
        d = comp_iv[out]
        ax.plot(d['horizon'], d['beta'], 'o-', color=comp_colors.get(out, 'gray'),
                linewidth=2, markersize=5, label=SHARE_LABELS.get(out, out))
ax.axhline(0, color='gray', linewidth=0.8)
ax.set_xlabel('Horizon (years)', fontsize=12); ax.set_ylabel(r'$\hat{\beta}_h$', fontsize=12)
ax.set_title('Composition Decomposition: Employment Shares', fontsize=14, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.2); ax.set_xticks(range(MAX_H+1))
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig_composition.pdf'), dpi=300, bbox_inches='tight')
print(f"  Saved: fig_composition.pdf")
plt.close()

# --- Figure 5: Robustness comparison (routine) ---
fig, ax = plt.subplots(figsize=(10, 6))
rob_plot = [
    ('Baseline', main_iv.get('mean_routine'), '#b2182b', 'o-', 2.5),
    ('Flex Controls', flex_iv.get('mean_routine'), '#4393c3', 's--', 1.5),
    ('State Trends', trend_iv.get('mean_routine'), '#1b7837', '^:', 1.5),
]
for name, d, col, style, lw in rob_plot:
    if d is not None:
        ax.plot(d['horizon'], d['beta'], style, color=col, linewidth=lw, markersize=5, label=name)
        ax.errorbar(d['horizon'], d['beta'],
                    yerr=[d['beta']-d['ci_lower'], d['ci_upper']-d['beta']],
                    fmt='none', ecolor=col, capsize=3, elinewidth=1.2, zorder=2)
ax.axhline(0, color='gray', linewidth=0.8)
ax.set_xlabel('Horizon (years)', fontsize=12); ax.set_ylabel(r'$\hat{\beta}_h$', fontsize=12)
ax.set_title('Routine: Robustness Across Specifications', fontsize=14, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.2); ax.set_xticks(range(MAX_H+1))
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig_robustness.pdf'), dpi=300, bbox_inches='tight')
print(f"  Saved: fig_robustness.pdf")
plt.close()


# ============================================================
# STATS.TEX — \newcommand for every stat
# ============================================================
print("\n" + "-" * 50)
print("Generating stats.tex")
print("-" * 50)

def safe_cmd(key):
    """Convert key to valid LaTeX command name. Digits spelled out since LaTeX commands can't contain numbers."""
    digit_words = {'0': 'Zero', '1': 'One', '2': 'Two', '3': 'Three', '4': 'Four',
                   '5': 'Five', '6': 'Six', '7': 'Seven', '8': 'Eight', '9': 'Nine'}
    parts = key.split('_')
    name = 'stats' + ''.join(p.capitalize() for p in parts)
    # Replace digits with words
    result = ''
    for c in name:
        if c.isdigit():
            result += digit_words[c]
        elif c.isalpha():
            result += c
    return result

stat_lines = ['% Auto-generated statistics — do not edit manually',
              f'% Generated from paper_outputs.py',
              '']

for key, val in sorted(S.items()):
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
        # Year-like integers (e.g. 1978, 2025) should print without a
        # thousands separator; counts (e.g. 2,448) should keep one.
        if 'year' in key.lower() and 1800 < val < 2200:
            formatted = f'{val}'
        else:
            formatted = f'{val:,}'
    elif isinstance(val, str):
        formatted = val
    else:
        formatted = str(val)
    
    stat_lines.append(f'\\newcommand{{\\{cmd}}}{{{formatted}}}')

# Also save as JSON
with open(os.path.join(PAPER_DIR, 'stats.json'), 'w') as f:
    json.dump({k: float(v) if isinstance(v, (np.floating, float)) else
               int(v) if isinstance(v, (np.integer, int)) else
               str(v) for k, v in S.items()}, f, indent=2)

with open(os.path.join(PAPER_DIR, 'stats.tex'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(stat_lines))

print(f"  Saved: stats.tex ({len(S)} commands)")
print(f"  Saved: stats.json")

# Print a sample
print(f"\n  Sample stats:")
for key in ['n_obs', 'n_states', 'fs_f', 'iv_routine_h0_beta', 'iv_routine_h0_pval',
            'routine_reversion_pct', 'ar_lower', 'ar_upper']:
    if key in S:
        print(f"    \\{safe_cmd(key)} = {S[key]}")


print("\n" + "=" * 70)
print("PAPER OUTPUTS COMPLETE")
print(f"Tables: {TABLE_DIR}")
print(f"Figures: {FIG_DIR}")
print(f"Stats: {os.path.join(PAPER_DIR, 'stats.tex')}")
print("=" * 70)
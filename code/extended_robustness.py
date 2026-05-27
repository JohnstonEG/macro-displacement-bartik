"""
Extended Robustness
====================
1. Task composition WITHIN Bartik vs non-Bartik industries
   (requires reading CPS microdata to compute task scores by industry group)
2. Multiple trend controls applied to all four task outcomes

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
# HELPERS (same as before)
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

def run_lp(df_in, outcomes, horizons, exog_cols):
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

def first_stage_f(df_in, exog_cols):
    df_p = df_in.set_index(['STATEFIP', 'YEAR'])
    y = df_p['delta_urate']
    x_cols = ['bartik_shock'] + exog_cols
    x = df_p[x_cols]
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
            return (pi/se)**2, pi, int(valid.sum())
    except:
        pass
    return 0, 0, 0


# ============================================================
# PART 1: TASK COMPOSITION WITHIN INDUSTRY GROUPS
#         (Requires CPS microdata processing)
# ============================================================
print("=" * 70)
print("PART 1: Task Composition Within Bartik vs Non-Bartik Industries")
print("=" * 70)

# --- Step 1: Identify Bartik vs non-Bartik industries ---
ces = pd.read_csv(os.path.join(DATA_DIR, "ces_national_industry.csv"))
ces_a = ces.groupby(['industry', 'year'])['employment'].mean().reset_index()
ces_a = ces_a.sort_values(['industry', 'year'])
ces_a['emp_lag'] = ces_a.groupby('industry')['employment'].shift(1)
ces_a['growth'] = (ces_a['employment'] - ces_a['emp_lag']) / ces_a['emp_lag']
growth_var = ces_a.groupby('industry')['growth'].var().reset_index()
growth_var.columns = ['industry', 'var_growth']
median_var = growth_var['var_growth'].median()

high_var = set(growth_var[growth_var['var_growth'] >= median_var]['industry'])
low_var = set(growth_var[growth_var['var_growth'] < median_var]['industry'])
print(f"  High-variance industries: {len(high_var)}")
print(f"  Low-variance industries: {len(low_var)}")

# --- Step 2: IND1990 -> supersector mapping (same as bartik_shares.py) ---
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
    if 940 <= code <= 960: return None
    return None

# --- Step 3: Read CPS microdata, compute task scores by industry group ---
print("\n  Reading CPS microdata (this will take several minutes)...")

# Load task scores crosswalk
occ_task = pd.read_csv(os.path.join(DATA_DIR, "occ2010_task_scores.csv"))
occ_task['OCC2010'] = occ_task['OCC2010'].astype(int)
print(f"  Task scores for {len(occ_task)} occupations")

# Read CPS in chunks — only need STATEFIP, YEAR, MONTH, WTFINL, EMPSTAT, OCC2010, IND1990
colspecs = [
    (0, 4),      # YEAR
    (9, 11),     # MONTH
    (48, 50),    # STATEFIP
    (52, 66),    # WTFINL
    (124, 126),  # EMPSTAT
    (131, 135),  # OCC2010
    (138, 141),  # IND1990
]
colnames = ['YEAR', 'MONTH', 'STATEFIP', 'WTFINL', 'EMPSTAT', 'OCC2010', 'IND1990']

CPS_DAT = os.path.join(DATA_DIR, "cps_00002.dat")
CHUNK = 1_000_000

# Accumulate state-year-industrygroup task measures
accum_bartik = []
accum_nonbartik = []

reader = pd.read_fwf(CPS_DAT, colspecs=colspecs, names=colnames,
                      chunksize=CHUNK, dtype=str)

total_read = 0
total_matched = 0

for i, chunk in enumerate(reader):
    total_read += len(chunk)
    
    # Convert types
    for col in ['YEAR', 'MONTH', 'STATEFIP', 'EMPSTAT', 'OCC2010', 'IND1990']:
        chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
    chunk['WTFINL'] = pd.to_numeric(chunk['WTFINL'], errors='coerce') / 10000
    
    # Filter: employed, valid occ and industry, positive weight
    mask = (
        chunk['EMPSTAT'].isin([10, 12]) &
        (chunk['OCC2010'] > 0) & (chunk['OCC2010'] < 9800) &
        (chunk['IND1990'] > 0) &
        (chunk['STATEFIP'] >= 1) & (chunk['STATEFIP'] <= 56) &
        (chunk['WTFINL'] > 0)
    )
    kept = chunk[mask].copy()
    
    # Map industry
    kept['supersector'] = kept['IND1990'].apply(ind1990_to_supersector)
    kept = kept[kept['supersector'].notna()]
    
    # Classify as Bartik or non-Bartik
    kept['is_bartik_ind'] = kept['supersector'].isin(high_var)
    
    # Merge task scores
    kept['OCC2010'] = kept['OCC2010'].astype(int)
    kept = kept.merge(occ_task, on='OCC2010', how='left')
    kept = kept[kept['abstract'].notna()]
    
    total_matched += len(kept)
    
    # Aggregate to state-year-group
    for is_bartik, acc in [(True, accum_bartik), (False, accum_nonbartik)]:
        sub = kept[kept['is_bartik_ind'] == is_bartik]
        if len(sub) == 0:
            continue
        
        agg = sub.groupby(['STATEFIP', 'YEAR']).apply(
            lambda g: pd.Series({
                'mean_abstract': np.average(g['abstract'], weights=g['WTFINL']),
                'mean_routine': np.average(g['routine'], weights=g['WTFINL']),
                'mean_manual': np.average(g['manual'], weights=g['WTFINL']),
                'mean_rti': np.average(g['rti'], weights=g['WTFINL']),
                'total_emp': g['WTFINL'].sum(),
                'n_obs': len(g),
            })
        ).reset_index()
        acc.append(agg)
    
    if (i + 1) % 10 == 0:
        print(f"    Chunk {i+1}: {total_read:,} read, {total_matched:,} matched")

print(f"  Total: {total_read:,} read, {total_matched:,} matched with task scores")

# Combine chunks
def combine_chunks(acc_list):
    """Combine chunk-level aggregates into state-year panel."""
    if not acc_list:
        return pd.DataFrame()
    combined = pd.concat(acc_list, ignore_index=True)
    # Re-aggregate: weighted mean across chunks within same state-year
    panel = combined.groupby(['STATEFIP', 'YEAR']).apply(
        lambda g: pd.Series({
            'mean_abstract': np.average(g['mean_abstract'], weights=g['total_emp']),
            'mean_routine': np.average(g['mean_routine'], weights=g['total_emp']),
            'mean_manual': np.average(g['mean_manual'], weights=g['total_emp']),
            'mean_rti': np.average(g['mean_rti'], weights=g['total_emp']),
            'total_emp': g['total_emp'].sum(),
            'n_obs': g['n_obs'].sum(),
        })
    ).reset_index()
    panel['STATEFIP'] = panel['STATEFIP'].astype(int)
    panel['YEAR'] = panel['YEAR'].astype(int)
    return panel

bartik_panel = combine_chunks(accum_bartik)
nonbartik_panel = combine_chunks(accum_nonbartik)

print(f"\n  Bartik-industry panel: {len(bartik_panel):,} state-year obs")
print(f"  Non-Bartik-industry panel: {len(nonbartik_panel):,} state-year obs")

# Save for reference
bartik_panel.to_csv(os.path.join(ROB_DIR, 'task_bartik_industries.csv'), index=False)
nonbartik_panel.to_csv(os.path.join(ROB_DIR, 'task_nonbartik_industries.csv'), index=False)

# --- Step 4: Merge with unemployment and Bartik, run LP-IV ---
laus = pd.read_csv(os.path.join(DATA_DIR, "laus_state_unemployment.csv"))
laus_a = laus.groupby(['STATEFIP', 'year']).agg(urate=('unemployment_rate', 'mean')).reset_index()
laus_a.rename(columns={'year': 'YEAR'}, inplace=True)
laus_a['STATEFIP'] = laus_a['STATEFIP'].astype(int)

bartik_inst = pd.read_csv(os.path.join(DATA_DIR, "bartik_instrument.csv"))
bartik_inst['STATEFIP'] = bartik_inst['STATEFIP'].astype(int)

# Also need the main panel for initial shares
sm = pd.read_csv(os.path.join(DATA_DIR, "state_month_panel.csv"))
sa_main = sm.groupby(['STATEFIP', 'YEAR']).apply(
    lambda g: pd.Series({'share_routine': wmean(g, 'share_routine') if 'share_routine' in g.columns else np.nan})
).reset_index()
sa_main['STATEFIP'] = sa_main['STATEFIP'].astype(int)
sa_main['YEAR'] = sa_main['YEAR'].astype(int)

ind_results = {}

for panel, group_label, prefix in [(bartik_panel, 'Bartik Industries', 'bi'),
                                    (nonbartik_panel, 'Non-Bartik Industries', 'nbi')]:
    print(f"\n  --- {group_label} ---")
    
    if len(panel) == 0:
        print(f"    No data — skipping")
        continue
    
    # Rename task columns with prefix
    renamed = panel.rename(columns={
        'mean_abstract': f'{prefix}_abstract',
        'mean_routine': f'{prefix}_routine',
        'mean_manual': f'{prefix}_manual',
        'mean_rti': f'{prefix}_rti',
    })
    
    # Merge with unemployment, Bartik, and initial shares
    mg = renamed.merge(laus_a, on=['STATEFIP', 'YEAR'], how='inner')
    mg = mg.merge(bartik_inst, on=['STATEFIP', 'YEAR'], how='inner')
    mg = mg.merge(sa_main[['STATEFIP', 'YEAR', 'share_routine']], 
                  on=['STATEFIP', 'YEAR'], how='left')
    mg = mg.sort_values(['STATEFIP', 'YEAR']).reset_index(drop=True)
    
    mg['delta_urate'] = mg.groupby('STATEFIP')['urate'].diff()
    
    # Tech control
    base_yr = mg['YEAR'].min() + 1
    base_sh = mg[mg['YEAR'] == base_yr].groupby('STATEFIP')['share_routine'].mean()
    mg = mg.merge(base_sh.rename('rs_init'), on='STATEFIP', how='left')
    mg['time_trend'] = mg['YEAR'] - mg['YEAR'].min()
    mg['tc_routine'] = mg['rs_init'] * mg['time_trend']
    
    # LP outcomes
    task_cols = [f'{prefix}_abstract', f'{prefix}_routine', f'{prefix}_manual', f'{prefix}_rti']
    for v in task_cols:
        mg[f'{v}_L1'] = mg.groupby('STATEFIP')[v].shift(1)
        for h in range(MAX_H + 1):
            mg[f'{v}_F{h}'] = mg.groupby('STATEFIP')[v].shift(-h)
            mg[f'{v}_lp{h}'] = mg[f'{v}_F{h}'] - mg[f'{v}_L1']
    
    # First stage
    f_val, pi_val, n_val = first_stage_f(mg, ['tc_routine'])
    print(f"    First stage: F={f_val:.1f}, N={n_val:,}")
    
    if f_val < 5:
        print(f"    Weak instrument — skipping")
        continue
    
    # LP-IV
    group_results = run_lp(mg, task_cols, list(range(MAX_H + 1)), ['tc_routine'])
    
    for tc in task_cols:
        clean_name = tc.replace(f'{prefix}_', '')
        if tc in group_results:
            ind_results[(group_label, clean_name)] = group_results[tc]
            print(f"    {OLABELS.get('mean_' + clean_name, clean_name)}:")
            for _, r in group_results[tc].iterrows():
                print(f"      h={int(r['horizon'])}: {r['beta']:+.5f} ({r['se']:.5f}){star(r['pval']):>4s}")
        else:
            print(f"    {OLABELS.get('mean_' + clean_name, clean_name)}: no results")


# ============================================================
# PART 2: ALL FOUR OUTCOMES × MULTIPLE TREND CONTROLS
# ============================================================
print("\n" + "=" * 70)
print("PART 2: Multiple Trend Controls — All Outcomes")
print("=" * 70)

# Build main panel (reuse from Part 1 merge logic)
sm = pd.read_csv(os.path.join(DATA_DIR, "state_month_panel.csv"))
all_vars = OUTCOMES + ['share_routine']
sa = sm.groupby(['STATEFIP', 'YEAR']).apply(
    lambda g: pd.Series({v: wmean(g, v) for v in all_vars if v in g.columns})
).reset_index()
sa['STATEFIP'] = sa['STATEFIP'].astype(int)
sa['YEAR'] = sa['YEAR'].astype(int)

df = sa.merge(laus_a, on=['STATEFIP', 'YEAR'], how='inner')
df = df.merge(bartik_inst, on=['STATEFIP', 'YEAR'], how='inner')
df = df.sort_values(['STATEFIP', 'YEAR']).reset_index(drop=True)
df['delta_urate'] = df.groupby('STATEFIP')['urate'].diff()

# Initial shares
base_yr = df['YEAR'].min() + 1
for var, label in [('share_routine', 'routine'), ('mean_abstract', 'abstract'),
                    ('mean_manual', 'manual')]:
    if var in df.columns:
        base_vals = df[df['YEAR'] == base_yr].groupby('STATEFIP')[var].mean()
        df = df.merge(base_vals.rename(f'init_{label}'), on='STATEFIP', how='left')

df['time_trend'] = df['YEAR'] - df['YEAR'].min()
df['tc_routine'] = df['init_routine'] * df['time_trend']
df['tc_abstract'] = df['init_abstract'] * df['time_trend']
df['tc_manual'] = df['init_manual'] * df['time_trend']

# LP outcomes
for v in OUTCOMES:
    df[f'{v}_L1'] = df.groupby('STATEFIP')[v].shift(1)
    for h in range(MAX_H + 1):
        df[f'{v}_F{h}'] = df.groupby('STATEFIP')[v].shift(-h)
        df[f'{v}_lp{h}'] = df[f'{v}_F{h}'] - df[f'{v}_L1']

# Specifications
spec_A = ['tc_routine']
spec_B = ['tc_routine', 'tc_abstract', 'tc_manual']

for spec_name, exog_cols in [('A: Routine trend only', spec_A),
                               ('B: All three task trends', spec_B)]:
    exog_valid = [c for c in exog_cols if c in df.columns and df[c].std() > 0]
    
    f_val, pi_val, n_val = first_stage_f(df, exog_valid)
    print(f"\n  {spec_name}: F={f_val:.1f}")
    
    if f_val < 5:
        print(f"    Weak instrument — skipping")
        continue
    
    results = run_lp(df, OUTCOMES, list(range(MAX_H + 1)), exog_valid)
    
    for out in OUTCOMES:
        if out in results:
            print(f"\n    {OLABELS[out]}:")
            for _, r in results[out].iterrows():
                print(f"      h={int(r['horizon'])}: {r['beta']:+.5f} ({r['se']:.5f}){star(r['pval']):>4s}")


# ============================================================
# PART 3: PLOTS
# ============================================================
print("\n" + "=" * 70)
print("PART 3: Plots")
print("=" * 70)

# Figure: Task composition within Bartik vs non-Bartik industries
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i, task in enumerate(['abstract', 'routine', 'manual', 'rti']):
    ax = axes.flatten()[i]
    
    for group, color, style in [('Bartik Industries', '#b2182b', 'o-'),
                                  ('Non-Bartik Industries', '#4393c3', 's--')]:
        key = (group, task)
        if key in ind_results:
            d = ind_results[key]
            ax.plot(d['horizon'], d['beta'], style, color=color,
                    linewidth=2, markersize=5, label=group, alpha=0.85)
            ax.errorbar(d['horizon'], d['beta'],
                        yerr=[d['beta']-d['ci_lower'], d['ci_upper']-d['beta']],
                        fmt='none', ecolor=color, capsize=3, elinewidth=1.2,
                        alpha=0.85, zorder=2)

    ax.axhline(0, color='gray', linewidth=0.8)
    ax.set_xlabel('Horizon (years)')
    ax.set_ylabel(r'$\hat{\beta}_h$')
    ax.set_title(OLABELS.get(f'mean_{task}', task), fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xticks(range(MAX_H + 1))

fig.suptitle('LP-IV: Task Composition Within Industry Groups\n'
             '(Bartik-driving vs non-Bartik industries)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
path = os.path.join(ROB_DIR, 'fig_task_by_industry_group.pdf')
fig.savefig(path, dpi=300, bbox_inches='tight')
print(f"  Saved: {path}")
plt.close()

# Figure: All outcomes × multiple trend controls
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i, out in enumerate(OUTCOMES):
    ax = axes.flatten()[i]
    
    for spec_name, exog_cols, color, style in [
        ('Routine trend only', spec_A, '#b2182b', 'o-'),
        ('All task trends', spec_B, '#4393c3', 's--')
    ]:
        exog_valid = [c for c in exog_cols if c in df.columns and df[c].std() > 0]
        res = run_lp(df, [out], list(range(MAX_H + 1)), exog_valid)
        if out in res:
            d = res[out]
            ax.plot(d['horizon'], d['beta'], style, color=color,
                    linewidth=2, markersize=5, label=spec_name, alpha=0.85)
            ax.errorbar(d['horizon'], d['beta'],
                        yerr=[d['beta']-d['ci_lower'], d['ci_upper']-d['beta']],
                        fmt='none', ecolor=color, capsize=3, elinewidth=1.2,
                        alpha=0.85, zorder=2)

    ax.axhline(0, color='gray', linewidth=0.8)
    ax.set_xlabel('Horizon (years)')
    ax.set_ylabel(r'$\hat{\beta}_h$')
    ax.set_title(OLABELS[out], fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xticks(range(MAX_H + 1))

fig.suptitle('LP-IV: All Outcomes with Multiple Trend Controls',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
path = os.path.join(ROB_DIR, 'fig_all_outcomes_trend_controls.pdf')
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

# Industry group results
for (group, task), d in ind_results.items():
    o = d.copy()
    o['specification'] = f'within_{group}'
    o['outcome'] = f'mean_{task}'
    all_rows.append(o)

# Multi-trend results for all outcomes
for spec_name, exog_cols in [('routine_trend', spec_A), ('all_trends', spec_B)]:
    exog_valid = [c for c in exog_cols if c in df.columns and df[c].std() > 0]
    res = run_lp(df, OUTCOMES, list(range(MAX_H + 1)), exog_valid)
    for out, d in res.items():
        o = d.copy()
        o['specification'] = spec_name
        o['outcome'] = out
        all_rows.append(o)

if all_rows:
    out_df = pd.concat(all_rows, ignore_index=True)
    p = os.path.join(ROB_DIR, 'extended_robustness_results.csv')
    out_df.to_csv(p, index=False, float_format='%.6f')
    print(f"  Saved: {p}")

print("\n" + "=" * 70)
print("EXTENDED ROBUSTNESS COMPLETE")
print("=" * 70)

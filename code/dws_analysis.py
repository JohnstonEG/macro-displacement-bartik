"""
DWS analysis: do routine-manual displaced workers exit RM at higher rates
when their state-year Bartik shock predicts higher unemployment?

Headline design (worker level, reduced form on Bartik):
    Pr(post_JS != RM | pre_JS = RM)  =  beta * Bartik_{s, displacement_year}
                                      + gamma * X_i + alpha_s + theta_t + eps

  alpha_s = state FE (state of displacement)
  theta_t = displacement-year FE
  X_i     = age, age^2, female, black, hispanic
  Cluster SE by state.

Three outcomes:
    1. rm_to_nonrm   = exits RM to any non-RM group     (broadest)
    2. rm_to_nrm     = exits RM to non-routine manual    (mirror of aggregate)
    3. rm_to_nrc     = exits RM to non-routine cognitive (upward mobility)

We report the reduced-form coefficient on the Bartik shock with cluster-robust
SE by state. This is the direct worker-level test of the mechanism that the
aggregate decomposition (Section 5.5) and within-occupation composition shifts
(Section 5.4) point to.

We do not run 2SLS at the worker level: the Bartik is the same state-year
variation as the aggregate identification, so the 2SLS LATE interpretation
would simply rescale by the first-stage coefficient. The reduced form is the
right object for this test.

Outputs:
  paper/tables/tab_dws.tex
  paper/figures/fig_dws_destinations.pdf (optional bar chart)
  paper/stats.tex (appended with DWS statistics)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

from paths import DATA_DIR
PANEL       = os.path.join(DATA_DIR, "dws_panel.parquet")
TAB_OUT     = os.path.join(DATA_DIR, "paper", "tables", "tab_dws.tex")
FIG_OUT     = os.path.join(DATA_DIR, "paper", "figures", "fig_dws_destinations.pdf")
STATS_OUT   = os.path.join(DATA_DIR, "paper", "stats_dws.tex")

# ----------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------
print("=" * 70)
print("DWS REDUCED-FORM ANALYSIS")
print("=" * 70)

df = pd.read_parquet(PANEL)
df['STATEFIP'] = df['STATEFIP'].astype(int)
df['displacement_year'] = df['displacement_year'].astype(int)

# Restrict to RM-displaced workers for the headline test
rm = df[df['js_pre'] == 'RM'].copy()
print(f"\nRM-displaced workers (currently employed): {len(rm):,}")
print(f"  DWS waves: {sorted(rm['YEAR'].unique().astype(int))}")
print(f"  Displacement-year range: {rm['displacement_year'].min()}–{rm['displacement_year'].max()}")
print(f"  States: {rm['STATEFIP'].nunique()}")

# ----------------------------------------------------------------------
# Two-way FE absorption (state FE + displacement-year FE), then OLS with
# clustered SE. Match the convention used in lp_iv.py / paper_outputs.py.
# ----------------------------------------------------------------------
def dd(x, s, t):
    """Double-demean by state and year (iterative for unbalanced panels)."""
    d = pd.DataFrame({'x': pd.to_numeric(x, errors='coerce'), 's': s, 't': t})
    for _ in range(20):
        d['x'] = d['x'] - d.groupby('s')['x'].transform('mean')
        d['x'] = d['x'] - d.groupby('t')['x'].transform('mean')
    return d['x'].values

def run_rf(df_in, outcome, treat='bartik_shock',
           controls=('AGE','age_sq','female','black','hispanic')):
    """Reduced-form OLS of `outcome` on `treat` with state & year FE,
       worker controls, cluster-robust SE by state."""
    sub = df_in[[outcome, treat, 'STATEFIP', 'displacement_year', *controls,
                 'DWSUPPWT']].dropna().copy()
    states = sub['STATEFIP'].values
    times  = sub['displacement_year'].values

    y_dd      = dd(sub[outcome].values, states, times)
    treat_dd  = dd(sub[treat].values,   states, times)
    ctrls_dd  = [dd(sub[c].values, states, times) for c in controls]

    X = np.column_stack([treat_dd] + [c.reshape(-1, 1) for c in ctrls_dd])
    beta, _, _, _ = np.linalg.lstsq(X, y_dd, rcond=None)
    resid = y_dd - X @ beta

    n, k = len(y_dd), X.shape[1]
    us = np.unique(states)
    G  = len(us)
    XtX_inv = np.linalg.pinv(X.T @ X)
    B = np.zeros((k, k))
    for sv in us:
        m  = states == sv
        Xe = X[m].T @ resid[m]
        B += np.outer(Xe, Xe)
    cor = (G / (G - 1)) * ((n - 1) / (n - k))
    V   = cor * XtX_inv @ B @ XtX_inv
    se  = np.sqrt(np.maximum(np.diag(V), 0))

    t_stat = beta[0] / se[0] if se[0] > 0 else 0
    p_val  = 2 * (1 - norm.cdf(abs(t_stat)))
    return {
        'outcome' : outcome,
        'beta'    : float(beta[0]),
        'se'      : float(se[0]),
        'tstat'   : float(t_stat),
        'pval'    : float(p_val),
        'n'       : int(n),
        'ymean'   : float(sub[outcome].mean()),
    }

# ----------------------------------------------------------------------
# Estimate
# ----------------------------------------------------------------------
results = []
for outcome in ['rm_to_nonrm', 'rm_to_nrm', 'rm_to_nrc']:
    r = run_rf(rm, outcome)
    results.append(r)
    stars = '***' if r['pval'] < 0.01 else ('**' if r['pval'] < 0.05 else ('*' if r['pval'] < 0.10 else ''))
    print(f"\n  {outcome:>12s}: beta={r['beta']:+.5f} ({r['se']:.5f}){stars}  "
          f"t={r['tstat']:+.2f}  p={r['pval']:.3f}  "
          f"N={r['n']:,}  ymean={r['ymean']:.3f}")

# ----------------------------------------------------------------------
# LaTeX table
# ----------------------------------------------------------------------
def fmt(b, se, p):
    stars = '$^{***}$' if p < 0.01 else ('$^{**}$' if p < 0.05 else ('$^{*}$' if p < 0.10 else ''))
    return f"{b:.4f}{stars}"

r_nonrm, r_nrm, r_nrc = results
os.makedirs(os.path.dirname(TAB_OUT), exist_ok=True)

table_tex = rf"""\begin{{table}}[htbp]
\centering
\caption{{Worker-Level Test: Displaced Routine-Manual Workers' Re-employment Destinations}}
\label{{tab:dws}}
\small
\begin{{threeparttable}}
\begin{{tabular}}{{lccc}}
\hline\hline
                                  & RM $\to$ any non-RM    & RM $\to$ NRM (service)  & RM $\to$ NRC (prof./tech.) \\
                                  & (1)                    & (2)                     & (3)                        \\
\hline
Bartik shock $B_{{s,t}}$            & {fmt(r_nonrm['beta'], r_nonrm['se'], r_nonrm['pval'])} & {fmt(r_nrm['beta'], r_nrm['se'], r_nrm['pval'])} & {fmt(r_nrc['beta'], r_nrc['se'], r_nrc['pval'])} \\
                                  & ({r_nonrm['se']:.4f})  & ({r_nrm['se']:.4f})     & ({r_nrc['se']:.4f})        \\[3pt]
\hline
Mean of dependent variable        & {r_nonrm['ymean']:.4f} & {r_nrm['ymean']:.4f} & {r_nrc['ymean']:.4f} \\
Observations                      & {r_nonrm['n']:,} & {r_nrm['n']:,} & {r_nrc['n']:,} \\
\hline\hline
\end{{tabular}}
\begin{{tablenotes}}\footnotesize
\item Reduced-form OLS of the post-displacement destination indicator on the
Bartik shock for the worker's state in the year of displacement. Sample:
routine-manual workers displaced involuntarily (plant closure, insufficient
work, or position abolished) from a full-time private-sector job, who are
currently employed at the time of the DWS interview. The Bartik shock is the
same state-year shift-share predictor used in the aggregate LP-IV
(Section~\ref{{sec:bartik}}).
\item All specifications include state fixed effects (state of displacement),
displacement-year fixed effects, and worker controls (age, age$^2$, sex,
Black, Hispanic). Fixed effects absorbed via iterative demeaning. Standard
errors clustered by state.
\item $^{{***}} p < 0.01$, $^{{**}} p < 0.05$, $^{{*}} p < 0.10$.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}
"""

with open(TAB_OUT, 'w', encoding='utf-8') as f:
    f.write(table_tex)
print(f"\nSaved: {TAB_OUT}")

# ----------------------------------------------------------------------
# Stats for inline use in main.tex
# ----------------------------------------------------------------------
# Post-displacement destination distribution among RM-displaced workers.
# These feed the inline percentages in Section 5.5 so no destination share
# is hand-typed into the manuscript.
dest = rm['js_post'].value_counts(normalize=True).reindex(
    ['RM', 'NRM', 'NRC', 'RC']).fillna(0)
pct_rm   = dest['RM']  * 100          # share returning to routine-manual
pct_exit = (1 - dest['RM']) * 100     # share NOT returning to RM
pct_nrm  = dest['NRM'] * 100
pct_nrc  = dest['NRC'] * 100
pct_rc   = dest['RC']  * 100
wave_min = int(rm['YEAR'].min())
wave_max = int(rm['YEAR'].max())

stats_tex = f"""% Auto-generated by dws_analysis.py
\\newcommand{{\\statsDwsNObs}}{{{r_nonrm['n']:,}}}
\\newcommand{{\\statsDwsBetaNonRm}}{{{r_nonrm['beta']:.4f}}}
\\newcommand{{\\statsDwsSeNonRm}}{{{r_nonrm['se']:.4f}}}
\\newcommand{{\\statsDwsPNonRm}}{{{r_nonrm['pval']:.3f}}}
\\newcommand{{\\statsDwsBetaNrm}}{{{r_nrm['beta']:.4f}}}
\\newcommand{{\\statsDwsSeNrm}}{{{r_nrm['se']:.4f}}}
\\newcommand{{\\statsDwsPNrm}}{{{r_nrm['pval']:.3f}}}
\\newcommand{{\\statsDwsBetaNrc}}{{{r_nrc['beta']:.4f}}}
\\newcommand{{\\statsDwsSeNrc}}{{{r_nrc['se']:.4f}}}
\\newcommand{{\\statsDwsPNrc}}{{{r_nrc['pval']:.3f}}}
\\newcommand{{\\statsDwsYmeanNonRm}}{{{r_nonrm['ymean']:.3f}}}
\\newcommand{{\\statsDwsYmeanNrm}}{{{r_nrm['ymean']:.3f}}}
\\newcommand{{\\statsDwsYmeanNrc}}{{{r_nrc['ymean']:.3f}}}
\\newcommand{{\\statsDwsPctRm}}{{{pct_rm:.0f}}}
\\newcommand{{\\statsDwsPctExitRm}}{{{pct_exit:.0f}}}
\\newcommand{{\\statsDwsPctNrm}}{{{pct_nrm:.0f}}}
\\newcommand{{\\statsDwsPctNrc}}{{{pct_nrc:.0f}}}
\\newcommand{{\\statsDwsPctRc}}{{{pct_rc:.0f}}}
\\newcommand{{\\statsDwsWaveMin}}{{{wave_min}}}
\\newcommand{{\\statsDwsWaveMax}}{{{wave_max}}}
"""

with open(STATS_OUT, 'w', encoding='utf-8') as f:
    f.write(stats_tex)
print(f"Saved: {STATS_OUT}")
print(f"  (Add `\\input{{../paper/stats_dws.tex}}` to main.tex if you want inline DWS stats.)")

# ----------------------------------------------------------------------
# Destination bar chart
# ----------------------------------------------------------------------
dest_shares = rm['js_post'].value_counts(normalize=True).reindex(['RM','NRM','NRC','RC']).fillna(0)
fig, ax = plt.subplots(figsize=(7, 4.5))
colors = ['#b2182b', '#1b7837', '#1E2761', '#762a83']
ax.bar(dest_shares.index, dest_shares.values, color=colors, edgecolor='black', alpha=0.85)
ax.set_ylabel('Share of RM-displaced workers')
ax.set_xlabel('Post-displacement Jaimovich-Siu group')
ax.set_title('Destination distribution: RM-displaced workers, currently employed')
ax.grid(alpha=0.25, axis='y')
for i, v in enumerate(dest_shares.values):
    ax.text(i, v + 0.005, f"{v*100:.1f}%", ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(FIG_OUT, dpi=300)
plt.close()
print(f"Saved: {FIG_OUT}")

print("\n" + "=" * 70)
print("DONE.")
print("=" * 70)

"""
Bartik Industry Shares from CPS Microdata
==========================================
Reads cps_00002.dat (fixed-width), extracts IND1990, computes
state-level employment shares by CES supersector.

Output: state_industry_shares.csv
  Columns: STATEFIP, YEAR, industry, emp_share, emp_count

Also outputs: bartik_instrument.csv
  Pre-built Bartik predicted shocks merged with CES national data.

Author: Ethan Johnston
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from paths import DATA_DIR

CPS_DAT = os.path.join(DATA_DIR, "cps_00002.dat")
CES_FILE = os.path.join(DATA_DIR, "ces_national_industry.csv")

# ============================================================
# IND1990 → CES Supersector Mapping
# ============================================================
# 1990 Census Industry codes: https://usa.ipums.org/usa-action/variables/IND1990
# CES Supersectors use NAICS-based groupings
# This mapping is approximate but standard in the Bartik literature

def ind1990_to_supersector(code):
    """Map 1990 Census industry code to CES supersector.

    NOTE: CES surveys nonfarm establishments only, so we DROP agriculture,
    forestry, and fishing (IND1990 codes 10-32). Earlier versions of this
    code mapped 10-32 to "Mining and logging," which contaminated the
    Bartik instrument because CPS shares (mostly agriculture) were
    multiplied by CES growth rates (only mining + logging). The CES
    "Mining and logging" supersector now corresponds only to IND1990
    40-50 (mining proper).
    """
    if code <= 0:
        return None

    # Agriculture, forestry, fishing — DROP (no CES match)
    # CES does not cover farm employment; mapping ag/forestry/fishing to
    # "Mining and logging" creates a CPS-share / CES-growth mismatch.
    if 10 <= code <= 32:
        return None

    # Mining (metal, coal, oil/gas, nonmetallic)
    if 40 <= code <= 50:
        return "Mining and logging"
    
    # Construction
    if 60 <= code <= 60:
        return "Construction"
    
    # Manufacturing — Nondurable goods
    # Food, tobacco, textiles, apparel, paper, printing, chemicals, petroleum, rubber
    if 100 <= code <= 222:
        return "Manufacturing - Nondurable goods"
    
    # Manufacturing — Durable goods  
    # Lumber, furniture, stone/clay/glass, metals, machinery, electronics, transport equip
    if 230 <= code <= 392:
        return "Manufacturing - Durable goods"
    
    # Transportation
    if 400 <= code <= 432:
        return "Transportation and warehousing"
    
    # Communications (part of Information in CES)
    if 440 <= code <= 442:
        return "Information"
    
    # Utilities
    if 450 <= code <= 472:
        return "Utilities"
    
    # Wholesale trade
    if 500 <= code <= 571:
        return "Wholesale trade"
    
    # Retail trade
    if 580 <= code <= 691:
        return "Retail trade"
    
    # Finance, insurance, real estate
    if 700 <= code <= 712:
        return "Financial activities"
    
    # Business and repair services → Professional and business services
    if 721 <= code <= 760:
        return "Professional and business services"
    
    # Personal services → Other services
    if 761 <= code <= 791:
        return "Other services"
    
    # Entertainment and recreation → Leisure and hospitality
    if 800 <= code <= 810:
        return "Leisure and hospitality"
    
    # Hospitals, health services, education → Education and health services
    if 812 <= code <= 860:
        return "Education and health services"
    
    # Other professional services → Professional and business services
    if 861 <= code <= 893:
        return "Professional and business services"
    
    # Public administration → Government
    if 900 <= code <= 932:
        return "Government"
    
    # Active duty military
    if 940 <= code <= 960:
        return None  # Exclude military
    
    return None


# ============================================================
# STEP 1: Read CPS and Compute Industry Shares
# ============================================================
print("=" * 70)
print("BARTIK INDUSTRY SHARES FROM CPS")
print("=" * 70)

# Only need a few columns — much faster than reading everything
colspecs = [
    (0, 4),      # YEAR
    (9, 11),     # MONTH  
    (48, 50),    # STATEFIP
    (52, 66),    # WTFINL (implied 4 decimals)
    (124, 126),  # EMPSTAT
    (138, 141),  # IND1990
]
colnames = ['YEAR', 'MONTH', 'STATEFIP', 'WTFINL', 'EMPSTAT', 'IND1990']

print(f"\nReading CPS: {CPS_DAT}")
print("This will take a few minutes for the 12 GB file...")

CHUNK_SIZE = 1_000_000
chunks = []
total_read = 0
total_kept = 0

reader = pd.read_fwf(CPS_DAT, colspecs=colspecs, names=colnames,
                      chunksize=CHUNK_SIZE, dtype=str)

for i, chunk in enumerate(reader):
    total_read += len(chunk)
    
    # Convert types
    chunk['YEAR'] = pd.to_numeric(chunk['YEAR'], errors='coerce')
    chunk['MONTH'] = pd.to_numeric(chunk['MONTH'], errors='coerce')
    chunk['STATEFIP'] = pd.to_numeric(chunk['STATEFIP'], errors='coerce')
    chunk['WTFINL'] = pd.to_numeric(chunk['WTFINL'], errors='coerce') / 10000
    chunk['EMPSTAT'] = pd.to_numeric(chunk['EMPSTAT'], errors='coerce')
    chunk['IND1990'] = pd.to_numeric(chunk['IND1990'], errors='coerce')
    
    # Filter: employed, valid industry, valid state, positive weight
    mask = (
        chunk['EMPSTAT'].isin([10, 12]) &
        (chunk['IND1990'] > 0) &
        (chunk['STATEFIP'] >= 1) & (chunk['STATEFIP'] <= 56) &
        (chunk['WTFINL'] > 0)
    )
    kept = chunk[mask][['YEAR', 'MONTH', 'STATEFIP', 'WTFINL', 'IND1990']].copy()
    
    # Map to supersector immediately to save memory
    kept['industry'] = kept['IND1990'].apply(ind1990_to_supersector)
    kept = kept[kept['industry'].notna()]
    
    total_kept += len(kept)
    chunks.append(kept)
    
    if (i + 1) % 10 == 0:
        print(f"  Chunk {i+1}: {total_read:,} read, {total_kept:,} kept")

print(f"\n  Total: {total_read:,} read, {total_kept:,} employed with valid industry")

cps = pd.concat(chunks, ignore_index=True)
cps['YEAR'] = cps['YEAR'].astype(int)
cps['STATEFIP'] = cps['STATEFIP'].astype(int)

print(f"  Industries mapped: {cps['industry'].nunique()}")
print(f"  Industry distribution:")
for ind in sorted(cps['industry'].unique()):
    n = len(cps[cps['industry'] == ind])
    pct = n / len(cps) * 100
    print(f"    {ind}: {n:,} ({pct:.1f}%)")


# ============================================================
# STEP 2: Compute State-Year Industry Employment Shares
# ============================================================
print("\n" + "-" * 50)
print("STEP 2: State-Year Industry Shares")
print("-" * 50)

# Aggregate to state-year-industry (annual for Bartik base shares)
state_ind = cps.groupby(['STATEFIP', 'YEAR', 'industry']).agg(
    emp_weighted=('WTFINL', 'sum'),
    n_obs=('WTFINL', 'count'),
).reset_index()

# Compute shares within state-year
state_totals = state_ind.groupby(['STATEFIP', 'YEAR'])['emp_weighted'].sum().reset_index()
state_totals.rename(columns={'emp_weighted': 'total_emp'}, inplace=True)

state_ind = state_ind.merge(state_totals, on=['STATEFIP', 'YEAR'])
state_ind['emp_share'] = state_ind['emp_weighted'] / state_ind['total_emp']

print(f"  State-year-industry obs: {len(state_ind):,}")
print(f"  States: {state_ind['STATEFIP'].nunique()}")
print(f"  Years: {state_ind['YEAR'].min()}-{state_ind['YEAR'].max()}")

# Save
shares_path = os.path.join(DATA_DIR, "state_industry_shares.csv")
state_ind.to_csv(shares_path, index=False)
print(f"  Saved: {shares_path}")

# Also compute state-quarter-industry for higher-frequency Bartik
cps['QUARTER'] = (cps['MONTH'] - 1) // 3 + 1
state_ind_q = cps.groupby(['STATEFIP', 'YEAR', 'QUARTER', 'industry']).agg(
    emp_weighted=('WTFINL', 'sum'),
).reset_index()

state_totals_q = state_ind_q.groupby(['STATEFIP', 'YEAR', 'QUARTER'])['emp_weighted'].sum().reset_index()
state_totals_q.rename(columns={'emp_weighted': 'total_emp'}, inplace=True)
state_ind_q = state_ind_q.merge(state_totals_q, on=['STATEFIP', 'YEAR', 'QUARTER'])
state_ind_q['emp_share'] = state_ind_q['emp_weighted'] / state_ind_q['total_emp']

shares_q_path = os.path.join(DATA_DIR, "state_industry_shares_quarterly.csv")
state_ind_q.to_csv(shares_q_path, index=False)
print(f"  Saved quarterly: {shares_q_path}")


# ============================================================
# STEP 3: Build Bartik Instrument
# ============================================================
print("\n" + "-" * 50)
print("STEP 3: Building Bartik Instrument")
print("-" * 50)

# Load CES national industry employment
ces = pd.read_csv(CES_FILE)
print(f"  CES national: {len(ces):,} obs, {ces['industry'].nunique()} industries")

# Compute national employment growth rates by industry-year
# (Annual: sum months, then year-over-year growth)
ces_annual = ces.groupby(['industry', 'year'])['employment'].mean().reset_index()
ces_annual = ces_annual.sort_values(['industry', 'year'])
ces_annual['emp_lag'] = ces_annual.groupby('industry')['employment'].shift(1)
ces_annual['growth_rate'] = (ces_annual['employment'] - ces_annual['emp_lag']) / ces_annual['emp_lag']
ces_annual = ces_annual.dropna(subset=['growth_rate'])

print(f"  National growth rates: {len(ces_annual):,} industry-year obs")

# Check which industries match between CPS and CES
cps_industries = set(state_ind['industry'].unique())
ces_industries = set(ces_annual['industry'].unique())
matched = cps_industries & ces_industries
missing_cps = cps_industries - ces_industries
missing_ces = ces_industries - cps_industries

print(f"\n  Industry matching:")
print(f"    CPS industries: {len(cps_industries)}")
print(f"    CES industries: {len(ces_industries)}")
print(f"    Matched: {len(matched)}")
if missing_cps:
    print(f"    In CPS but not CES: {missing_cps}")
if missing_ces:
    print(f"    In CES but not CPS: {missing_ces}")

# Build Bartik: for each state-year, predicted employment shock =
# Σ_j ω_{sj,t-1} × g_{jt}^{-s}
#
# TRUE LEAVE-ONE-OUT IMPLEMENTATION
# ---------------------------------
# Earlier versions of this file approximated g_{jt}^{-s} ≈ g_{jt}^{national},
# which is biased for large states (CA, TX, NY account for >10% of some
# industry-totals). We now compute g_{jt}^{-s} properly by:
#   1. Using CPS state-industry employment to get each state's share of
#      total national CPS employment in industry j, year t.
#   2. Applying that share to CES national levels to back out the state's
#      "CES-equivalent" employment in (j, t).
#   3. Subtracting from CES national to get LOO levels.
#   4. Computing year-over-year growth on the LOO levels.

# Step 1: state's share of national CPS employment in each (industry, year)
si = state_ind[['STATEFIP', 'YEAR', 'industry', 'emp_weighted']].copy()
si['cps_nat_total'] = si.groupby(['YEAR', 'industry'])['emp_weighted'].transform('sum')
si['state_nat_share'] = si['emp_weighted'] / si['cps_nat_total']

# Step 2: merge CES national levels and compute state's CES-equivalent
ces_lev = ces_annual[['industry', 'year', 'employment']].rename(
    columns={'year': 'YEAR', 'employment': 'ces_emp_nat'}
)
si = si.merge(ces_lev, on=['YEAR', 'industry'], how='inner')
si['state_ces_equiv'] = si['state_nat_share'] * si['ces_emp_nat']

# Step 3: LOO national level for each (state, industry, year)
si['loo_emp'] = si['ces_emp_nat'] - si['state_ces_equiv']

# Step 4: LOO growth rate (year-over-year, within state-industry)
si = si.sort_values(['STATEFIP', 'industry', 'YEAR']).reset_index(drop=True)
si['loo_emp_lag'] = si.groupby(['STATEFIP', 'industry'])['loo_emp'].shift(1)
si['g_loo'] = (si['loo_emp'] - si['loo_emp_lag']) / si['loo_emp_lag']
si = si.dropna(subset=['g_loo']).copy()

# Step 5: combine with lagged shares (ω_{sj,t-1}) and aggregate
# Lagged shares: take state_ind at year t-1 and align to year t
prev_shares = state_ind[['STATEFIP', 'YEAR', 'industry', 'emp_share']].copy()
prev_shares['YEAR'] = prev_shares['YEAR'] + 1
prev_shares.rename(columns={'emp_share': 'omega_lag'}, inplace=True)

bartik_pieces = si[['STATEFIP', 'YEAR', 'industry', 'g_loo']].merge(
    prev_shares, on=['STATEFIP', 'YEAR', 'industry'], how='inner'
)
bartik_pieces['contribution'] = bartik_pieces['omega_lag'] * bartik_pieces['g_loo']

bartik_df = bartik_pieces.groupby(['STATEFIP', 'YEAR'])['contribution'].sum().reset_index()
bartik_df.columns = ['STATEFIP', 'YEAR', 'bartik_shock']
bartik_df = bartik_df.sort_values(['STATEFIP', 'YEAR']).reset_index(drop=True)

# Sanity check: how much does state s contribute to its own predicted shock
# under the LOO construction? Should be small but non-zero only via the share.
print(f"\n  LOO diagnostic:")
state_total_share = si.groupby(['STATEFIP', 'YEAR'])['state_nat_share'].sum()
print(f"    Avg cumulative state share across industries: {state_total_share.mean():.4f}")
print(f"    Max  cumulative state share across industries: {state_total_share.max():.4f}")

bartik_rows = [bartik_df]  # keep variable name compatible with downstream block

if bartik_rows:
    bartik_df = pd.concat(bartik_rows, ignore_index=True)
    bartik_df = bartik_df.sort_values(['STATEFIP', 'YEAR']).reset_index(drop=True)
    
    bartik_path = os.path.join(DATA_DIR, "bartik_instrument.csv")
    bartik_df.to_csv(bartik_path, index=False)
    
    print(f"\n  Bartik instrument: {len(bartik_df):,} state-year obs")
    print(f"  Years: {bartik_df['YEAR'].min()}-{bartik_df['YEAR'].max()}")
    print(f"  States: {bartik_df['STATEFIP'].nunique()}")
    print(f"  Bartik shock distribution:")
    print(f"    mean:   {bartik_df['bartik_shock'].mean():.4f}")
    print(f"    sd:     {bartik_df['bartik_shock'].std():.4f}")
    print(f"    min:    {bartik_df['bartik_shock'].min():.4f}")
    print(f"    max:    {bartik_df['bartik_shock'].max():.4f}")
    print(f"    p10:    {bartik_df['bartik_shock'].quantile(0.1):.4f}")
    print(f"    p90:    {bartik_df['bartik_shock'].quantile(0.9):.4f}")
    print(f"\n  Saved: {bartik_path}")
else:
    print("\n  WARNING: Could not construct Bartik — check industry matching")


# ============================================================
# STEP 4: Also build quarterly Bartik
# ============================================================
print("\n" + "-" * 50)
print("STEP 4: Quarterly Bartik")
print("-" * 50)

# Quarterly Bartik with proper LOO
# Same logic as annual but on (state, year, quarter, industry) panel.
ces['QUARTER'] = (ces['month'] - 1) // 3 + 1
ces_q = ces.groupby(['industry', 'year', 'QUARTER'])['employment'].mean().reset_index()
ces_q = ces_q.sort_values(['industry', 'year', 'QUARTER']).reset_index(drop=True)

# Need state shares of national CPS employment by industry-year-quarter
# (state_ind_q already has emp_weighted by state-year-quarter-industry)
siq = state_ind_q[['STATEFIP', 'YEAR', 'QUARTER', 'industry', 'emp_weighted']].copy()
siq['cps_nat_total'] = siq.groupby(['YEAR', 'QUARTER', 'industry'])['emp_weighted'].transform('sum')
siq['state_nat_share'] = siq['emp_weighted'] / siq['cps_nat_total']

# Merge CES levels (note: ces_q uses 'year' lowercase)
ces_q_lev = ces_q[['industry', 'year', 'QUARTER', 'employment']].rename(
    columns={'year': 'YEAR', 'employment': 'ces_emp_nat'}
)
siq = siq.merge(ces_q_lev, on=['YEAR', 'QUARTER', 'industry'], how='inner')
siq['state_ces_equiv'] = siq['state_nat_share'] * siq['ces_emp_nat']
siq['loo_emp'] = siq['ces_emp_nat'] - siq['state_ces_equiv']

# 4-quarter lag for year-over-year growth at quarterly frequency
siq['yq'] = siq['YEAR'] * 4 + siq['QUARTER']
siq = siq.sort_values(['STATEFIP', 'industry', 'yq']).reset_index(drop=True)
siq['loo_emp_lag4'] = siq.groupby(['STATEFIP', 'industry'])['loo_emp'].shift(4)
siq['g_loo'] = (siq['loo_emp'] - siq['loo_emp_lag4']) / siq['loo_emp_lag4']
siq = siq.dropna(subset=['g_loo']).copy()

# Lagged shares (one year prior for the same quarter)
prev_shares_q = state_ind_q[['STATEFIP', 'YEAR', 'QUARTER', 'industry', 'emp_share']].copy()
prev_shares_q['YEAR'] = prev_shares_q['YEAR'] + 1
prev_shares_q.rename(columns={'emp_share': 'omega_lag'}, inplace=True)

bartik_q_pieces = siq[['STATEFIP', 'YEAR', 'QUARTER', 'industry', 'g_loo']].merge(
    prev_shares_q, on=['STATEFIP', 'YEAR', 'QUARTER', 'industry'], how='inner'
)
bartik_q_pieces['contribution'] = bartik_q_pieces['omega_lag'] * bartik_q_pieces['g_loo']

bartik_q_df = bartik_q_pieces.groupby(['STATEFIP', 'YEAR', 'QUARTER'])['contribution'].sum().reset_index()
bartik_q_df.columns = ['STATEFIP', 'YEAR', 'QUARTER', 'bartik_shock']
bartik_q_df = bartik_q_df.sort_values(['STATEFIP', 'YEAR', 'QUARTER']).reset_index(drop=True)

bartik_q_rows = [bartik_q_df]  # keep variable name compatible with downstream block

if bartik_q_rows:
    bartik_q_df = pd.concat(bartik_q_rows, ignore_index=True)
    bartik_q_df = bartik_q_df.sort_values(['STATEFIP', 'YEAR', 'QUARTER']).reset_index(drop=True)
    
    bq_path = os.path.join(DATA_DIR, "bartik_instrument_quarterly.csv")
    bartik_q_df.to_csv(bq_path, index=False)
    
    print(f"  Quarterly Bartik: {len(bartik_q_df):,} state-quarter obs")
    print(f"  Saved: {bq_path}")
else:
    print("  Could not build quarterly Bartik")


print("\n" + "=" * 70)
print("BARTIK CONSTRUCTION COMPLETE")
print("=" * 70)
print(f"\nOutput files:")
print(f"  {shares_path}")
print(f"  {shares_q_path}")
print(f"  {bartik_path if bartik_rows else 'FAILED'}")
print(f"  {bq_path if bartik_q_rows else 'FAILED'}")
print(f"\nNext: Build LP-IV using these + laus_state_unemployment.csv + state panels")

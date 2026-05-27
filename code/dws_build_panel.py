"""
Build the Displaced Worker Survey (DWS) analysis panel from cps_00003.dat.

Output: dws_panel.parquet (one row per displaced worker, pre- and post-
displacement occupation classified into Jaimovich-Siu groups, merged with
state-year Bartik shock for the displacement year).

Pipeline:
  1. Parse fixed-width cps_00003.dat using IPUMS codebook column positions.
  2. Filter to DWS-eligible respondents (DWRESP == 02, supplement weight > 0).
  3. Filter to involuntary displaced workers (DWSTAT == 01 and DWREAS in 01/02/03).
  4. Build pre-displacement JS group from DWOCC1990.
  5. Build post-displacement JS group from current OCC2010.
  6. Compute displacement year from CPS year and DWLASTWRK ("years ago").
  7. Merge with bartik_instrument.csv on (STATEFIP, displacement_year).
  8. Save panel.

Author: Ethan Johnston (with assist).
"""

import os
import numpy as np
import pandas as pd

from paths import DATA_DIR
DAT_FILE    = os.path.join(DATA_DIR, "cps_00003.dat")
BARTIK_CSV  = os.path.join(DATA_DIR, "bartik_instrument.csv")
OUT_PARQUET = os.path.join(DATA_DIR, "dws_panel.parquet")

# ----------------------------------------------------------------------
# 1) Fixed-width column layout (1-indexed in codebook → 0-indexed for pandas)
# ----------------------------------------------------------------------
COLSPECS = [
    (0,   4),   # YEAR
    (9,  11),   # MONTH
    (48, 50),   # STATEFIP
    (50, 52),   # PERNUM
    (52, 66),   # WTFINL
    (66, 80),   # CPSIDP
    (110,112),  # AGE
    (112,113),  # SEX
    (113,116),  # RACE
    (121,124),  # HISPAN
    (124,126),  # EMPSTAT
    (126,127),  # LABFORCE
    (131,135),  # OCC2010 (post-displacement occupation if currently working)
    (135,138),  # OCC1990 (post-displacement occupation, harmonized backup)
    (138,141),  # IND1990 (current industry)
    (151,153),  # DWSTAT
    (153,155),  # DWREAS
    (155,157),  # DWLASTWRK
    (157,161),  # DWYEARS
    (161,163),  # DWFULLTIME
    (163,165),  # DWCLASS
    (165,168),  # DWIND1990
    (168,171),  # DWOCC1990  <-- pre-displacement occupation, harmonized
    (171,173),  # DWRESP
    (173,183),  # DWSUPPWT
    (183,186),  # DWWKSUN
]
COLNAMES = [
    'YEAR','MONTH','STATEFIP','PERNUM','WTFINL','CPSIDP',
    'AGE','SEX','RACE','HISPAN','EMPSTAT','LABFORCE',
    'OCC2010','OCC1990','IND1990',
    'DWSTAT','DWREAS','DWLASTWRK','DWYEARS','DWFULLTIME','DWCLASS',
    'DWIND1990','DWOCC1990','DWRESP','DWSUPPWT','DWWKSUN',
]
NUMERIC = [c for c in COLNAMES if c not in ()]

# ----------------------------------------------------------------------
# 2) Jaimovich-Siu group classification by OCC1990 code
#    Mapping rationale (Jaimovich & Siu 2020, restud):
#      NRC: managerial + professional + technical (broad 003-235)
#      RC : sales + administrative support (243-391)
#      NRM: service occupations (405-469)
#      RM : precision production, operators, transportation, helpers,
#           laborers (503-890); military separate.
# ----------------------------------------------------------------------
def js_from_occ1990(occ):
    o = pd.to_numeric(occ, errors='coerce')
    if pd.isna(o): return np.nan
    o = int(o)
    if 3   <= o <= 235:  return 'NRC'   # managerial + professional + technical
    if 243 <= o <= 391:  return 'RC'    # sales + admin support
    if 405 <= o <= 469:  return 'NRM'   # service occupations
    if 503 <= o <= 890:  return 'RM'    # production, operators, transport, laborers
    if 905 == o:         return np.nan  # military: drop
    return np.nan

# Post-displacement JS group from OCC2010 (uses your existing mapping)
def js_from_occ2010(occ):
    o = pd.to_numeric(occ, errors='coerce')
    if pd.isna(o): return np.nan
    o = int(o)
    if 10   <= o <= 3540:                              return 'NRC'
    if (4700 <= o <= 4965) or (5000 <= o <= 5940):     return 'RC'
    if 3600 <= o <= 4650:                              return 'NRM'
    if 6005 <= o <= 9750:                              return 'RM'
    return np.nan

# ----------------------------------------------------------------------
# 3) Parse the fixed-width file in chunks
# ----------------------------------------------------------------------
print("=" * 70)
print("BUILDING DWS PANEL FROM cps_00003.dat")
print("=" * 70)
print(f"Reading: {DAT_FILE}")

CHUNK = 1_000_000
keepers = []
total_read = 0

for i, chunk in enumerate(
    pd.read_fwf(DAT_FILE, colspecs=COLSPECS, names=COLNAMES,
                chunksize=CHUNK, dtype=str)
):
    total_read += len(chunk)
    # Convert numeric columns
    for c in COLNAMES:
        chunk[c] = pd.to_numeric(chunk[c], errors='coerce')

    # Filter to DWS-eligible interviewed respondents only
    # DWRESP == 02 = "Eligible, interviewed"
    mask = chunk['DWRESP'] == 2
    sub = chunk.loc[mask].copy()
    if len(sub) > 0:
        keepers.append(sub)

    print(f"  Chunk {i+1}: {total_read:>10,} read, {sum(len(k) for k in keepers):>9,} DWS-eligible kept")

if not keepers:
    raise RuntimeError("No DWS-eligible records found. Check column positions.")

df = pd.concat(keepers, ignore_index=True)
del keepers
print(f"\nDWS-eligible interviewed: {len(df):,} respondents")
print(f"Year range: {int(df['YEAR'].min())}–{int(df['YEAR'].max())}")
print(f"DWS waves: {sorted(df['YEAR'].unique().astype(int))}")

# ----------------------------------------------------------------------
# 4) Apply weights, demographic filters
# ----------------------------------------------------------------------
# DWSUPPWT is implied-4-decimal in IPUMS fixed-width (like WTFINL)
df['DWSUPPWT'] = df['DWSUPPWT'] / 10000.0
df['WTFINL']   = df['WTFINL']   / 10000.0

# Filter: positive supplement weight, working-age, valid state
df = df[
    (df['DWSUPPWT'] > 0) &
    (df['AGE'] >= 20) & (df['AGE'] <= 64) &
    (df['STATEFIP'] >= 1) & (df['STATEFIP'] <= 56)
].copy()
print(f"After basic filters (age 20–64, positive weight, valid state): {len(df):,}")

# ----------------------------------------------------------------------
# 5) Identify displaced workers (DWSTAT == 01)
#    Filter to involuntary reasons: 01 plant closed, 02 insufficient work,
#    03 position abolished.
# ----------------------------------------------------------------------
df['displaced'] = (df['DWSTAT'] == 1).astype(int)
print(f"  Displaced (DWSTAT==1): {df['displaced'].sum():,}")

dw = df[df['displaced'] == 1].copy()
print(f"\nDisplaced workers before reason filter: {len(dw):,}")

# Involuntary displacement only (drop seasonal, business-failure, other)
dw = dw[dw['DWREAS'].isin([1, 2, 3])].copy()
print(f"After involuntary-reason filter (DWREAS in 1,2,3): {len(dw):,}")

# Drop respondents who never had a full-time job lost (optional but standard)
# DWFULLTIME 02 = Yes, full-time
dw = dw[dw['DWFULLTIME'] == 2].copy()
print(f"After full-time-on-lost-job filter: {len(dw):,}")

# Drop displaced from government/self-employed/unpaid (focus on private sector
# where the demand-shock-displacement story is clean)
dw = dw[dw['DWCLASS'] == 2].copy()
print(f"After private-sector-only filter (DWCLASS==2): {len(dw):,}")

# ----------------------------------------------------------------------
# 6) Build JS groups (pre and post)
# ----------------------------------------------------------------------
dw['js_pre']  = dw['DWOCC1990'].apply(js_from_occ1990)
dw['js_post'] = dw['OCC2010'].apply(js_from_occ2010)

print(f"\nPre-displacement JS group distribution:")
print(dw['js_pre'].value_counts(dropna=False))
print(f"\nPost-displacement JS group distribution (NaN = not currently employed):")
print(dw['js_post'].value_counts(dropna=False))

# Require valid pre-displacement JS group
dw = dw.dropna(subset=['js_pre']).copy()
print(f"\nAfter requiring valid pre-displacement JS group: {len(dw):,}")

# ----------------------------------------------------------------------
# 7) Compute displacement year
#    DWLASTWRK = years ago last worked at lost job
#    Displacement year = CPS YEAR - DWLASTWRK
#    DWLASTWRK codes: 00=this year, 01=last year, 02-05=2-5 years ago, 95+=other
# ----------------------------------------------------------------------
dw['displacement_year'] = np.where(
    dw['DWLASTWRK'].between(0, 5),
    dw['YEAR'] - dw['DWLASTWRK'],
    np.nan
)
dw = dw.dropna(subset=['displacement_year']).copy()
dw['displacement_year'] = dw['displacement_year'].astype(int)
print(f"After requiring valid displacement_year (DWLASTWRK 0–5 years): {len(dw):,}")

# ----------------------------------------------------------------------
# 8) Build outcome indicators
#    rm_to_nrm: pre = RM and post = NRM
#    rm_exit  : pre = RM and post != RM (and post not missing, i.e. currently working)
#    Note: post being NaN means worker is currently not employed; we treat
#    those separately. The main analysis uses respondents currently working.
# ----------------------------------------------------------------------
dw['currently_employed'] = dw['EMPSTAT'].isin([10, 12]).astype(int)
print(f"\nCurrently employed: {dw['currently_employed'].sum():,} / {len(dw):,}")

# Restrict to currently-employed for transition analysis
sample = dw[dw['currently_employed'] == 1].dropna(subset=['js_post']).copy()
print(f"Currently-employed displaced workers with valid post-JS: {len(sample):,}")

sample['rm_to_nrm']    = ((sample['js_pre'] == 'RM') & (sample['js_post'] == 'NRM')).astype(int)
sample['rm_to_nonrm']  = ((sample['js_pre'] == 'RM') & (sample['js_post'] != 'RM')).astype(int)
sample['rm_to_nrc']    = ((sample['js_pre'] == 'RM') & (sample['js_post'] == 'NRC')).astype(int)
sample['rm_to_rc']     = ((sample['js_pre'] == 'RM') & (sample['js_post'] == 'RC')).astype(int)

# ----------------------------------------------------------------------
# 9) Demographic controls
# ----------------------------------------------------------------------
sample['female']   = (sample['SEX']    == 2).astype(int)
sample['hispanic'] = (sample['HISPAN'] > 0).astype(int)
sample['black']    = (sample['RACE']   == 200).astype(int)
sample['age_sq']   = sample['AGE'] ** 2

# ----------------------------------------------------------------------
# 10) Merge with Bartik shock at displacement-year level
# ----------------------------------------------------------------------
bartik = pd.read_csv(BARTIK_CSV)
bartik = bartik.rename(columns={'YEAR': 'displacement_year'})
bartik['STATEFIP'] = bartik['STATEFIP'].astype(int)
bartik['displacement_year'] = bartik['displacement_year'].astype(int)

sample['STATEFIP'] = sample['STATEFIP'].astype(int)
sample = sample.merge(bartik, on=['STATEFIP', 'displacement_year'], how='inner')
print(f"\nAfter Bartik merge on (STATEFIP, displacement_year): {len(sample):,}")
print(f"Year range of displacements: {sample['displacement_year'].min()}–{sample['displacement_year'].max()}")

# Subset: RM-displaced workers (the population for our main test)
rm_sample = sample[sample['js_pre'] == 'RM'].copy()
print(f"\nRM-displaced subsample: {len(rm_sample):,}")
print(f"  Of which → RM    (stayed): {(rm_sample['js_post']=='RM').sum():>6,}")
print(f"  Of which → NRM   (service): {(rm_sample['js_post']=='NRM').sum():>6,}")
print(f"  Of which → RC    (clerical/sales): {(rm_sample['js_post']=='RC').sum():>6,}")
print(f"  Of which → NRC   (prof/tech/mgmt): {(rm_sample['js_post']=='NRC').sum():>6,}")

# ----------------------------------------------------------------------
# 11) Save
# ----------------------------------------------------------------------
sample.to_parquet(OUT_PARQUET, index=False)
print(f"\nSaved panel: {OUT_PARQUET}")
print(f"  Total displaced workers: {len(sample):,}")
print(f"  RM-displaced subsample:  {len(rm_sample):,}")
print(f"  States: {sample['STATEFIP'].nunique()}")
print(f"  DWS waves: {sorted(sample['YEAR'].unique().astype(int))}")
print("Done.")

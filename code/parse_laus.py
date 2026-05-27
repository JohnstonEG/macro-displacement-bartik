"""
Parse BLS LAUS State Unemployment from Excel
=============================================
Input:  ststdnsadata.xlsx (not seasonally adjusted)
Output: laus_state_unemployment.csv

Author: Ethan Johnston
"""

import pandas as pd
import numpy as np
import os

from paths import DATA_DIR

# Read skipping header rows, no header (we'll name columns manually)
df = pd.read_excel(
    os.path.join(DATA_DIR, "ststdnsadata.xlsx"),
    header=None,
    skiprows=8,  # data starts at row 8
)

print(f"Raw: {len(df)} rows, {df.shape[1]} columns")
print(f"Sample row:\n{df.iloc[0].tolist()}")

# Columns based on BLS layout:
# 0: FIPS Code
# 1: State and area
# 2: Year
# 3: Month
# 4: Civilian non-institutional population
# 5: Civilian labor force (Total)
# 6: LF participation rate
# 7: Employment (Total)
# 8: Employment/pop ratio
# 9: Unemployment (Total)
# 10: Unemployment Rate

df.columns = ['fips_raw', 'area', 'year', 'month', 'pop', 
              'lf_total', 'lf_rate', 'emp_total', 'emp_pop_ratio',
              'unemp_total', 'unemp_rate']

# Clean FIPS: keep only valid state-level FIPS (1-56, 1-2 digit codes)
# Drop sub-state areas like "Los Angeles County" (3+ digit FIPS)
df['fips_raw'] = df['fips_raw'].astype(str).str.strip()
df = df[df['fips_raw'].str.match(r'^\d{1,2}$', na=False)].copy()
df['STATEFIP'] = df['fips_raw'].astype(int)

# Keep only valid state FIPS (1-56, excluding 3=AS, 7=CZ, etc.)
valid_fips = set(range(1, 57)) - {3, 7, 14, 43, 52}
df = df[df['STATEFIP'].isin(valid_fips)].copy()

# Clean year/month
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['month'] = pd.to_numeric(df['month'], errors='coerce')
df = df.dropna(subset=['year', 'month'])
df['year'] = df['year'].astype(int)
df['month'] = df['month'].astype(int)

# Clean unemployment rate
df['unemployment_rate'] = pd.to_numeric(df['unemp_rate'], errors='coerce')
df['unemployment_total'] = pd.to_numeric(df['unemp_total'], errors='coerce')
df['employment_total'] = pd.to_numeric(df['emp_total'], errors='coerce')
df['labor_force'] = pd.to_numeric(df['lf_total'], errors='coerce')
df['population'] = pd.to_numeric(df['pop'], errors='coerce')

# Select and save
out = df[['STATEFIP', 'area', 'year', 'month', 'unemployment_rate',
          'unemployment_total', 'employment_total', 'labor_force', 'population']].copy()
out = out.rename(columns={'area': 'state_name'})
out = out.sort_values(['STATEFIP', 'year', 'month']).reset_index(drop=True)

path = os.path.join(DATA_DIR, "laus_state_unemployment.csv")
out.to_csv(path, index=False)

print(f"\nSaved: {path}")
print(f"  {len(out):,} obs")
print(f"  {out['STATEFIP'].nunique()} states")
print(f"  Years: {out['year'].min()}-{out['year'].max()}")
print(f"\n  Sample (Alabama 2020):")
s = out[(out['STATEFIP'] == 1) & (out['year'] == 2020) & (out['month'].isin([1, 4, 7]))]
print(s[['STATEFIP', 'state_name', 'year', 'month', 'unemployment_rate']].to_string(index=False))

# Check for gaps
print(f"\n  Obs per state (should be ~{12 * (out['year'].max() - out['year'].min() + 1)}):")
counts = out.groupby('STATEFIP').size()
print(f"    min={counts.min()}, max={counts.max()}, median={counts.median():.0f}")

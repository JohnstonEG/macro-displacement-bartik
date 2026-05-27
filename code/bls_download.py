"""
BLS Data Download via API
==========================
Downloads:
  1. CES (Current Employment Statistics) — national employment by industry
     Used for: Bartik instrument national shocks
  2. LAUS (Local Area Unemployment Statistics) — state-level unemployment
     Used for: ΔUnemployment_st in LP-IV

BLS API v2: https://api.bls.gov/publicAPI/v2/timeseries/data/
No key = 25 requests/day, 10 years max per request
With key = 500 requests/day, 20 years max per request

If you have a key, set BLS_API_KEY below.

Outputs to: D:\Data\MacroDisplacement\
  - ces_national_industry.csv
  - laus_state_unemployment.csv

Author: Ethan Johnston
"""

import requests
import pandas as pd
import json
import time
import os

# ============================================================
# CONFIG
# ============================================================
from paths import DATA_DIR
os.makedirs(DATA_DIR, exist_ok=True)

# BLS API key — optional; it only raises the request rate limit.
# Register a free key at https://data.bls.gov/registrationEngine/ and supply it
# through the BLS_API_KEY environment variable rather than hardcoding it here:
#   PowerShell:  $env:BLS_API_KEY = "your-key"
#   bash/zsh:    export BLS_API_KEY="your-key"
# If the variable is unset, the script falls back to keyless access (lower limits).
BLS_API_KEY = os.environ.get("BLS_API_KEY")

API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

# Year ranges — split into chunks of 10 (no-key limit) or 20 (with key)
CHUNK_SIZE = 20 if BLS_API_KEY else 10
START_YEAR = 1976
END_YEAR = 2025

def year_chunks(start, end, size):
    """Split year range into chunks."""
    chunks = []
    y = start
    while y <= end:
        ye = min(y + size - 1, end)
        chunks.append((y, ye))
        y = ye + 1
    return chunks

YEAR_CHUNKS = year_chunks(START_YEAR, END_YEAR, CHUNK_SIZE)

def bls_fetch(series_ids, start_year, end_year):
    """Fetch series from BLS API."""
    payload = {
        "seriesid": series_ids,
        "startyear": str(start_year),
        "endyear": str(end_year),
    }
    if BLS_API_KEY:
        payload["registrationkey"] = BLS_API_KEY

    headers = {"Content-type": "application/json"}
    r = requests.post(API_URL, data=json.dumps(payload), headers=headers)
    r.raise_for_status()
    data = r.json()

    if data['status'] != 'REQUEST_SUCCEEDED':
        print(f"  API error: {data.get('message', 'unknown')}")
        return []

    return data.get('Results', {}).get('series', [])


# ============================================================
# 1. CES — National Employment by Industry Supersector
# ============================================================
print("=" * 70)
print("1. CES National Employment by Industry")
print("=" * 70)

# CES series IDs for major industry supersectors (seasonally adjusted)
# Format: CES{seasonal}{supersector}{data_type}
# Seasonal: S = seasonally adjusted
# Data type: 01 = All employees, thousands
#
# Supersectors (2-digit NAICS mapping):
CES_SERIES = {
    "CES0500000001": "Total private",
    "CES1000000001": "Mining and logging",
    "CES2000000001": "Construction",
    "CES3000000001": "Manufacturing",
    "CES3100000001": "Manufacturing - Durable goods",
    "CES3200000001": "Manufacturing - Nondurable goods",
    "CES4000000001": "Trade, transportation, utilities",
    "CES4142000001": "Wholesale trade",
    "CES4200000001": "Retail trade",
    "CES4300000001": "Transportation and warehousing",
    "CES4400000001": "Utilities",
    "CES5000000001": "Information",
    "CES5500000001": "Financial activities",
    "CES6000000001": "Professional and business services",
    "CES6500000001": "Education and health services",
    "CES7000000001": "Leisure and hospitality",
    "CES8000000001": "Other services",
    "CES9000000001": "Government",
    "CES9091000001": "Federal government",
    "CES9092000001": "State government",
    "CES9093000001": "Local government",
}

series_ids = list(CES_SERIES.keys())
all_ces = []

# BLS limits to 50 series per request (25 without key)
MAX_SERIES = 50 if BLS_API_KEY else 25

for chunk_start, chunk_end in YEAR_CHUNKS:
    print(f"\n  Fetching {chunk_start}-{chunk_end}...")
    
    # Split series into batches if needed
    for batch_start in range(0, len(series_ids), MAX_SERIES):
        batch = series_ids[batch_start:batch_start + MAX_SERIES]
        
        try:
            results = bls_fetch(batch, chunk_start, chunk_end)
            
            for s in results:
                sid = s['seriesID']
                label = CES_SERIES.get(sid, sid)
                for obs in s['data']:
                    year = int(obs['year'])
                    period = obs['period']
                    if not period.startswith('M'):
                        continue
                    month = int(period[1:])
                    value = obs['value'].replace(',', '')
                    try:
                        value = float(value)
                    except ValueError:
                        continue
                    
                    all_ces.append({
                        'series_id': sid,
                        'industry': label,
                        'year': year,
                        'month': month,
                        'employment': value,
                    })
            
            n = len([r for r in results])
            print(f"    Batch {batch_start//MAX_SERIES + 1}: {n} series returned")
            
        except Exception as e:
            print(f"    Error: {e}")
        
        time.sleep(1)  # Rate limiting

ces_df = pd.DataFrame(all_ces)
if len(ces_df) > 0:
    ces_df = ces_df.sort_values(['series_id', 'year', 'month']).reset_index(drop=True)
    ces_path = os.path.join(DATA_DIR, "ces_national_industry.csv")
    ces_df.to_csv(ces_path, index=False)
    print(f"\nSaved CES: {ces_path}")
    print(f"  {len(ces_df):,} obs, {ces_df['industry'].nunique()} industries, "
          f"{ces_df['year'].min()}-{ces_df['year'].max()}")
    print(f"\n  Industries:")
    for ind in ces_df['industry'].unique():
        n = len(ces_df[ces_df['industry'] == ind])
        print(f"    {ind}: {n} obs")
else:
    print("\n  WARNING: No CES data retrieved!")


# ============================================================
# 2. LAUS — State Unemployment Rates
# ============================================================
print("\n" + "=" * 70)
print("2. LAUS State Unemployment Rates")
print("=" * 70)

# LAUS series format: LAUST{FIPS}{measure}
# Measure: 03 = unemployment rate
# FIPS codes for 50 states + DC (2-digit, zero-padded)
FIPS_STATES = {
    '01': 'Alabama', '02': 'Alaska', '04': 'Arizona', '05': 'Arkansas',
    '06': 'California', '08': 'Colorado', '09': 'Connecticut', '10': 'Delaware',
    '11': 'DC', '12': 'Florida', '13': 'Georgia', '15': 'Hawaii',
    '16': 'Idaho', '17': 'Illinois', '18': 'Indiana', '19': 'Iowa',
    '20': 'Kansas', '21': 'Kentucky', '22': 'Louisiana', '23': 'Maine',
    '24': 'Maryland', '25': 'Massachusetts', '26': 'Michigan', '27': 'Minnesota',
    '28': 'Mississippi', '29': 'Missouri', '30': 'Montana', '31': 'Nebraska',
    '32': 'Nevada', '33': 'New Hampshire', '34': 'New Jersey', '35': 'New Mexico',
    '36': 'New York', '37': 'North Carolina', '38': 'North Dakota', '39': 'Ohio',
    '40': 'Oklahoma', '41': 'Oregon', '42': 'Pennsylvania', '44': 'Rhode Island',
    '45': 'South Carolina', '46': 'South Dakota', '47': 'Tennessee', '48': 'Texas',
    '49': 'Utah', '50': 'Vermont', '51': 'Virginia', '53': 'Washington',
    '54': 'West Virginia', '55': 'Wisconsin', '56': 'Wyoming',
}

# Build LAUS series IDs
# Format: LASST{FIPS}0000000000003  (unemployment rate)
# Note: LAUS uses a different format — state series are:
# LASST{FIPS}00000003 for unemployment rate
laus_series = {}
for fips, name in FIPS_STATES.items():
    sid = f"LASST{fips}0000000000003"
    laus_series[sid] = {'fips': int(fips), 'state': name}

series_ids = list(laus_series.keys())
all_laus = []

print(f"\n  Fetching {len(series_ids)} state series...")

for chunk_start, chunk_end in YEAR_CHUNKS:
    print(f"\n  Years {chunk_start}-{chunk_end}...")
    
    for batch_start in range(0, len(series_ids), MAX_SERIES):
        batch = series_ids[batch_start:batch_start + MAX_SERIES]
        
        try:
            results = bls_fetch(batch, chunk_start, chunk_end)
            
            for s in results:
                sid = s['seriesID']
                info = laus_series.get(sid, {})
                fips = info.get('fips', 0)
                state = info.get('state', 'Unknown')
                
                for obs in s['data']:
                    year = int(obs['year'])
                    period = obs['period']
                    if not period.startswith('M'):
                        continue
                    month = int(period[1:])
                    value = obs['value'].replace(',', '')
                    try:
                        value = float(value)
                    except ValueError:
                        continue
                    
                    all_laus.append({
                        'STATEFIP': fips,
                        'state_name': state,
                        'year': year,
                        'month': month,
                        'unemployment_rate': value,
                    })
            
            n = len([r for r in results])
            print(f"    Batch {batch_start//MAX_SERIES + 1}: {n} series")
            
        except Exception as e:
            print(f"    Error: {e}")
        
        time.sleep(1)

laus_df = pd.DataFrame(all_laus)
if len(laus_df) > 0:
    laus_df = laus_df.sort_values(['STATEFIP', 'year', 'month']).reset_index(drop=True)
    laus_path = os.path.join(DATA_DIR, "laus_state_unemployment.csv")
    laus_df.to_csv(laus_path, index=False)
    print(f"\nSaved LAUS: {laus_path}")
    print(f"  {len(laus_df):,} obs, {laus_df['STATEFIP'].nunique()} states, "
          f"{laus_df['year'].min()}-{laus_df['year'].max()}")
    
    # Quick sanity check
    print(f"\n  Sample (Alabama, Jan 2020):")
    sample = laus_df[(laus_df['STATEFIP'] == 1) & (laus_df['year'] == 2020) & (laus_df['month'] == 1)]
    if len(sample) > 0:
        print(f"    Unemployment rate: {sample['unemployment_rate'].values[0]}%")
else:
    print("\n  WARNING: No LAUS data retrieved!")
    print("  Falling back to Excel file...")
    print("  You can parse ststdnsadata.xlsx manually if the API fails.")


# ============================================================
# 3. Also need: state-level industry shares from CPS
#    This comes from the CPS microdata, not BLS API.
#    Print instructions for building Bartik shares.
# ============================================================
print("\n" + "=" * 70)
print("3. Bartik Shares — From CPS Microdata")
print("=" * 70)
print("""
The Bartik instrument requires state-level industry employment shares.
These come from your CPS extract (cps_00002.dat), which has IND1990.

To build the shares, you'll need to add industry share computation
to the cps_build_panel.py pipeline, or run a separate script that:
  1. Loads the CPS microdata
  2. For each state-year (or state-quarter), computes:
     ω_sj = Σ WTFINL[state=s, industry=j] / Σ WTFINL[state=s]
  3. Maps IND1990 codes to CES supersector categories
  4. Saves state_industry_shares.csv

The IND1990 → CES supersector mapping is:
  IND1990 codes → NAICS supersectors used in CES data

This is the most involved piece — we'll build it next.
""")

print("=" * 70)
print("DOWNLOAD COMPLETE")
print("=" * 70)

"""
CPS Data Pipeline: Load, Clean, Merge Task Scores, Aggregate
For: Macro Displacement / COVID Polarization Paper
Author: Ethan Johnston

Optimized with parallel processing and vectorized operations.
"""

import pandas as pd
import numpy as np
import os
import time
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
from paths import DATA_DIR

def _find_data(fname):
    """Locate a data file in the project root or the archived data folder.
    Returns None if it is in neither, so callers can treat it as optional."""
    for cand in (os.path.join(DATA_DIR, fname),
                 os.path.join(DATA_DIR, "archive", "old_data", fname)):
        if os.path.exists(cand):
            return cand
    return None

CPS_DAT      = os.path.join(DATA_DIR, "cps_00002.dat")
RTI_CSV      = os.path.join(DATA_DIR, "rti_by_occupation.csv")
CW_2010      = os.path.join(DATA_DIR, "2010-occ-codes-with-crosswalk-from-2002-2011.xls")
# BEA SAGDP/SAINC files feed only state_year_panel.csv (unused by the
# estimation pipeline). They may live in the project root or in
# archive/old_data/ after the repo reorganization — resolve either.
SAINC1_CSV   = _find_data("SAINC1__ALL_AREAS_1929_2024.csv")
SAGDP9_CSV   = _find_data("SAGDP9__ALL_AREAS_1997_2024.csv")
USRECQM_CSV  = os.path.join(DATA_DIR, "USRECQM.csv")

OUTPUT_DIR   = DATA_DIR
N_WORKERS    = max(1, cpu_count() - 2)  # Leave 2 cores free
CHUNK_SIZE   = 500_000

# Column layout from SAS syntax file (0-indexed for Python read_fwf)
COLSPECS = [
    (0, 4),      # YEAR        1-4
    (4, 9),      # SERIAL      5-9
    (9, 11),     # MONTH       10-11
    (11, 21),    # HWTFINL     12-21
    (21, 35),    # CPSID       22-35
    (35, 36),    # ASECFLAG    36
    (36, 37),    # HFLAG       37
    (37, 48),    # ASECWTH     38-48
    (48, 50),    # STATEFIP    49-50
    (50, 52),    # PERNUM      51-52
    (52, 66),    # WTFINL      53-66
    (66, 80),    # CPSIDP      67-80
    (80, 95),    # CPSIDV      81-95
    (95, 106),   # ASECWT      96-106
    (106, 110),  # RELATE      107-110
    (110, 112),  # AGE         111-112
    (112, 113),  # SEX         113
    (113, 116),  # RACE        114-116
    (116, 117),  # MARST       117
    (117, 118),  # POPSTAT     118
    (118, 120),  # ASIAN       119-120
    (120, 121),  # VETSTAT     121
    (121, 124),  # HISPAN      122-124
    (124, 126),  # EMPSTAT     125-126
    (126, 127),  # LABFORCE    127
    (127, 131),  # OCC         128-131
    (131, 135),  # OCC2010     132-135
    (135, 138),  # OCC1990     136-138
    (138, 141),  # IND1990     139-141
    (141, 144),  # OCC1950     142-144
    (144, 148),  # IND         145-148
    (148, 151),  # IND1950     149-151
]

COLNAMES = [
    'YEAR', 'SERIAL', 'MONTH', 'HWTFINL', 'CPSID', 'ASECFLAG', 'HFLAG',
    'ASECWTH', 'STATEFIP', 'PERNUM', 'WTFINL', 'CPSIDP', 'CPSIDV',
    'ASECWT', 'RELATE', 'AGE', 'SEX', 'RACE', 'MARST', 'POPSTAT',
    'ASIAN', 'VETSTAT', 'HISPAN', 'EMPSTAT', 'LABFORCE',
    'OCC', 'OCC2010', 'OCC1990', 'IND1990', 'OCC1950', 'IND', 'IND1950'
]

KEEP_COLS = [
    'YEAR', 'MONTH', 'STATEFIP', 'WTFINL', 'CPSIDV',
    'AGE', 'SEX', 'RACE', 'MARST', 'POPSTAT', 'HISPAN',
    'EMPSTAT', 'LABFORCE', 'OCC2010', 'OCC1990', 'IND1990', 'OCC1950'
]

NUMERIC_COLS = [
    'YEAR', 'MONTH', 'STATEFIP', 'AGE', 'SEX', 'RACE', 'MARST',
    'POPSTAT', 'HISPAN', 'EMPSTAT', 'LABFORCE',
    'OCC2010', 'OCC1990', 'IND1990', 'OCC1950'
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def timer(msg):
    """Simple timer context manager."""
    class Timer:
        def __init__(self, msg):
            self.msg = msg
        def __enter__(self):
            self.start = time.time()
            print(f"\n>> {self.msg}...", flush=True)
            return self
        def __exit__(self, *args):
            elapsed = time.time() - self.start
            print(f"   Done in {elapsed:.1f}s", flush=True)
    return Timer(msg)


def process_chunk(chunk_df):
    """Process a single chunk: keep columns, convert types, filter."""
    chunk = chunk_df[KEEP_COLS].copy()
    for col in NUMERIC_COLS:
        chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
    chunk['WTFINL'] = pd.to_numeric(chunk['WTFINL'], errors='coerce') / 10000
    chunk['CPSIDV'] = pd.to_numeric(chunk['CPSIDV'], errors='coerce')
    # Filter to civilian age 16+
    # POPSTAT is NaN in early CPS years, so use AGE + exclude Armed Forces (EMPSTAT==1)
    # This matches Jaimovich & Siu: "civilian non-institutionalized individuals aged 16+"
    age_ok = chunk['AGE'] >= 16
    not_military = chunk['EMPSTAT'] != 1  # EMPSTAT 01 = Armed Forces
    # If POPSTAT is available, use it; otherwise rely on age + not military
    if chunk['POPSTAT'].notna().any():
        civilian = (chunk['POPSTAT'] == 1) | chunk['POPSTAT'].isna()  # Keep if civilian OR missing
    else:
        civilian = not_military
    chunk = chunk[age_ok & civilian]
    return chunk


def weighted_mean_safe(values, weights):
    """Compute weighted mean, handling NaN."""
    mask = values.notna() & weights.notna() & (weights > 0)
    if mask.sum() == 0:
        return np.nan
    return np.average(values[mask], weights=weights[mask])


def aggregate_group_task(group_df):
    """Compute weighted task means and employment for a group."""
    w = group_df['WTFINL']
    n = len(group_df)
    return pd.Series({
        'mean_abstract': weighted_mean_safe(group_df['abstract'], w),
        'mean_routine': weighted_mean_safe(group_df['routine'], w),
        'mean_manual': weighted_mean_safe(group_df['manual'], w),
        'mean_rti': weighted_mean_safe(group_df['rti'], w),
        'total_employed': w.sum(),
        'n_obs': n,
    })


def aggregate_group_shares(group_df):
    """Compute J-S occupation group shares for a group."""
    total_w = group_df['WTFINL'].sum()
    if total_w == 0:
        return pd.Series({
            'share_NRC': np.nan, 'share_RC': np.nan,
            'share_NRM': np.nan, 'share_RM': np.nan,
            'share_routine': np.nan,
        })
    return pd.Series({
        'share_NRC': group_df.loc[group_df['js_group'] == 'NRC', 'WTFINL'].sum() / total_w,
        'share_RC': group_df.loc[group_df['js_group'] == 'RC', 'WTFINL'].sum() / total_w,
        'share_NRM': group_df.loc[group_df['js_group'] == 'NRM', 'WTFINL'].sum() / total_w,
        'share_RM': group_df.loc[group_df['js_group'] == 'RM', 'WTFINL'].sum() / total_w,
        'share_routine': group_df.loc[group_df['js_group'].isin(['RC', 'RM']), 'WTFINL'].sum() / total_w,
    })


def aggregate_state_chunk(args):
    """Aggregate a subset of state-months. For parallel processing."""
    chunk_df, min_obs = args
    results = []
    for (yr, mo, st), g in chunk_df.groupby(['YEAR', 'MONTH', 'STATEFIP']):
        task_g = g[g['abstract'].notna()]
        if len(task_g) < min_obs:
            row = {'YEAR': yr, 'MONTH': mo, 'STATEFIP': st,
                   'mean_abstract': np.nan, 'mean_routine': np.nan,
                   'mean_manual': np.nan, 'mean_rti': np.nan,
                   'total_employed': g['WTFINL'].sum(), 'n_obs': len(g)}
        else:
            w = task_g['WTFINL']
            row = {
                'YEAR': yr, 'MONTH': mo, 'STATEFIP': st,
                'mean_abstract': np.average(task_g['abstract'], weights=w),
                'mean_routine': np.average(task_g['routine'], weights=w),
                'mean_manual': np.average(task_g['manual'], weights=w),
                'mean_rti': np.average(task_g['rti'], weights=w),
                'total_employed': g['WTFINL'].sum(),
                'n_obs': len(g),
            }
        # J-S shares
        js_g = g[g['js_group'].notna()]
        tw = js_g['WTFINL'].sum()
        if tw > 0:
            row['share_routine'] = js_g.loc[js_g['js_group'].isin(['RC', 'RM']), 'WTFINL'].sum() / tw
            row['share_NRC'] = js_g.loc[js_g['js_group'] == 'NRC', 'WTFINL'].sum() / tw
            row['share_NRM'] = js_g.loc[js_g['js_group'] == 'NRM', 'WTFINL'].sum() / tw
            row['share_RC'] = js_g.loc[js_g['js_group'] == 'RC', 'WTFINL'].sum() / tw
            row['share_RM'] = js_g.loc[js_g['js_group'] == 'RM', 'WTFINL'].sum() / tw
        else:
            row.update({'share_routine': np.nan, 'share_NRC': np.nan,
                        'share_NRM': np.nan, 'share_RC': np.nan, 'share_RM': np.nan})
        results.append(row)
    return pd.DataFrame(results)


# ============================================================
# MAIN PIPELINE
# ============================================================

if __name__ == '__main__':
    pipeline_start = time.time()
    print("=" * 70)
    print(f"CPS MACRO DISPLACEMENT PIPELINE")
    print(f"Using {N_WORKERS} parallel workers")
    print("=" * 70)

    # ========================================================
    # STEP 1: Load CPS Fixed-Width Data
    # ========================================================
    with timer("STEP 1: Loading CPS data"):
        chunks = []
        reader = pd.read_fwf(
            CPS_DAT, colspecs=COLSPECS, names=COLNAMES,
            chunksize=CHUNK_SIZE, dtype=str
        )
        total_rows = 0
        for i, raw_chunk in enumerate(reader):
            processed = process_chunk(raw_chunk)
            chunks.append(processed)
            total_rows += len(raw_chunk)
            kept = len(processed)
            print(f"   Chunk {i+1}: {len(raw_chunk):,} read, {kept:,} kept "
                  f"(running total: {total_rows:,})", flush=True)

        cps = pd.concat(chunks, ignore_index=True)
        del chunks
        print(f"   Final CPS sample: {len(cps):,} person-month obs")
        print(f"   Year range: {int(cps['YEAR'].min())} - {int(cps['YEAR'].max())}")

    # ========================================================
    # STEP 2: Derive Variables (vectorized, no .apply)
    # ========================================================
    with timer("STEP 2: Deriving variables (vectorized)"):
        # Date
        cps['date'] = pd.to_datetime(
            cps['YEAR'].astype(int).astype(str) + '-' +
            cps['MONTH'].astype(int).astype(str).str.zfill(2) + '-01'
        )

        # Employment: EMPSTAT 10=At work, 12=Has job not at work
        cps['employed'] = cps['EMPSTAT'].isin([10, 12]).astype(np.int8)

        # Demographics (vectorized with np.select)
        cps['female'] = (cps['SEX'] == 2).astype(np.int8)
        cps['hispanic'] = (cps['HISPAN'] > 0).astype(np.int8)
        cps['race_cat'] = np.select(
            [cps['RACE'] == 100, cps['RACE'] == 200,
             cps['RACE'].isin([650, 651, 652])],
            ['White', 'Black', 'Asian'],
            default='Other'
        )

        # Jaimovich-Siu occupation groups (vectorized with np.select)
        occ = cps['OCC2010']
        cps['js_group'] = np.select(
            [
                (occ >= 10) & (occ <= 3540),                              # NRC
                ((occ >= 4700) & (occ <= 4965)) | ((occ >= 5000) & (occ <= 5940)),  # RC
                (occ >= 3600) & (occ <= 4650),                             # NRM
                (occ >= 6005) & (occ <= 9750),                             # RM
                (occ >= 9800) & (occ <= 9830),                             # MIL
            ],
            ['NRC', 'RC', 'NRM', 'RM', 'MIL'],
            default=''
        )
        cps.loc[cps['js_group'] == '', 'js_group'] = np.nan

        print(f"   Employment rate: {cps['employed'].mean():.3f}")
        print(f"   Unique OCC2010 codes: {cps['OCC2010'].nunique()}")
        emp = cps[cps['employed'] == 1]
        print(f"   J-S group distribution (employed):")
        print(f"   {emp['js_group'].value_counts(dropna=False).to_string()}")

    # ========================================================
    # STEP 3: Build OCC2010 → Task Score Crosswalk
    # ========================================================
    with timer("STEP 3: Building crosswalk"):
        rti = pd.read_csv(RTI_CSV)
        print(f"   RTI file: {len(rti)} O*NET occupations")

        # Parse 2010 Census → SOC crosswalk
        # Structure: [empty, description, census_code, soc_code] with header=None
        xls_2010 = pd.read_excel(CW_2010, header=None)
        xls_2010.columns = ['empty', 'description', 'census_code', 'soc_code']

        xls_2010['census_str'] = xls_2010['census_code'].astype(str).str.strip()
        mask_2010 = (
            xls_2010['census_str'].str.match(r'^\d+$', na=False) &
            ~xls_2010['census_str'].str.contains('-', na=True) &
            xls_2010['soc_code'].astype(str).str.strip().str.match(r'^\d{2}-\d{4}', na=False)
        )
        cw = xls_2010.loc[mask_2010, ['description', 'census_code', 'soc_code']].copy()
        cw['census_2010'] = pd.to_numeric(cw['census_code'], errors='coerce').astype(int)
        cw['soc_code'] = cw['soc_code'].astype(str).str.strip()
        cw = cw[['description', 'census_2010', 'soc_code']].reset_index(drop=True)
        cw['onet_soc'] = cw['soc_code'].apply(
            lambda x: x + '.00' if '.' not in str(x) else str(x)
        )
        print(f"   Crosswalk parsed: {len(cw)} mappings")

        # Build SOC → task score lookup with prefix-averaging
        rti['soc_6'] = rti['onet_soc'].str.replace(r'\.\d+$', '', regex=True)
        rti_by_soc = rti.groupby('soc_6')[['abstract', 'routine', 'manual', 'rti']].mean()
        rti_lookup = rti_by_soc.to_dict('index')

        def match_soc(soc_code):
            soc_clean = str(soc_code).strip()
            if len(soc_clean) < 7:
                return np.nan, np.nan, np.nan, np.nan
            soc_7 = soc_clean[:7]
            if soc_7 in rti_lookup:
                r = rti_lookup[soc_7]
                return r['abstract'], r['routine'], r['manual'], r['rti']
            for plen in [6, 5, 4]:
                prefix = soc_7[:plen]
                matches = rti[rti['soc_6'].str.startswith(prefix)]
                if len(matches) > 0:
                    m = matches[['abstract', 'routine', 'manual', 'rti']].mean()
                    return m['abstract'], m['routine'], m['manual'], m['rti']
            return np.nan, np.nan, np.nan, np.nan

        cw[['abstract', 'routine', 'manual', 'rti']] = pd.DataFrame(
            cw['soc_code'].apply(match_soc).tolist(), index=cw.index
        )

        matched = cw['abstract'].notna().sum()
        print(f"   Matched: {matched}/{len(cw)} ({matched/len(cw):.1%})")

        unmatched = cw[cw['abstract'].isna()]
        if len(unmatched) > 0:
            print(f"   Unmatched ({len(unmatched)}):")
            for _, row in unmatched.head(10).iterrows():
                print(f"     Census {row['census_2010']}: {row['soc_code']} - {row['description']}")

        # Final lookup: census_2010 → task scores (average if multiple SOC per census code)
        occ_task = cw.groupby('census_2010')[['abstract', 'routine', 'manual', 'rti']].mean().reset_index()
        occ_task.rename(columns={'census_2010': 'OCC2010'}, inplace=True)
        print(f"   Final mapping: {len(occ_task)} OCC2010 codes with task scores")

    # ========================================================
    # STEP 4: Merge Task Scores onto CPS (vectorized merge)
    # ========================================================
    with timer("STEP 4: Merging task scores"):
        cps = cps.merge(occ_task, on='OCC2010', how='left')
        has_task = cps['abstract'].notna() & (cps['employed'] == 1)
        print(f"   Employed with task scores: {has_task.sum():,} / "
              f"{cps['employed'].sum():,} ({has_task.sum()/cps['employed'].sum():.1%})")

    # ========================================================
    # STEP 5: National-Month Aggregation
    # ========================================================
    with timer("STEP 5: National-month aggregation"):
        employed = cps[cps['employed'] == 1].copy()

        # Task content means
        national_task = employed.groupby(['YEAR', 'MONTH']).apply(
            aggregate_group_task
        ).reset_index()

        # J-S shares
        js_emp = employed[employed['js_group'].notna()]
        national_shares = js_emp.groupby(['YEAR', 'MONTH']).apply(
            aggregate_group_shares
        ).reset_index()

        national_month = national_task.merge(national_shares, on=['YEAR', 'MONTH'], how='left')
        national_month['date'] = pd.to_datetime(
            national_month['YEAR'].astype(int).astype(str) + '-' +
            national_month['MONTH'].astype(int).astype(str).str.zfill(2) + '-01'
        )
        print(f"   National-month panel: {len(national_month)} rows")
        print(f"   Date range: {national_month['date'].min()} to {national_month['date'].max()}")

    # ========================================================
    # STEP 6: State-Month Aggregation (PARALLEL)
    # ========================================================
    with timer(f"STEP 6: State-month aggregation ({N_WORKERS} workers)"):
        # Split employed data into chunks by STATEFIP for parallel processing
        states = sorted(employed['STATEFIP'].unique())
        n_per_worker = max(1, len(states) // N_WORKERS)
        state_chunks = [states[i:i+n_per_worker] for i in range(0, len(states), n_per_worker)]

        chunk_args = [
            (employed[employed['STATEFIP'].isin(sc)].copy(), 10)
            for sc in state_chunks
        ]

        print(f"   Distributing {len(states)} states across {len(chunk_args)} chunks...")

        with Pool(N_WORKERS) as pool:
            results = pool.map(aggregate_state_chunk, chunk_args)

        state_month = pd.concat(results, ignore_index=True)
        state_month['date'] = pd.to_datetime(
            state_month['YEAR'].astype(int).astype(str) + '-' +
            state_month['MONTH'].astype(int).astype(str).str.zfill(2) + '-01'
        )

        print(f"   State-month panel: {len(state_month)} rows")
        print(f"   States: {state_month['STATEFIP'].nunique()}")
        print(f"   Avg obs per state-month: {state_month['n_obs'].mean():.0f}")

    # ========================================================
    # STEP 7: State-Year Aggregation
    # ========================================================
    with timer("STEP 7: State-year aggregation"):
        state_year = state_month.groupby(['YEAR', 'STATEFIP']).agg({
            'mean_abstract': 'mean',
            'mean_routine': 'mean',
            'mean_manual': 'mean',
            'mean_rti': 'mean',
            'total_employed': 'mean',
            'n_obs': 'sum',
            'share_routine': 'mean',
            'share_NRC': 'mean',
            'share_NRM': 'mean',
        }).reset_index()
        print(f"   State-year panel: {len(state_year)} rows")

    # ========================================================
    # STEP 8: NBER Recession Dates
    # ========================================================
    with timer("STEP 8: Loading NBER recession dates"):
        rec = pd.read_csv(USRECQM_CSV)
        rec['date'] = pd.to_datetime(rec['observation_date'])
        rec.rename(columns={'USRECQM': 'recession'}, inplace=True)
        rec = rec[['date', 'recession']]
        national_month = national_month.merge(rec, on='date', how='left')

        trough_dates = {
            1975: '1975-03-01', 1982: '1982-11-01', 1991: '1991-03-01',
            2001: '2001-11-01', 2009: '2009-06-01', 2020: '2020-04-01',
        }
        print("   Trough dates:")
        for yr, dt in trough_dates.items():
            print(f"     {yr}: {dt}")

    # ========================================================
    # STEP 9: BEA State GDP and Population (optional)
    # ========================================================
    # These columns land only in state_year_panel.csv, which no estimation
    # script consumes. A missing BEA file is therefore a warning, not a
    # fatal error.
    if SAGDP9_CSV and SAINC1_CSV:
        print("\n>> STEP 9: Loading BEA state data...")
        # State Annual Real GDP (from bulk download ALL_AREAS file)
        # Columns: GeoFIPS, GeoName, Region, TableName, LineCode, IndustryClassification, Description, Unit, 1997, 1998, ...
        sagdp = pd.read_csv(SAGDP9_CSV)
        print(f"   SAGDP9 raw shape: {sagdp.shape}")
        
        # Filter to "All industry total" (LineCode == 1)
        sagdp_total = sagdp[sagdp['LineCode'] == 1].copy()
        print(f"   LineCode==1 rows: {len(sagdp_total)}")
        
        # Identify year columns (everything that's a 4-digit number)
        meta_cols = ['GeoFIPS', 'GeoName', 'Region', 'TableName', 'LineCode', 
                     'IndustryClassification', 'Description', 'Unit']
        year_cols = [c for c in sagdp_total.columns if c not in meta_cols]
        
        sagdp_long = sagdp_total.melt(
            id_vars=['GeoFIPS', 'GeoName'], value_vars=year_cols,
            var_name='YEAR', value_name='real_gdp'
        )
        sagdp_long['YEAR'] = pd.to_numeric(sagdp_long['YEAR'], errors='coerce')
        sagdp_long['real_gdp'] = pd.to_numeric(
            sagdp_long['real_gdp'].astype(str).str.replace(',', '').str.strip(), 
            errors='coerce'
        )
        
        # Extract state FIPS from GeoFIPS (5-digit: "01000" → 01)
        sagdp_long['GeoFIPS_clean'] = sagdp_long['GeoFIPS'].astype(str).str.strip().str.replace('"', '').str.replace("'", "")
        sagdp_long['STATEFIP'] = pd.to_numeric(sagdp_long['GeoFIPS_clean'].str[:2], errors='coerce')
        
        sagdp_long = sagdp_long.dropna(subset=['YEAR', 'STATEFIP', 'real_gdp'])
        sagdp_long['STATEFIP'] = sagdp_long['STATEFIP'].astype(int)
        sagdp_long['YEAR'] = sagdp_long['YEAR'].astype(int)
        sagdp_long = sagdp_long[(sagdp_long['STATEFIP'] >= 1) & (sagdp_long['STATEFIP'] <= 56)]
        sagdp_long = sagdp_long[['YEAR', 'STATEFIP', 'real_gdp']]
        print(f"   State GDP: {len(sagdp_long)} rows, {sagdp_long['STATEFIP'].nunique()} states")
        print(f"   Year range: {sagdp_long['YEAR'].min()} - {sagdp_long['YEAR'].max()}")

        # State Population from SAINC1 (from bulk download ALL_AREAS file)
        sainc = pd.read_csv(SAINC1_CSV)
        print(f"\n   SAINC1 raw shape: {sainc.shape}")
        print(f"   LineCode values: {sorted(sainc['LineCode'].unique()[:10])}")
        
        # Population is LineCode == 2
        pop_rows = sainc[sainc['LineCode'] == 2].copy()
        if len(pop_rows) == 0:
            # Fallback: search by description
            pop_rows = sainc[sainc['Description'].str.contains('Population', case=False, na=False)].copy()
        print(f"   Population rows (LineCode==2): {len(pop_rows)}")

        pop_meta_cols = ['GeoFIPS', 'GeoName', 'Region', 'TableName', 'LineCode',
                         'IndustryClassification', 'Description', 'Unit']
        pop_year_cols = [c for c in pop_rows.columns if c not in pop_meta_cols]
        
        pop_long = pop_rows.melt(
            id_vars=['GeoFIPS', 'GeoName'], value_vars=pop_year_cols,
            var_name='YEAR', value_name='population'
        )
        pop_long['YEAR'] = pd.to_numeric(pop_long['YEAR'], errors='coerce')
        pop_long['population'] = pd.to_numeric(
            pop_long['population'].astype(str).str.replace(',', '').str.strip(),
            errors='coerce'
        )
        
        pop_long['GeoFIPS_clean'] = pop_long['GeoFIPS'].astype(str).str.strip().str.replace('"', '').str.replace("'", "")
        pop_long['STATEFIP'] = pd.to_numeric(pop_long['GeoFIPS_clean'].str[:2], errors='coerce')
        
        pop_long = pop_long.dropna(subset=['YEAR', 'STATEFIP', 'population'])
        pop_long['STATEFIP'] = pop_long['STATEFIP'].astype(int)
        pop_long['YEAR'] = pop_long['YEAR'].astype(int)
        pop_long = pop_long[(pop_long['STATEFIP'] >= 1) & (pop_long['STATEFIP'] <= 56)]
        pop_long = pop_long[['YEAR', 'STATEFIP', 'population']]
        print(f"   Population: {len(pop_long)} rows, {pop_long['STATEFIP'].nunique()} states")
        print(f"   Year range: {pop_long['YEAR'].min()} - {pop_long['YEAR'].max()}")

        # Merge onto state-year panel
        state_year = state_year.merge(sagdp_long, on=['YEAR', 'STATEFIP'], how='left')
        state_year = state_year.merge(pop_long, on=['YEAR', 'STATEFIP'], how='left')
        
        # Per-capita GDP: real_gdp is in millions of dollars, population is in persons
        state_year['gdp_pc'] = np.where(
            state_year['population'] > 0,
            state_year['real_gdp'] * 1_000_000 / state_year['population'],
            np.nan
        )
        print(f"\n   GDP coverage: {state_year['real_gdp'].notna().mean():.1%}")
        print(f"   Population coverage: {state_year['population'].notna().mean():.1%}")
    else:
        print("\n>> STEP 9: BEA SAGDP/SAINC files not found in the project "
              "root or archive/old_data — skipping. real_gdp, population, and "
              "gdp_pc will be absent from state_year_panel.csv; no estimation "
              "step uses them, so this does not affect the paper.")

    # ========================================================
    # STEP 10: Save Outputs
    # ========================================================
    with timer("STEP 10: Saving output files"):
        files = {
            'national_month_panel.csv': national_month,
            'state_month_panel.csv': state_month,
            'state_year_panel.csv': state_year,
            'occ2010_task_scores.csv': occ_task,
        }
        for fname, df in files.items():
            path = os.path.join(OUTPUT_DIR, fname)
            df.to_csv(path, index=False)
            print(f"   {fname}: {len(df):,} rows")

    # ========================================================
    # STEP 10b: Inline data-construction stats for the manuscript
    # ========================================================
    # The Data section of the paper cites the extract size, the employed
    # analysis-sample size, and the CPS coverage window. Emit them here as
    # \stats... macros so none of those numbers is hand-typed into the LaTeX.
    with timer("STEP 10b: Writing data-construction stats"):
        n_extract  = int(total_rows)        # raw IPUMS extract person-months
        n_employed = int(has_task.sum())    # employed obs with a valid occupation/task score
        d0 = national_month['date'].min()
        d1 = national_month['date'].max()
        stats_cps = (
            "% Auto-generated by cps_build_panel.py — do not edit manually.\n"
            f"\\newcommand{{\\statsCpsExtractMillions}}{{{round(n_extract / 1e6)}}}\n"
            f"\\newcommand{{\\statsCpsEmployedMillions}}{{{round(n_employed / 1e6)}}}\n"
            f"\\newcommand{{\\statsCpsStart}}{{{d0.strftime('%B %Y')}}}\n"
            f"\\newcommand{{\\statsCpsEnd}}{{{d1.strftime('%B %Y')}}}\n"
            f"\\newcommand{{\\statsCpsStartYear}}{{{d0.year}}}\n"
            f"\\newcommand{{\\statsCpsEndYear}}{{{d1.year}}}\n"
        )
        cps_stats_path = os.path.join(DATA_DIR, "paper", "stats_cps.tex")
        os.makedirs(os.path.dirname(cps_stats_path), exist_ok=True)
        with open(cps_stats_path, 'w', encoding='utf-8') as f:
            f.write(stats_cps)
        print(f"   stats_cps.tex: extract={n_extract:,}, employed={n_employed:,}")

    # ========================================================
    # SUMMARY
    # ========================================================
    total_time = time.time() - pipeline_start
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"\nNational-month panel: {len(national_month)} rows "
          f"({national_month['date'].min().strftime('%Y-%m')} to {national_month['date'].max().strftime('%Y-%m')})")
    print(f"State-month panel:   {len(state_month):,} rows ({state_month['STATEFIP'].nunique()} states)")
    print(f"State-year panel:    {len(state_year):,} rows")
    print(f"Task score coverage: {occ_task['abstract'].notna().sum()}/{len(occ_task)} occupations")

    # Quick sanity checks
    print(f"\nSanity checks:")
    early = national_month[national_month['YEAR'] <= national_month['YEAR'].min() + 4]
    late = national_month[national_month['YEAR'] >= national_month['YEAR'].max() - 4]
    print(f"  Mean RTI early ({int(early['YEAR'].min())}-{int(early['YEAR'].max())}): {early['mean_rti'].mean():.4f}")
    print(f"  Mean RTI late  ({int(late['YEAR'].min())}-{int(late['YEAR'].max())}):  {late['mean_rti'].mean():.4f}")
    print(f"  Routine share early: {early['share_routine'].mean():.3f}")
    print(f"  Routine share late:  {late['share_routine'].mean():.3f}")
    if early['mean_rti'].mean() > late['mean_rti'].mean():
        print(f"  ✓ RTI declining over time (expected from de-routinization)")
    if early['share_routine'].mean() > late['share_routine'].mean():
        print(f"  ✓ Routine share declining over time (expected from polarization)")
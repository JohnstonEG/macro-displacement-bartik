# Cyclical Demand Shocks and Aggregate Task Composition

Replication package for **"Cyclical Demand Shocks and Aggregate Task Composition: Evidence from a Bartik Instrument"** (Ethan Johnston, Department of Economics, Illinois State University).

The paper estimates the effect of cyclical labor-demand shocks on the task composition of US employment. It builds a state-year panel of task content from CPS microdata and estimates local projections instrumented with a Bartik shift-share predictor, recovering the impulse response of aggregate task composition over a five-year horizon.

## Repository structure

```
.
├── code/                   Python analysis pipeline (see Replication below)
├── Tex/
│   ├── manuscript.tex       Labour Economics submission (Elsevier cas-sc class)
│   ├── references.bib
│   └── cas-sc.cls, cas-common.sty, cas-model2-names.bst
├── paper/
│   ├── tables/              generated LaTeX tables (\input by manuscript.tex)
│   ├── figures/             generated figures
│   ├── stats.tex            generated inline-statistic macros
│   └── stats.json
├── results/robustness/     estimation output for the robustness/extension sections
├── requirements.txt
├── organize_repo.ps1       one-time folder-reorganization script
└── README.md
```

The raw IPUMS-CPS microdata extracts, the local `archive/` of superseded code,
reference PDFs, and LaTeX build artifacts are excluded from version control —
see `.gitignore`. The constructed analysis panels and the public BLS, O*NET,
and FRED inputs are tracked, so the estimation steps can be run directly.

## Data

All inputs come from public sources. The constructed analysis panels and the
public BLS, O*NET, FRED, and Census inputs are included in this repository, so
the estimation can be reproduced directly. The raw IPUMS-CPS microdata extracts
are **not** included — they are too large for GitHub — but the IPUMS extract
codebooks are provided so the extracts can be reconstructed at no cost from
cps.ipums.org. To rebuild the panels from scratch, recreate the extracts, place
the `.dat` files in the repository root, and run the Stage 1 scripts.

| Source | Used for | How to obtain |
|---|---|---|
| IPUMS-CPS basic monthly, 1976–2026 | task-composition panel, Bartik industry shares | cps.ipums.org (extract `cps_00002`) |
| IPUMS-CPS Displaced Worker Supplements, 1984–2024 | DWS descriptive evidence | cps.ipums.org (extract `cps_00003`; codebook `cps_00003.cbk.pdf` is included) |
| O*NET Work Activities & Work Context | occupation-level task scores | onetcenter.org |
| BLS Current Employment Statistics (CES) | Bartik national industry shocks | bls.gov / BLS public API |
| BLS Local Area Unemployment Statistics (LAUS) | state unemployment rates | bls.gov |
| FRED `USRECQM` | NBER recession dating | fred.stlouisfed.org |
| Census occupation crosswalks | OCC2010 ↔ SOC mapping | census.gov |

## Software

Python 3.10+ with the packages in `requirements.txt`
(`pip install -r requirements.txt`): pandas, numpy, matplotlib, scipy,
linearmodels, pyarrow, openpyxl. Compiling the paper requires a LaTeX
distribution (TeX Live or MiKTeX).

## Replication

Each script defines an absolute project path (`DATA_DIR`) at the top of the
file. Update it if you relocate the repository, then run the stages in order.

**Stage 1 — Data preparation**

- `bls_download.py` — BLS CES national industry employment (and LAUS via API)
- `parse_laus.py` — state unemployment from the BLS LAUS spreadsheet
- `cps_build_panel.py` — state task-composition panels from CPS microdata
- `bartik_shares.py` — state industry shares and the Bartik instrument
- `dws_build_panel.py` — Displaced Worker Survey panel

**Stage 2 — Estimation**

- `lp_iv_annual.py` — main annual LP-IV specification
- `lp_iv_robustness.py` — pre-trend placebos, composition decomposition, Rotemberg weights, Anderson-Rubin
- `mechanism_tests.py` — within- vs between-industry, labor-force participation
- `additional_robustness.py`, `extended_robustness.py`, `extensions.py`, `final_extensions.py` — robustness and heterogeneity
- `dws_analysis.py` — DWS post-displacement destinations

**Stage 3 — Paper outputs**

- `paper_outputs.py` — main tables, figures, and `stats.tex`
- `supp_tables.py`, `ext_tables.py` — supplementary and extension tables
- `patch_tables.py` — fits wide tables within the page margins

**Stage 4 — Compile the paper**

```
cd Tex
pdflatex manuscript
bibtex   manuscript
pdflatex manuscript
pdflatex manuscript
```

## Contact

Ethan Johnston — Department of Economics, Illinois State University — egjohn3@ilstu.edu

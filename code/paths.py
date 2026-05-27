"""
Centralized path configuration for the Macro Displacement replication pipeline.
==============================================================================

All scripts in code/ import their data, paper, and results paths from this
module so that the project can be relocated to any directory or drive without
editing the individual scripts.

Resolution order for the data root:
  1. The MACRODISP_DATA environment variable, if set
  2. The repository root (the parent of the directory containing this file)
  3. Otherwise: raise with a helpful message

To override the data root from a non-standard location:
  PowerShell:  $env:MACRODISP_DATA = "D:\\Data\\MacroDisplacement"
  bash/zsh:    export MACRODISP_DATA=/path/to/data

Exported constants (all strings, so they remain compatible with os.path.join,
pandas read_csv, open(), etc.):

  DATA_DIR        Project root
  CODE_DIR        DATA_DIR/code
  PAPER_DIR       DATA_DIR/paper
  TABLES_DIR      DATA_DIR/paper/tables
  FIGURES_DIR     DATA_DIR/paper/figures
  TEX_DIR         DATA_DIR/Tex
  RESULTS_DIR     DATA_DIR/results
  ROBUSTNESS_DIR  DATA_DIR/results/robustness

Author: Ethan Johnston
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve the project root
# ---------------------------------------------------------------------------
_env = os.environ.get("MACRODISP_DATA")
if _env:
    _root = Path(_env).expanduser().resolve()
else:
    # paths.py lives in <project_root>/code/; the project root is its parent
    _root = Path(__file__).resolve().parents[1]

# Sanity check: confirm this looks like the project tree, not some random
# directory the env var was pointed at by mistake.
if not (_root / "code").is_dir():
    raise RuntimeError(
        f"Could not locate the Macro Displacement project root at '{_root}'.\n"
        "Either run scripts from inside the project, or set the "
        "MACRODISP_DATA environment variable to override:\n"
        '    PowerShell:  $env:MACRODISP_DATA = "D:\\\\Data\\\\MacroDisplacement"\n'
        '    bash/zsh:    export MACRODISP_DATA=/path/to/data'
    )

# ---------------------------------------------------------------------------
# Exported path constants
# ---------------------------------------------------------------------------
DATA_DIR        = str(_root)
CODE_DIR        = os.path.join(DATA_DIR, "code")
PAPER_DIR       = os.path.join(DATA_DIR, "paper")
TABLES_DIR      = os.path.join(PAPER_DIR, "tables")
FIGURES_DIR     = os.path.join(PAPER_DIR, "figures")
TEX_DIR         = os.path.join(DATA_DIR, "Tex")
RESULTS_DIR     = os.path.join(DATA_DIR, "results")
ROBUSTNESS_DIR  = os.path.join(RESULTS_DIR, "robustness")

# run_all.py

import subprocess
import sys
from pathlib import Path

# Folder containing your Python scripts
CODE_DIR = Path(r"D:\Data\MacroDisplacement\code")

scripts = [
    "cps_build_panel.py",
    "dws_analysis.py",
    "paper_outputs.py",
    "supp_tables.py",
]

for script in scripts:
    script_path = CODE_DIR / script

    print(f"\nRunning: {script_path}")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=CODE_DIR
    )

    if result.returncode != 0:
        print(f"\nError while running {script}")
        sys.exit(result.returncode)

print("\nAll scripts completed successfully.")
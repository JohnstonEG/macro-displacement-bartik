"""
Patch wide LaTeX tables to fit within page margins.
Adds \\resizebox{\\textwidth}{!}{...} around the tabular environment.
"""

import os
import re

from paths import TABLES_DIR as TABLE_DIR

# Tables that need resizing
WIDE_TABLES = ['tab_lpiv_main.tex', 'tab_composition.tex', 'tab_loo.tex']

for fname in WIDE_TABLES:
    path = os.path.join(TABLE_DIR, fname)
    if not os.path.exists(path):
        print(f"  {fname}: not found — skipping")
        continue
    
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already patched
    if '\\resizebox' in content:
        print(f"  {fname}: already patched")
        continue
    
    # Wrap the tabular environment in resizebox
    # Replace \begin{tabular} with \resizebox{\textwidth}{!}{\begin{tabular}
    # Replace the LAST \end{tabular} with \end{tabular}}
    
    content = content.replace(
        '\\begin{tabular}',
        '\\resizebox{\\textwidth}{!}{\\begin{tabular}',
        1  # only first occurrence
    )
    
    # Find last \end{tabular} and add closing brace
    last_pos = content.rfind('\\end{tabular}')
    if last_pos >= 0:
        content = content[:last_pos] + '\\end{tabular}}' + content[last_pos + len('\\end{tabular}'):]
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"  {fname}: patched with resizebox")

print("\nDone. Recompile LaTeX.")

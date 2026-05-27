# ============================================================
#  organize_repo.ps1
#  One-time reorganization of D:\Data\MacroDisplacement into a
#  clean, GitHub-ready layout.
#
#  WHAT IT DOES
#    code\      <-  the 17 current analysis-pipeline scripts
#    archive\   <-  9 superseded scripts + old data / figures /
#                   results / LaTeX drafts
#    Tex\       <-  receives copies of the 3 Elsevier class files
#
#  SAFETY
#    This script only MOVES and COPIES files inside the project
#    folder. It DELETES nothing. Review it before running.
#    Re-running is safe: missing items are reported and skipped.
#
#  TO RUN (from a PowerShell window):
#    cd D:\Data\MacroDisplacement
#    powershell -ExecutionPolicy Bypass -File .\organize_repo.ps1
# ============================================================

$root = "D:\Data\MacroDisplacement"
if (-not (Test-Path $root)) { Write-Error "Project folder not found: $root"; exit 1 }
Set-Location $root

function New-Dir($path) {
    if (-not (Test-Path $path)) { New-Item -ItemType Directory -Path $path | Out-Null }
}

function Move-Safe($relSource, $destDir) {
    $src = Join-Path $root $relSource
    if (Test-Path -LiteralPath $src) {
        Move-Item -LiteralPath $src -Destination $destDir -Force
        Write-Host "  moved   $relSource"
    } else {
        Write-Host "  skip    $relSource  (not found)"
    }
}

Write-Host ""
Write-Host "=== 1. Creating folder structure ==="
New-Dir "$root\code"
New-Dir "$root\archive"
New-Dir "$root\archive\old_scripts"
New-Dir "$root\archive\old_data"
New-Dir "$root\archive\old_figures"
New-Dir "$root\archive\old_results"
New-Dir "$root\archive\old_tex"

Write-Host ""
Write-Host "=== 2. Current pipeline scripts -> code\ ==="
$currentScripts = @(
    "bls_download.py", "parse_laus.py", "cps_build_panel.py", "bartik_shares.py",
    "dws_build_panel.py", "lp_iv_annual.py", "lp_iv_robustness.py", "mechanism_tests.py",
    "additional_robustness.py", "extended_robustness.py", "extensions.py",
    "final_extensions.py", "dws_analysis.py", "paper_outputs.py", "supp_tables.py",
    "ext_tables.py", "patch_tables.py"
)
foreach ($f in $currentScripts) { Move-Safe $f "$root\code" }

Write-Host ""
Write-Host "=== 3. Superseded scripts -> archive\old_scripts\ ==="
# extract.py            -> downloaded unused BEA data
# national_event_study  -> phase 1 (national descriptive event study)
# state_twfe_*          -> phase 2 (state TWFE / event study)
# lp_iv.py              -> quarterly LP-IV, superseded by lp_iv_annual.py
# test.py               -> scratch
# diagnose_cps_link /   -> abandoned worker-level micro analysis
#   run_micro_*
$oldScripts = @(
    "extract.py", "national_event_study.py", "state_twfe_panel.py", "state_twfe_v3.py",
    "lp_iv.py", "test.py", "diagnose_cps_link.py", "run_micro_analysis.py",
    "run_micro_separation.py"
)
foreach ($f in $oldScripts) { Move-Safe $f "$root\archive\old_scripts" }

Write-Host ""
Write-Host "=== 4. Superseded data -> archive\old_data\ ==="
# BEA GDP/income series (not used in the paper), quarterly panels
# (superseded by the annual specification), seasonally-adjusted LAUS.
$oldData = @(
    "SAGDP9.csv", "SQGDP9.csv", "SAINC1.csv", "SAGDP.zip", "SAINC.zip",
    "SAGDP9__ALL_AREAS_1997_2024.csv", "SAGDP1__ALL_AREAS_1997_2024.csv",
    "SAINC11__ALL_AREAS_1999_2024.csv", "SAINC12__ALL_AREAS_1930_2024.csv",
    "SAINC1__ALL_AREAS_1929_2024.csv", "national_month_panel.csv",
    "state_quarter_panel.csv", "state_industry_shares_quarterly.csv",
    "bartik_instrument_quarterly.csv", "ststdsadata.xlsx"
)
foreach ($f in $oldData) { Move-Safe $f "$root\archive\old_data" }

Write-Host ""
Write-Host "=== 5. Superseded figures -> archive\old_figures\ ==="
if (Test-Path "$root\figures") {
    Get-ChildItem -LiteralPath "$root\figures" -Force | ForEach-Object {
        Move-Item -LiteralPath $_.FullName -Destination "$root\archive\old_figures" -Force
        Write-Host "  moved   figures\$($_.Name)"
    }
    Remove-Item -LiteralPath "$root\figures" -Force -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "=== 6. Superseded results -> archive\old_results\ ==="
$oldResults = @(
    "fig_event_coeff_separate.pdf", "twfe_coefficients.csv", "fig_coeff_separate.pdf",
    "fig_coeff_pooled.pdf", "fig_2009_vs_2020.pdf", "twfe_coefficients_v2.csv",
    "table_rti.tex", "table_abstract.tex", "table_routine.tex", "table_manual.tex",
    "fig_es_quarterly_all.pdf", "fig_es_monthly_all.pdf", "fig_es_quarterly_2009v2020.pdf",
    "event_study_coefficients_v3.csv", "dynamic_panel_results.csv",
    "fig_es_quarterly_v4.pdf", "fig_es_monthly_v4.pdf", "fig_es_quarterly_2009v2020_v4.pdf",
    "event_study_coefficients_v4.csv", "dynamic_panel_detrended_v4.csv",
    "fig_lp_iv_impulse_response.pdf", "fig_lp_iv_only.pdf", "lp_iv_results.csv",
    "first_stage_results.txt", "table_lp_iv.tex"
)
foreach ($f in $oldResults) { Move-Safe "results\$f" "$root\archive\old_results" }
if (Test-Path "$root\results\micro") {
    Move-Item -LiteralPath "$root\results\micro" -Destination "$root\archive\old_results\micro" -Force
    Write-Host "  moved   results\micro\"
}

Write-Host ""
Write-Host "=== 7. Abandoned worker-level micro outputs -> archive\old_results\ ==="
$oldMicro = @(
    "paper\tables\tab_micro_landing.tex", "paper\tables\tab_micro_main.tex",
    "paper\tables\tab_micro_robustness.tex", "paper\tables\tab_micro_separation.tex",
    "paper\figures\fig_micro_transitions.pdf"
)
foreach ($f in $oldMicro) { Move-Safe $f "$root\archive\old_results" }

Write-Host ""
Write-Host "=== 8. Superseded LaTeX drafts -> archive\old_tex\ ==="
# main.* is the pre-conversion (article-class) draft; manuscript.tex
# is the current Labour Economics submission and is kept in Tex\.
$texPatterns = @("main.*", "main_macro.pdf", "preliminary.*", "final.*", "final_inline.*", "presentation.*")
foreach ($pat in $texPatterns) {
    Get-ChildItem -Path "$root\Tex" -Filter $pat -File -ErrorAction SilentlyContinue | ForEach-Object {
        Move-Item -LiteralPath $_.FullName -Destination "$root\archive\old_tex" -Force
        Write-Host "  moved   Tex\$($_.Name)"
    }
}
# Stray presentation build artifacts at the project root
Get-ChildItem -Path $root -Filter "presentation.*" -File -ErrorAction SilentlyContinue | ForEach-Object {
    Move-Item -LiteralPath $_.FullName -Destination "$root\archive\old_tex" -Force
    Write-Host "  moved   $($_.Name)"
}
Move-Safe "paper\presentation.tex" "$root\archive\old_tex"

Write-Host ""
Write-Host "=== 9. Copying Elsevier class files into Tex\ ==="
$casSrc = "$root\Tex\els-cas-templates"
foreach ($f in @("cas-sc.cls", "cas-common.sty", "cas-model2-names.bst")) {
    $src = Join-Path $casSrc $f
    if (Test-Path -LiteralPath $src) {
        Copy-Item -LiteralPath $src -Destination "$root\Tex" -Force
        Write-Host "  copied  $f -> Tex\"
    } else {
        Write-Host "  WARN    $f not found under Tex\els-cas-templates\"
    }
}

Write-Host ""
Write-Host "=== Done. ==="
Write-Host "Review the new layout, then initialise git:"
Write-Host "    cd $root"
Write-Host "    git init"
Write-Host "    git add ."
Write-Host "    git commit -m ""Initial commit: replication package"""
Write-Host ""
Write-Host "The .gitignore keeps data, archive\, Citation Papers\, and build"
Write-Host "artifacts out of the commit. Run 'git status' to confirm before pushing."

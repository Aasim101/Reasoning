# Run from repo root (Testing/). Rebuilds precision trees then paper artefacts.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$py = if (Test-Path .\.venv\Scripts\python.exe) { ".\.venv\Scripts\python.exe" } else { "python" }

Write-Host "=== split_precision ==="
& $py scripts\split_precision.py --src results_bundle
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== paper analysis (n-bootstrap=1000) ==="
& $py -m src.analysis.paper `
  --results-dir results_fp16 `
  --precision-results-dir results_4bit `
  --out-dir paper_artefacts `
  --n-bootstrap 1000 `
  --log-level INFO

exit $LASTEXITCODE

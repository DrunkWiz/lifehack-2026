# Task shortcuts (Windows). Usage:  .\run.ps1 all   then   .\run.ps1 demo
param([Parameter(Position=0)][string]$Task = "help")

# Always use this project's venv when it exists. Without this, whatever venv
# happens to be first on PATH wins, which is how you end up running the
# pipeline with someone else's interpreter.
$venv = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$py   = if (Test-Path $venv) { $venv } else { "python" }
if ($py -eq "python") {
  Write-Host "! no .venv found — using whatever 'python' resolves to:" -ForegroundColor Yellow
  Write-Host "  $((Get-Command python -ErrorAction SilentlyContinue).Source)" -ForegroundColor Yellow
  Write-Host "  create one with:  py -3.12 -m venv .venv" -ForegroundColor Yellow
}

$env:PYTHONPATH = Join-Path $PSScriptRoot "src"

switch ($Task) {
  "setup"    { & $py -m pip install --upgrade pip; & $py -m pip install -r requirements.txt }
  "seed"     { & $py scripts/gen_seed_data.py }
  "doctor"   { & $py -m agentcard doctor }
  "enrich"   { & $py -m agentcard enrich }
  "index"    { & $py -m agentcard index }
  "simulate" { & $py -m agentcard simulate }
  "all"      { & $py -m agentcard all }
  "offline"  { & $py -m agentcard all --provider local --embed-provider local }
  "demo"     { & $py -m streamlit run app/streamlit_app.py }
  "test"     { & $py -m pytest tests -q }
  "ask"      { & $py -m agentcard ask @args }
  "which"    { Write-Host "interpreter: $py"; & $py -V }
  default {
    Write-Host "tasks: setup seed doctor enrich index simulate all offline demo test ask which"
    Write-Host "e.g.   .\run.ps1 all   then   .\run.ps1 demo"
  }
}

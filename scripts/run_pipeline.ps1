<#
Phase 2 -> decision -> Phase 3 on one Windows machine: the PowerShell twin of run_pipeline.sh
(docs/BENCHMARK.md). CPU arms run in parallel (-Jobs) while TabPFN campaigns share the GPU
(-GpuJobs at a time). Finished campaigns are skipped, so after an interruption run the same
command again.

Set up once, from the repository root:
  uv sync --link-mode copy --extra tabpfn35 --extra cu128     (cu126 for RTX 40 and older)

Run from the repository root:
  powershell -ExecutionPolicy Bypass -File scripts\run_pipeline.ps1 -Phase 2        # Phase 2 and the decision
  powershell -ExecutionPolicy Bypass -File scripts\run_pipeline.ps1 -Phase 3        # Phase 3 (needs Phase 2)
  powershell -ExecutionPolicy Bypass -File scripts\run_pipeline.ps1 -Pilot -Jobs 4  # quick machinery check
#>
param(
    [int]$Jobs = [Math]::Max(1, [Environment]::ProcessorCount - 2),
    [int]$GpuJobs = 1,
    [ValidateSet('all', '2', '3')][string]$Phase = 'all',
    [switch]$Pilot
)
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw '.venv not found: run the uv sync command in the header of this script first.' }

$flag = @()
$suffix = ''
if ($Pilot) { $flag = @('--pilot'); $suffix = '-pilot' }
$parallel = @('--jobs', "$Jobs", '--gpu-jobs', "$GpuJobs")

function Bench {
    Write-Host ('>> metis_bench ' + ($args -join ' ')) -ForegroundColor Cyan
    & $python -W ignore -m metis_bench @args
    if ($LASTEXITCODE -ne 0) { throw "metis_bench $($args -join ' ') failed with exit code $LASTEXITCODE" }
}

# TabPFN arms use TabPFN's own schedule, tuned beforehand on the replay data (amendment 4).
if (-not $Pilot -and -not (Test-Path 'experiments\tuned\tabpfn_schedule.json')) {
    throw 'experiments\tuned\tabpfn_schedule.json is missing: pull the latest code (docs/REPLAY.md).'
}

# Load the committed real-data oracles once, so parallel workers never build them at the same time.
Bench oracles

if ($Phase -in @('all', '2')) {
    # The surrogate comparison first, so TabPFN vs XGBoost is answered before the side questions.
    # phase2_import is not run (amendment 6).
    foreach ($exp in @('phase2_surrogates', 'phase2_init', 'phase2_batch')) {
        Bench run "experiments/$exp.yaml" @flag @parallel
        Bench report "experiments/$exp.yaml" @flag
    }
    $decide = @($flag)
    # Pilots have too few seeds for significance; pick by mean so Phase 3 has something to run.
    if ($Pilot) { $decide += '--ignore-significance' }
    Bench decide @decide "results/phase2_init$suffix/report" "results/phase2_surrogates$suffix/report" "results/phase2_batch$suffix/report"
}

if ($Phase -in @('all', '3')) {
    # No schedule re-tuning (amendment 6): every model keeps its own schedule.
    Bench tune experiments/phase3_hparams_tune.yaml @flag @parallel
    Bench run experiments/phase3_hparams.yaml @flag @parallel
    Bench report experiments/phase3_hparams.yaml @flag
    Bench run experiments/phase3_ablation.yaml @flag @parallel
    Bench report experiments/phase3_ablation.yaml @flag
}

Bench summary @flag
Write-Host 'Done. Overview: results\SUMMARY.md (or SUMMARY-pilot.md)' -ForegroundColor Green

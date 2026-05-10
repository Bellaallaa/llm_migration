# Streamlit paper-style eval (commit 3cce9679, mig1 app_session_test)
# Prerequisites: conda activate llm_migration; $env:SILICONFLOW_API_KEY set
#   pip install pytest-cov
#   pip install -e "<repo>\_repos\streamlit\lib"

$ErrorActionPreference = "Stop"
# llm_migration (parent of experiments/)
$Root = Split-Path -Parent $PSScriptRoot
$Repo = Join-Path $Root "_repos\streamlit"
$LibRoot = Join-Path $Repo "lib"
$TestFile = Join-Path $LibRoot "tests\streamlit\runtime\app_session_test.py"
$Before = Join-Path $Root "TestMigrationsInPy\projects\streamlit\1\diff\mig1-before-app_session_test.py"
$After = Join-Path $Root "TestMigrationsInPy\projects\streamlit\1\diff\mig1-after-app_session_test.py"

if (-not (Test-Path $Repo)) {
    Write-Host "Missing repo at $Repo — clone first (see experiments/README_STREAMLIT.txt)"
    exit 1
}

$Model = if ($args[0]) { $args[0] } else { "Qwen/Qwen3-8B" }
$OutDir = Join-Path $Root "outputs\iterative\streamlit_mig1_$($Model.Replace('/','_'))"

Write-Host "Repo: $Repo"
Write-Host "Model: $Model"
Write-Host "Output: $OutDir"

Set-Location $Root
python run_paper_task.py `
    --repo $Repo `
    --test-file "lib/tests/streamlit/runtime/app_session_test.py" `
    --before $Before `
    --after $After `
    --cov-package streamlit `
    --model $Model `
    --output-dir $OutDir `
    --execute

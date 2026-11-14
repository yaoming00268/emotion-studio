$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Cannot create Python environment.' }
}
$emotionPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
& $emotionPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed.' }
$emotionWheel = Join-Path $PSScriptRoot 'runtime/wheels/torch-2.8.0+cu128-cp312-cp312-win_amd64.whl'
if ((Test-Path -LiteralPath $emotionWheel) -and (& $emotionPython -c "import sys; print(sys.version_info[:2] == (3,12))") -eq 'True') {
    & $emotionPython -m pip install $emotionWheel
} else {
    & $emotionPython -m pip install 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu128
}
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
& $emotionPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& $emotionPython -m emotionlab diagnose
if ($LASTEXITCODE -ne 0) { throw 'CUDA verification failed.' }
Write-Host 'Ready. Run run.bat to open Emotion Studio.'

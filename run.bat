@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" (
  echo Run powershell -ExecutionPolicy Bypass -File setup.ps1 first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m emotionlab gui %*
if errorlevel 1 pause

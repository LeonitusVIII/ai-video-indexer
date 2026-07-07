@echo off
setlocal
cd /d "%~dp0"

if not exist venv\Scripts\streamlit.exe (
  echo Run setup.bat first to install dependencies.
  pause
  exit /b 1
)

if not exist config.json (
  copy /Y config.example.json config.json >nul
)

if not exist data mkdir data
if not exist jobs mkdir jobs
if not exist logs mkdir logs

echo Starting AI Video Indexer...
echo Opening http://localhost:8501 in your default browser...
echo Press Ctrl+C in this window to stop the app.
echo.

start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:8501"

venv\Scripts\streamlit.exe run app.py --server.headless true

@echo off
setlocal
cd /d "%~dp0"

echo.
echo  AI Video Indexer - First-time setup
echo  ===================================
echo.

where python >nul 2>&1
if errorlevel 1 (
  echo ERROR: Python was not found on PATH.
  echo Install Python 3.11 or 3.12 from https://www.python.org/downloads/
  echo and check "Add python.exe to PATH" during install.
  echo.
  pause
  exit /b 1
)

for /f "delims=" %%v in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PYVER=%%v
echo Using Python %PYVER%

if not exist venv (
  echo Creating virtual environment...
  python -m venv venv
  if errorlevel 1 (
    echo Failed to create venv.
    pause
    exit /b 1
  )
)

if not exist config.json (
  echo Creating config.json from template...
  copy /Y config.example.json config.json >nul
)

echo.
echo Installing Streamlit (required to launch the app)...
venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 (
  echo Failed to upgrade pip.
  pause
  exit /b 1
)
venv\Scripts\python.exe -m pip install streamlit streamlit-autorefresh
if errorlevel 1 (
  echo ERROR: Streamlit installation failed.
  pause
  exit /b 1
)
if not exist venv\Scripts\streamlit.exe (
  echo ERROR: streamlit.exe was not created. Re-run setup.bat.
  pause
  exit /b 1
)
echo Streamlit installed.

echo.
echo Checking FFmpeg...
where ffmpeg >nul 2>&1
if errorlevel 1 (
  echo FFmpeg not found on PATH. Attempting install via winget...
  where winget >nul 2>&1
  if errorlevel 1 (
    echo WARNING: winget is not available on this PC.
    echo Install FFmpeg manually from https://ffmpeg.org/download.html
    echo Normalization and transcription will fail without it.
    echo.
  ) else (
    winget install --id Gyan.FFmpeg -e --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
      echo WARNING: winget FFmpeg install failed or was skipped.
      echo Install manually from https://ffmpeg.org/download.html
      echo.
    ) else (
      echo FFmpeg installed via winget.
      echo If ffmpeg is still not recognized, close this window and run setup.bat again.
      echo.
    )
  )
) else (
  echo FFmpeg found on PATH.
)

echo.
echo Installing remaining dependencies. This can take 20-40 minutes on first run.
echo Downloads include PyTorch, Whisper, and vision models on first use.
echo.

if not exist data mkdir data
venv\Scripts\python.exe scripts\install_dependencies.py --status-file data\install_status.json
if errorlevel 1 (
  echo.
  echo Setup failed. Check data\install_status.json for details.
  pause
  exit /b 1
)

echo.
echo Verifying install...
if not exist venv\Scripts\streamlit.exe (
  echo ERROR: streamlit.exe is missing. Re-run setup.bat.
  pause
  exit /b 1
)

where ffmpeg >nul 2>&1
if errorlevel 1 (
  echo WARNING: ffmpeg is still not on PATH.
  echo Install FFmpeg before running normalize or transcribe jobs.
  echo https://ffmpeg.org/download.html
) else (
  echo FFmpeg OK.
)

echo.
echo Setup complete.
echo Run start.bat to launch the app in your browser.
echo.
pause

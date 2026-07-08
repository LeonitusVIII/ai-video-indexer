@echo off
setlocal
cd /d "%~dp0"
echo Registering Windows scheduled task for overnight pipeline runs...
"%~dp0venv\Scripts\python.exe" "%~dp0scripts\scheduled_runner.py" >nul 2>&1
schtasks /Create /TN "AI Video Indexer Overnight" /TR "cmd /c cd /d \"%~dp0\" && \"%~dp0venv\Scripts\python.exe\" \"%~dp0scripts\scheduled_runner.py\"" /SC MINUTE /MO 5 /F
if errorlevel 1 (
    echo Failed to register task. Try running this file as Administrator.
    pause
    exit /b 1
)
echo.
echo Task registered: runs every 5 minutes.
echo Enable the schedule under Tools/System in the app, then leave the PC on overnight.
echo.
pause

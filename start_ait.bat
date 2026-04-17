@echo off
REM AiT Application Launcher
REM Double-click this file to start the AiT application suite

cd /d "%~dp0"

REM Check if processing dependencies are installed
python -c "import ultralytics" 2>nul
if errorlevel 1 (
    echo Processing libraries not found. Installing dependencies...
    echo This only happens once and may take a few minutes.
    echo.
    pip install -e .
    if errorlevel 1 (
        echo.
        echo Installation failed. See error above.
        pause
        exit /b 1
    )
    echo.
    echo Dependencies installed successfully.
    echo.
)

python -m ait.launcher
pause

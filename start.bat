@echo off
rem Starts vis10n with the project's venv and opens it in the browser.
rem Close this window (or press Ctrl+C) to stop the server.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo No venv found in .venv - see the README for setup.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" main.py
pause

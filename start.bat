@echo off
rem Starts vis10n with the project's venv and opens it in its own window.
rem Closing the app window also stops the server.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo No venv found in .venv - see the README for setup.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" main.py
rem Close with the app; stay open only if it crashed, so the error can be read
if errorlevel 1 pause

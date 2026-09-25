@echo off
rem Launch the Jev Agent Gym Console (uses the venv when present).
cd /d "%~dp0"
set PY=python
if exist "%~dp0.venv\Scripts\python.exe" set PY=%~dp0.venv\Scripts\python.exe
start "" http://127.0.0.1:7860
"%PY%" webui.py

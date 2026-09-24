@echo off
rem Run the Jev decision agent on CartPole-v1 (Windows).
cd /d "%~dp0"
rem Prefer the local venv (laya engine) if it exists.
set PY=python
if exist "%~dp0.venv\Scripts\python.exe" set PY=%~dp0.venv\Scripts\python.exe
"%PY%" -m pip install -r requirements.txt
"%PY%" run_agent.py --episodes 3 --seed 0 --render

@echo off
rem Run the Jev decision agent on CartPole-v1 (Windows).
cd /d "%~dp0"
python -m pip install -r requirements.txt
python run_agent.py --episodes 3 --seed 0 --render

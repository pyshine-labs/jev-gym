#!/usr/bin/env sh
# Run the Jev decision agent on CartPole-v1 (Linux / macOS / Git Bash).
cd "$(dirname "$0")"

# Pick a real python: on Windows, python3 may be the Microsoft Store stub.
PY=python3
if ! command -v "$PY" >/dev/null 2>&1 || ! "$PY" -c "import sys" 2>/dev/null; then
    PY=python
fi

"$PY" -m pip install -r requirements.txt
"$PY" run_agent.py --episodes 3 --seed 0 --render

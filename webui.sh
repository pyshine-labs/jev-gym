#!/usr/bin/env sh
# Launch the Jev Agent Gym Console (uses the venv when present).
cd "$(dirname "$0")"
if [ -x ".venv/bin/python" ]; then
    PY=.venv/bin/python
else
    PY=python3
    command -v "$PY" >/dev/null 2>&1 || PY=python
fi
"$PY" webui.py &
"$PY" - <<'EOF'
import time, webbrowser
time.sleep(1.5)
webbrowser.open("http://127.0.0.1:7860")
EOF
wait

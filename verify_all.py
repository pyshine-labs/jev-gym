"""One-by-one verification of every WebUI env through the live HTTP API."""
import json
import urllib.request

BASE = "http://127.0.0.1:7860"


def post(path, body=None):
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


ENVS = [
    ("CartPole-v1", 1),            # target: survive 500 steps
    ("MountainCar-v0", 1),         # target: reach the flag
    ("MountainCarContinuous-v0", 1),
    ("Acrobot-v1", 1),             # target: goal height
    ("Pendulum-v1", 1),            # target: return >= -400
    ("LunarLander-v3", 1),         # target: return >= 200
]

print(f"{'env':28s} {'steps':>5s} {'return':>8s}  outcome")
for env_id, seed in ENVS:
    snap = post("/api/start", {"env": env_id, "engine": "local", "seed": seed})
    if snap.get("error"):
        print(f"{env_id:28s} START FAIL {snap['error']}")
        continue
    while not snap["done"]:
        snap = post("/api/step")
    post("/api/stop")
    outcome = ("survived" if snap["truncated"] else
               "goal reached" if snap["terminated"] else "?")
    print(f"{env_id:28s} {snap['steps']:5d} {snap['reward']:8.1f}  "
          f"{outcome} (engine {snap['engine']})")

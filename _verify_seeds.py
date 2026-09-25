"""Double-check all envs: N seeds each, through the webui Session stack.

Usage: python _verify_seeds.py [N]
"""
import sys

import webui
from webui import Session

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5

TARGETS = {
    "CartPole-v1": (500, "survive", 1),
    "MountainCar-v0": (1000, "goal", 20),
    "MountainCarContinuous-v0": (1000, "goal", 20),
    "Acrobot-v1": (1000, "goal", 20),
    "Pendulum-v1": (1000, "score", 20),
    "LunarLander-v3": (1000, "score", 20),
    "BipedalWalker-v3": (1600, "score", 20),
}

grand_pass = grand_total = 0
for env_id, (max_steps, kind, decimate) in TARGETS.items():
    fails = []
    for seed in range(N):
        s = Session({"env": env_id, "engine": "local", "min_conf": 0.60,
                     "decimate": decimate, "seed": seed,
                     "max_steps": max_steps})
        while not (s.terminated or s.truncated or s.steps >= s.max_steps):
            s.step()
        ok, msg = s.outcome()
        if not ok:
            fails.append((seed, round(s.reward, 1)))
        print(f"  {env_id:26s} seed {seed}: steps={s.steps:4d} "
              f"return={s.reward:8.1f} policy={s.env_id and ('PPO' if env_id in webui.LEARNED else 'rule')} "
              f"ok={ok} ({msg})", flush=True)
    grand_pass += N - len(fails)
    grand_total += N
    print(f"{env_id}: {N - len(fails)}/{N} pass"
          + (f"  FAILS: {fails}" if fails else ""), flush=True)
print(f"TOTAL: {grand_pass}/{grand_total} episodes pass", flush=True)

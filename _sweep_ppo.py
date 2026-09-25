"""Evaluate saved PPO checkpoints over many seeds.

Usage: python _sweep_ppo.py <env_id> <model_path> <threshold> [n_seeds]
Success = total episode reward >= threshold
(Pendulum -400, LunarLander 200, BipedalWalker 300).
"""
import sys

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

env_id, path, thr = sys.argv[1], sys.argv[2], float(sys.argv[3])
n = int(sys.argv[4]) if len(sys.argv) > 4 else 300

model = PPO.load(path, device="cuda")
env = gym.make(env_id)
rets, fails = [], []
for s in range(n):
    obs, _ = env.reset(seed=s)
    total, done = 0.0, False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(a)
        total += float(r)
        done = term or trunc
    rets.append(total)
    if total < thr:
        fails.append((s, round(total, 1)))
rets = np.array(rets)
print(f"{env_id} {path}: n={n} mean={rets.mean():.1f} min={rets.min():.1f} "
      f"fails={len(fails)}/{n}", flush=True)
for s, r in fails[:20]:
    print(f"  seed {s}: {r}")

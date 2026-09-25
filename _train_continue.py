"""Continue training a saved PPO checkpoint.

Usage: python _train_continue.py <env> <src_model> <steps> <tag> [lr]
Saves the best checkpoint (by 12-episode eval) to .models/<tag>/best_model,
then prints a final 20-seed eval.
"""
import sys

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

import gymnasium

env_id, src, steps, tag = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
lr = float(sys.argv[5]) if len(sys.argv) > 5 else 3e-4

train = make_vec_env(env_id, n_envs=8, seed=1000)
eval_env = Monitor(gymnasium.make(env_id))

model = PPO.load(src, device="cuda")
model.set_env(train)
model.lr_schedule = (lambda progress, _lr=lr: _lr) if lr else model.lr_schedule

cb = EvalCallback(eval_env, best_model_save_path=f".models/{tag}",
                  eval_freq=max(100_000 // 8, 1), n_eval_episodes=12,
                  deterministic=True, verbose=1)
model.learn(total_timesteps=steps, callback=cb, reset_num_timesteps=False)

best = PPO.load(f".models/{tag}/best_model")
rets = []
for s in range(20):
    obs, _ = eval_env.reset(seed=s)
    total, done = 0.0, False
    while not done:
        a, _ = best.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = eval_env.step(a)
        total += float(r)
        done = term or trunc
    rets.append(total)
print(f"{tag} final eval over 20 seeds: mean={np.mean(rets):.1f} "
      f"min={np.min(rets):.1f}", flush=True)

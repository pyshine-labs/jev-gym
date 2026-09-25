"""PPO with Pendulum-appropriate hyperparameters (SB3 zoo recipe):
gemma 0.9, use_sde=True (squashed actions), lr 1e-3.

Usage: python _train_ppo2.py <env> <steps> <tag> [seed_offset]
"""
import sys

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

import gymnasium

env_id, steps, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
seed_off = int(sys.argv[4]) if len(sys.argv) > 4 else 0

train = make_vec_env(env_id, n_envs=8, seed=seed_off)
eval_env = Monitor(gymnasium.make(env_id))

model = PPO("MlpPolicy", train, device="cuda", verbose=0,
            learning_rate=1e-3, n_steps=1024, batch_size=1024,
            gamma=0.9, gae_lambda=0.95, use_sde=True, ent_coef=0.0)
cb = EvalCallback(eval_env, best_model_save_path=f".models/{tag}",
                  eval_freq=max(100_000 // 8, 1), n_eval_episodes=12,
                  deterministic=True, verbose=1)
model.learn(total_timesteps=steps, callback=cb)

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

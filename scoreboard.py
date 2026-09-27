"""Scoreboard: pure-laya mode, 5 episodes per env, tuned checkpoint."""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)  # .models checkpoints resolve relative to the repo root

import numpy as np
import gymnasium as gym

from jev_agent import make_engine, questions_for
from webui import choose_action, car_features, LEARNED

ENVS = ["CartPole-v1", "MountainCar-v0", "MountainCarContinuous-v0",
        "Acrobot-v1", "Pendulum-v1", "LunarLander-v3", "CarRacing-v3",
        "BipedalWalker-v3"]


def run(env_id, engine, seed):
    env = gym.make(env_id)
    obs, _ = env.reset(seed=seed)
    total, steps, done = 0.0, 0, False
    while not done:
        # CarRacing pixels are useless to the engines: ask on the driving
        # feature vector (same as the WebUI's ask state)
        state = car_features(env) if env_id.startswith("CarRacing") else obs
        answers = engine.ask(env_id, state, questions_for(env_id))
        action = choose_action(env, env_id, obs, answers, jev_drives=True)
        obs, r, term, trunc, _ = env.step(action)
        total += float(r)
        steps += 1
        done = term or trunc or steps >= 1000
    env.close()
    return total


def main():
    engine = make_engine("laya")
    print(f"engine: {engine.engine}", flush=True)
    for env_id in ENVS:
        scores = [run(env_id, engine, 1000 + i) for i in range(5)]
        mean = sum(scores) / len(scores)
        print(f"{env_id:28s} mean {mean:8.1f}  {scores}", flush=True)


if __name__ == "__main__":
    main()

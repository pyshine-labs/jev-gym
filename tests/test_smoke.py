import gymnasium as gym
import numpy as np


def test_cartpole_step():
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (4,)
    obs2, reward, terminated, truncated, _ = env.step(1)
    assert reward == 1.0
    assert obs2.shape == (4,)
    assert not (terminated and truncated)
    env.close()


def test_numpy_present():
    assert int(np.__version__.split(".")[0]) >= 1

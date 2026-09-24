"""Run the Jev-style local decision agent on the CartPole-v1 inverted pendulum.

Every step: the environment state goes to the decision engine, a typed answer
set comes back (choice / noul / score, with probabilities and confidence), and
the controller maps it to one of two actions. Nothing is generated as text.

Examples:
    python run_agent.py --episodes 5 --seed 0
    python run_agent.py --episodes 1 --render --verbose
    python run_agent.py --engine laya --log decisions.csv
"""

from __future__ import annotations

import argparse
import csv
import time

import gymnasium as gym

from controller import decide
from jev_agent import DEFAULT_QUESTIONS, make_engine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--engine", choices=["local", "laya"], default="local",
                   help="local = offline head (default); laya = pip install laya")
    p.add_argument("--render", action="store_true", help="open a window")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=500,
                   help="CartPole-v1 truncates at 500 anyway")
    p.add_argument("--log", default="", help="write a CSV decision trail")
    p.add_argument("--verbose", action="store_true", help="print every step")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    engine = make_engine(args.engine)

    logf = None
    writer = None
    if args.log:
        logf = open(args.log, "w", newline="")
        writer = csv.writer(logf)
        writer.writerow(["episode", "step", "x", "x_dot", "theta", "theta_dot",
                         "direction", "dir_conf", "at_risk", "instability",
                         "action", "engine"])

    env = gym.make("CartPole-v1", render_mode="human" if args.render else None)
    print(f"engine={engine.engine} env=CartPole-v1 episodes={args.episodes} "
          f"seed={args.seed}")

    for ep in range(1, args.episodes + 1):
        state, _ = env.reset(seed=args.seed + ep - 1)
        steps = 0
        done = False
        confs: list[float] = []
        engines: dict[str, int] = {}
        latency: list[float] = []

        while not done and steps < args.max_steps:
            t0 = time.perf_counter()
            answers = engine.ask(state, DEFAULT_QUESTIONS)
            latency.append(time.perf_counter() - t0)

            direction = answers["direction"]
            confs.append(direction.confidence)
            engines[direction.engine] = engines.get(direction.engine, 0) + 1

            action = decide(state, answers)

            if writer:
                writer.writerow([ep, steps + 1,
                                 *(round(float(v), 4) for v in state),
                                 direction.value, round(direction.confidence, 3),
                                 answers["at_risk"].value,
                                 answers["instability"].value,
                                 action, direction.engine])
            if args.verbose:
                print(f"  step {steps + 1:3d} | {direction!r} | "
                      f"risk {answers['at_risk'].value:.2f} | "
                      f"instability {answers['instability'].value:.2f} | "
                      f"action {'PUSH_RIGHT' if action else 'PUSH_LEFT'}")

            state, _reward, term, trunc, _info = env.step(action)
            steps += 1
            done = term or trunc

        avg_conf = sum(confs) / len(confs) if confs else 0.0
        avg_ms = 1000.0 * sum(latency) / len(latency) if latency else 0.0
        eng = ", ".join(f"{k} {v}" for k, v in engines.items()) or "-"
        print(f"episode {ep}: survived {steps} steps | "
              f"avg direction confidence {avg_conf:.2f} | "
              f"avg decision {avg_ms:.2f} ms | decided by: {eng}")

    env.close()
    if logf:
        logf.close()
        print(f"decision trail written to {args.log}")


if __name__ == "__main__":
    main()

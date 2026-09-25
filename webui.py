"""WebUI: launch any gymnasium env and watch a Jev-style agent control it.

Run with the repo's venv (GPU laya) or any python with gymnasium:
    python webui.py            # http://127.0.0.1:7860
"""
from __future__ import annotations

import base64
import io
import math
import threading
import time

import numpy as np
from flask import Flask, jsonify, render_template, request

import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from PIL import Image

from controller import decide as cartpole_decide
from jev_agent import DEFAULT_QUESTIONS, LocalDecisionHead, make_engine

app = Flask(__name__)

LOCK = threading.Lock()
SESSION: "Session | None" = None

# Families with 1-D Box observations and rgb_array rendering. Toy-text envs
# (integer observations) cannot answer numeric typed questions, so they are
# not offered. CarRacing is excluded too: pixel observations.
FAMILIES = ("CartPole", "MountainCar", "Acrobot", "Pendulum",
            "LunarLander", "BipedalWalker")


def canonical_state(env_id: str, obs) -> np.ndarray:
    """Map any supported observation to the agent's 4-D cart frame
    [x, x_dot, theta, theta_dot]."""
    v = np.asarray(obs, dtype=float).ravel()
    if env_id.startswith("CartPole"):
        return v
    if env_id.startswith("Pendulum"):
        return np.array([0.0, 0.0, math.atan2(v[1], v[0]), v[2]])
    if env_id.startswith("Acrobot"):  # [cos1, sin1, cos2, sin2, v1, v2]
        return np.array([0.0, 0.0, math.atan2(v[3], v[2]), v[4]])
    if env_id.startswith("MountainCar"):  # [position, velocity]
        return np.array([v[0], v[1], 0.0, 0.0])
    if env_id.startswith("LunarLander"):  # [x, y, vx, vy, angle, vang, ...]
        return np.array([v[0], v[2], v[4], v[5]])
    return np.zeros(4)  # BipedalWalker et al: neutral posture


def discrete_action(env_id: str, n: int, to_right: bool) -> int:
    if env_id.startswith("LunarLander") and n >= 4:
        return 3 if to_right else 1        # 1 = left engine, 3 = right
    return n - 1 if to_right else 0


def list_envs() -> list[str]:
    # gymnasium 1.3 removed render_modes from EnvSpec; registry only lists
    # envs whose packages import cleanly, so family matching is enough.
    out = [eid for eid in gym.registry if eid.startswith(FAMILIES)]
    return sorted(set(out)) or ["CartPole-v1"]


def frame_b64(env) -> str | None:
    frame = env.render()
    if frame is None:
        return None
    img = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=72)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def choose_action(env, env_id, state, answers):
    act = env.action_space
    to_right = answers["direction"].value == "right"
    if isinstance(act, Discrete):
        if np.isscalar(state):
            return discrete_action(env_id, int(act.n), to_right)
        if len(state) == 4 and act.n == 2:
            return int(cartpole_decide(state, answers))  # tuned law + wall guard
        return discrete_action(env_id, int(act.n), to_right)
    if isinstance(act, Box):
        u = float(LocalDecisionHead().features(
            canonical_state(env_id, state))["lean"])
        u += float(LocalDecisionHead().features(
            canonical_state(env_id, state))["align"])
        return np.clip(u, act.low, act.high).astype(act.dtype).reshape(act.shape)
    return act.sample()


def answers_json(answers) -> list:
    out = []
    for q in DEFAULT_QUESTIONS:
        a = answers.get(q.name)
        if a is None:
            continue
        out.append({
            "name": a.name, "qtype": a.qtype,
            "value": str(a.value), "confidence": round(float(a.confidence), 3),
            "engine": a.engine,
            "probabilities": {str(k): round(float(v), 4)
                              for k, v in (a.probabilities or {}).items()},
        })
    return out


class Session:
    def __init__(self, cfg: dict):
        self.env_id = cfg["env"]
        self.env = gym.make(self.env_id, render_mode="rgb_array")
        self.engine = make_engine(
            cfg.get("engine", "local"),
            model_path=cfg.get("model_path") or None,
            min_confidence=float(cfg.get("min_conf", 0.60)),
        )
        self.decimate = max(1, int(cfg.get("decimate", 1)))
        self.max_steps = int(cfg.get("max_steps", 500))
        self.state, _ = self.env.reset(seed=int(cfg.get("seed", 0)))
        self.steps = 0
        self.reward = 0.0
        self.terminated = False
        self.truncated = False
        self.answers = None
        self.engines: dict[str, int] = {}
        self.latencies: list[float] = []
        self.trail: list[dict] = []

        self._ask()
        self._apply()

    def _ask(self) -> None:
        if self.answers is None or self.steps % self.decimate == 0:
            t0 = time.perf_counter()
            self.answers = self.engine.ask(
                canonical_state(self.env_id, self.state), DEFAULT_QUESTIONS)
            self.latencies.append(time.perf_counter() - t0)

    def _apply(self) -> None:
        direction = self.answers["direction"]
        self.engines[direction.engine] = self.engines.get(direction.engine, 0) + 1
        self.trail.append({
            "step": self.steps + 1,
            "answers": answers_json(self.answers),
            "engine": direction.engine,
            "conf": round(float(direction.confidence), 3),
        })
        self.trail = self.trail[-40:]

    def step(self) -> dict:
        with LOCK:
            if self.terminated or self.truncated or self.steps >= self.max_steps:
                return self.snapshot(done=True)
            self._ask()
            action = choose_action(self.env, self.env_id, self.state,
                                   self.answers)
            state, reward, term, trunc, _ = self.env.step(action)
            self.state = state
            self.reward += float(reward)
            self.steps += 1
            self.terminated = bool(term)
            self.truncated = bool(trunc)
            self._apply()
            return self.snapshot()

    def snapshot(self, done: bool = False) -> dict:
        confs = [t["conf"] for t in self.trail]
        risk = None
        if self.answers is not None:
            risk = self.answers["at_risk"].value
        return {
            "frame": frame_b64(self.env),
            "env": self.env_id,
            "engine": self.engine.engine,
            "steps": self.steps,
            "reward": round(self.reward, 2),
            "done": done or self.terminated or self.truncated,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "action": self.trail[-1] if self.trail else None,
            "trail": self.trail[-12:],
            "engines": self.engines,
            "avg_ms": round(1000 * sum(self.latencies) / len(self.latencies), 1)
                      if self.latencies else 0.0,
            "risk": round(float(risk), 3) if risk is not None else None,
            "avg_conf": round(sum(confs) / len(confs), 3) if confs else 0.0,
        }


def get_session() -> Session | None:
    with LOCK:
        return SESSION


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/envs")
def api_envs():
    return jsonify({"envs": list_envs()})


@app.route("/api/start", methods=["POST"])
def api_start():
    global SESSION
    cfg = request.get_json(force=True, silent=True) or {}
    try:
        session = Session(cfg)
    except Exception as exc:  # noqa: BLE001 - report env/engine problems to UI
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400
    with LOCK:
        if SESSION is not None:
            try:
                SESSION.env.close()
            except Exception:  # noqa: BLE001
                pass
        SESSION = session
    return jsonify(session.snapshot())


@app.route("/api/step", methods=["POST"])
def api_step():
    session = get_session()
    if session is None:
        return jsonify({"error": "no session; press Start first"}), 400
    return jsonify(session.step())


@app.route("/api/stop", methods=["POST"])
def api_stop():
    global SESSION
    with LOCK:
        if SESSION is not None:
            try:
                SESSION.env.close()
            except Exception:  # noqa: BLE001
                pass
        SESSION = None
    return jsonify({"ok": True})


if __name__ == "__main__":
    print("WebUI on http://127.0.0.1:7860")
    app.run(host="127.0.0.1", port=7860, threaded=True)

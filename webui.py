"""WebUI: launch any gymnasium env and watch a Jev-style agent control it.

Run with the repo's venv (GPU laya) or any python with gymnasium:
    python webui.py            # http://127.0.0.1:7860
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
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

# Learned (PPO) policies, optional. Where a trained checkpoint exists it
# drives the motor action; the Jev typed questions still assess every step.
LEARNED: dict = {}
try:
    from stable_baselines3 import PPO as _PPO

    for _eid, _paths in (
        ("Pendulum-v1", (".models/pendulum_final/best_model",)),
        ("LunarLander-v3", (".models/lander_final/best_model",)),
        ("BipedalWalker-v3", (".models/walker_final/best_model",
                              ".models/best3/best_model")),
        ("BipedalWalkerHardcore-v3", (".models/hardcore_final/best_model",)),
    ):
        for _p in _paths:
            try:
                LEARNED[_eid] = _PPO.load(_p)
                break
            except Exception:  # noqa: BLE001 - missing checkpoint is fine
                pass
except ImportError:  # stable-baselines3 not installed: rule laws only
    pass


def canonical_state(env_id: str, obs) -> np.ndarray:
    """Map any supported observation to the agent's 4-D cart frame
    [x, x_dot, theta, theta_dot]."""
    v = np.asarray(obs, dtype=float).ravel()
    if env_id.startswith("CartPole"):
        return v
    if env_id.startswith("Pendulum"):
        return np.array([0.0, 0.0, math.atan2(v[1], v[0]), v[2]])
    if env_id.startswith("Acrobot"):  # [cos1, sin1, cos2, sin2, v1, v2]
        return np.array([0.0, 0.0, math.atan2(v[3], v[2]), v[5]])
    if env_id.startswith("MountainCar"):  # [position, velocity]
        return np.array([v[0], v[1], 0.0, 0.0])
    if env_id.startswith("LunarLander"):  # [x, y, vx, vy, angle, vang, ...]
        return np.array([v[0], v[2], v[4], v[5]])
    return np.zeros(4)  # BipedalWalker et al: neutral posture


def discrete_action(env_id: str, n: int, to_right: bool) -> int:
    if env_id.startswith("LunarLander") and n >= 4:
        return 3 if to_right else 1        # 1 = left engine, 3 = right
    return n - 1 if to_right else 0


# ---------------------------------------------------------------------------
# Per-family control laws. Each solves its env's actual objective; the Jev
# typed questions (direction / at_risk / instability) still assess every step.
# ---------------------------------------------------------------------------

def policy_mountaincar(obs) -> int:
    """Energy pumping: push in the direction of motion (classic solver)."""
    x, v = float(obs[0]), float(obs[1])
    return 2 if v > 0 else 0


def policy_acrobot(obs) -> int:
    """Pump elbow energy: torque in the direction of the elbow velocity.

    When the elbow stalls (|v2| tiny), pump on the shoulder velocity so the
    chain never sits motionless at the bottom (fixes rare stall seeds).
    """
    v1, v2 = float(obs[4]), float(obs[5])
    if abs(v2) < 0.05:
        return 2 if v1 > 0 else 0
    return 2 if v2 > 0 else 0


PENDULUM = dict(k_pump=0.1, k1=16.0, k2=4.0, th_sw=0.7, w_sw=2.0,
                c=15.0, e_target=30.0)


def policy_pendulum(obs, gains: dict = PENDULUM) -> float:
    """Swing-up by pumping energy to the upright level, then PD hold.

    Gym dynamics: wdot = c*sin(th) + 3*u (c=15, th measured from up,
    th=pi hangs). E = 0.5*w^2 + c*(1+cos th) changes only via torque:
    dE/dt = 3*u*w. Driving u ~ (E_target - E)*w pumps E toward the
    upright-rest level 2c, then a PD law holds it there.
    """
    c, s, w = (float(v) for v in obs[:3])
    th = math.atan2(s, c)
    energy = 0.5 * w * w + gains.get("c", 15.0) * (1.0 + c)
    if abs(th) < gains["th_sw"] and abs(w) < gains["w_sw"]:
        u = -gains["k1"] * th - gains["k2"] * w
    else:
        u = gains["k_pump"] * (gains.get("e_target", 30.0) - energy) * w
    return float(np.clip(u, -2.0, 2.0))


LANDER = dict(kx=0.08, kvx=0.7, kvang=0.75, max_ang=0.35, db=0.12,
              side_plus=1, kt=0.3, vmin=0.05)


def policy_lander(obs, g: dict = LANDER) -> int:
    """Attitude hold + fuel-conscious descent control for LunarLander.

    Main engine fires when the fall speed exceeds the suicide-burn bound
    vy < -kt*sqrt(y) (deceleration needed grows with sqrt of height),
    floored at vmin to avoid hovering burns just above the pad.
    """
    x, y, vx, vy, ang, vang, leg1, leg2 = (float(v) for v in obs[:8])
    if leg1 or leg2:                      # touched ground: cut engines
        return 0
    want = float(np.clip(g["kx"] * x + g["kvx"] * vx - g.get("kvang", 0.0) * vang,
                         -g["max_ang"], g["max_ang"]))
    err = want - ang
    if err > g["db"]:
        return g["side_plus"]             # rotate toward want
    if err < -g["db"]:
        return 1 if g["side_plus"] == 3 else 3
    need = g["kt"] * math.sqrt(max(y, 0.05))
    if vy < -max(need, g["vmin"]):
        return 2                          # main engine
    return 0


def list_envs() -> list[str]:
    # gymnasium 1.3 removed render_modes from EnvSpec; registry only lists
    # envs whose packages import cleanly, so family matching is enough.
    out = [eid for eid in gym.registry if eid.startswith(FAMILIES)]
    # Hardcore stays hidden until its learned policy passes; running it
    # untrained is a guaranteed failure. hardcore_final is only created
    # after a checkpoint passes the 300-seed sweep.
    hardcore_ready = os.path.isfile(".models/hardcore_final/best_model.zip")
    out = [eid for eid in out
           if not eid.startswith("BipedalWalkerHardcore") or hardcore_ready]
    # Continuous lander has no passing policy and the lander law is tuned
    # for the discrete action set: keep it out of the dropdown.
    out = [eid for eid in out if not eid.startswith("LunarLanderContinuous")]
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
    learned = LEARNED.get(env_id)
    if learned is not None:
        a, _ = learned.predict(np.asarray(state, dtype=np.float32),
                               deterministic=True)
        a = np.asarray(a)
        return int(a.item()) if a.ndim == 0 else a
    act = env.action_space
    to_right = answers["direction"].value == "right"
    obs = np.asarray(state, dtype=float).ravel()
    if isinstance(act, Discrete):
        if env_id.startswith("MountainCar") and obs.size == 2:
            return policy_mountaincar(obs)
        if env_id.startswith("Acrobot"):
            return policy_acrobot(obs)
        if env_id.startswith("LunarLander"):
            return policy_lander(obs)
        if len(state) == 4 and act.n == 2:
            return int(cartpole_decide(state, answers))  # tuned law + wall guard
        return discrete_action(env_id, int(act.n), to_right)
    if isinstance(act, Box):
        if env_id.startswith("Pendulum"):
            return np.array([policy_pendulum(obs)], dtype=act.dtype)
        if env_id.startswith("MountainCar"):  # continuous: energy pumping
            return np.array([1.0 if obs[1] > 0 else -1.0], dtype=act.dtype)
        u = 1.0 if to_right else -1.0
        return np.clip(u, act.low, act.high).astype(act.dtype).reshape(act.shape)
    return act.sample()


def action_label(env_id: str, action) -> str:
    """Human-readable action for the UI."""
    if action is None:
        return "-"
    if isinstance(action, (int, np.integer)):
        n = int(action)
        if env_id.startswith("CartPole"):
            return ["push LEFT", "push RIGHT"][n] if 0 <= n <= 1 else str(n)
        if env_id.startswith("MountainCar") and "Continuous" not in env_id:
            return ["push left", "no push", "push right"][n] if 0 <= n <= 2 else str(n)
        if env_id.startswith("Acrobot"):
            return ["hold", "hip torque -1", "hip torque +1",
                    "knee torque -1", "knee torque +1"][n] if 0 <= n <= 4 else str(n)
        if env_id.startswith("LunarLander"):
            return ["no fire", "left engine", "main engine", "right engine"][n] \
                if 0 <= n <= 3 else str(n)
        return str(n)
    vals = [round(float(v), 3) for v in np.asarray(action).ravel()]
    return "[" + ", ".join(map(str, vals)) + "]"


def state_list(state) -> list:
    """Rounded observation vector for the UI."""
    try:
        return [round(float(v), 3) for v in np.asarray(state).ravel()]
    except Exception:  # noqa: BLE001 - never let display kill the loop
        return []


def action_value(action):
    """Numeric action (float or list of floats) for live charts."""
    if action is None:
        return None
    arr = np.asarray(action, dtype=float).ravel()
    return arr.tolist() if arr.size > 1 else round(float(arr[0]), 3)


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
        self.max_steps = int(cfg.get("max_steps", 0) or 0)
        if self.max_steps <= 0:  # default: the env's own time limit
            self.max_steps = getattr(self.env.spec, "max_episode_steps",
                                     None) or 1000
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
        self._apply(state=self.state)

    def _ask(self) -> None:
        if self.answers is None or self.steps % self.decimate == 0:
            t0 = time.perf_counter()
            self.answers = self.engine.ask(
                canonical_state(self.env_id, self.state), DEFAULT_QUESTIONS)
            self.latencies.append(time.perf_counter() - t0)

    def _apply(self, action=None, state=None) -> None:
        direction = self.answers["direction"]
        self.engines[direction.engine] = self.engines.get(direction.engine, 0) + 1
        self.trail.append({
            "step": self.steps + 1,
            "answers": answers_json(self.answers),
            "engine": direction.engine,
            "conf": round(float(direction.confidence), 3),
            "state": state_list(state),
            "cstate": state_list(canonical_state(self.env_id, state)),
            "action": action_label(self.env_id, action),
            "aval": action_value(action),
        })
        self.trail = self.trail[-40:]

    def step(self) -> dict:
        with LOCK:
            if self.terminated or self.truncated or self.steps >= self.max_steps:
                return self.snapshot(done=True)
            self._ask()
            pre_state = self.state
            action = choose_action(self.env, self.env_id, self.state,
                                   self.answers)
            state, reward, term, trunc, _ = self.env.step(action)
            self.state = state
            self.reward += float(reward)
            self.steps += 1
            self.terminated = bool(term)
            self.truncated = bool(trunc)
            self._apply(action, pre_state)
            return self.snapshot()

    def outcome(self):
        """(success, message) using each env's own objective.

        gymnasium semantics: terminated = the env's own end condition
        (goal OR failure), truncated = the time limit ran out. Which one
        means success depends on the env.
        """
        eid = self.env_id
        if self.terminated:
            if eid.startswith("CartPole"):
                return False, "pole fell"
            if eid.startswith("LunarLander"):
                ok = self.reward >= 200
                return ok, ("landed successfully" if ok else
                            "on the ground, below the 200 solve score")
            if eid.startswith("BipedalWalker"):
                ok = self.reward >= 300
                return ok, ("completed the course" if ok else
                            "stopped, below the 300 solve score")
            return True, "goal reached"
        if eid.startswith("CartPole"):
            return True, "survived the full episode"
        if eid.startswith("Pendulum"):
            ok = self.reward >= -400
            return ok, ("held upright all episode" if ok else
                        "time up, below the -400 target")
        if eid.startswith("BipedalWalker"):
            ok = self.reward >= 300
            return ok, ("completed the course" if ok else
                        "time up, below the 300 solve score")
        return False, "time limit reached before the goal"

    def snapshot(self, done: bool = False) -> dict:
        confs = [t["conf"] for t in self.trail]
        risk = None
        if self.answers is not None:
            risk = self.answers["at_risk"].value
        success, message = (None, None)
        if self.terminated or self.truncated:
            success, message = self.outcome()
        return {
            "frame": frame_b64(self.env),
            "env": self.env_id,
            "engine": self.engine.engine,
            "policy": "PPO" if self.env_id in LEARNED else "rule",
            "steps": self.steps,
            "reward": round(self.reward, 2),
            "done": done or self.terminated or self.truncated,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "success": success,
            "message": message,
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


@app.after_request
def no_cache_html(resp):
    # templates change often during development: never let the browser
    # serve a stale page on a plain refresh
    if resp.content_type and resp.content_type.startswith("text/html"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/envs")
def api_envs():
    return jsonify({"envs": list_envs()})


def _read_cfg(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001 - missing checkpoint is fine
        return {}


@app.route("/api/arch")
def api_arch():
    """Real layer facts for the WebUI architecture panel.

    Encoder + typed-decisions head come from the laya checkpoint configs;
    the motor layer is the PPO checkpoint (trained) or the hand law.
    """
    enc_cfg = _read_cfg(os.path.join(".models", "laya", "encoder",
                                     "config.json"))
    types = enc_cfg.get("layer_types", [])
    encoder = {
        "layers": enc_cfg.get("num_hidden_layers", 0),
        "full_attn": types.count("full_attention"),
        "sliding_attn": types.count("sliding_attention"),
        "hidden": enc_cfg.get("hidden_size"),
        "heads": enc_cfg.get("num_attention_heads"),
        "intermediate": enc_cfg.get("intermediate_size"),
        "vocab": enc_cfg.get("vocab_size"),
        "context": enc_cfg.get("max_position_embeddings"),
    } if enc_cfg else {}
    head_cfg = _read_cfg(os.path.join(".models", "laya", "typed-decisions",
                                      "encoder", "config.json"))
    head = {"layers": head_cfg.get("num_hidden_layers", 0)} if head_cfg else {}
    env = SESSION.env_id if SESSION else None
    if env and env in LEARNED:
        motor = {"type": "PPO policy (MLP)", "trained": True,
                 "source": "learned per env - stable-baselines3 checkpoint"}
    else:
        motor = {"type": "physics control law", "trained": False,
                 "source": "hand-tuned rules per env family"}
    return jsonify({"encoder": encoder, "head": head, "motor": motor,
                    "env": env, "learned": sorted(LEARNED.keys()),
                    "engine": getattr(getattr(SESSION, "engine", None),
                                      "engine", None)})


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

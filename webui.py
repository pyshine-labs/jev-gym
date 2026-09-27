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
from jev_agent import (canonical_state, make_engine, questions_for)


app = Flask(__name__)

LOCK = threading.Lock()
SESSION: "Session | None" = None

# Families with 1-D Box observations and rgb_array rendering. Toy-text envs
# (integer observations) cannot answer numeric typed questions, so they are
# not offered. CarRacing renders pixels, but the driving state (speed, safe
# speed, heading error, lateral offset) comes from the Box2D world itself.
FAMILIES = ("CartPole", "MountainCar", "Acrobot", "Pendulum",
            "LunarLander", "BipedalWalker", "CarRacing")

# Learned (PPO) policies, optional. Where a trained checkpoint exists it
# drives the motor action; the Jev typed questions still assess every step.
# All checkpoint paths are anchored to this file's folder so the server
# works regardless of the process working directory.
_ROOT = os.path.dirname(os.path.abspath(__file__))
_M = lambda *parts: os.path.join(_ROOT, ".models", *parts)  # noqa: E731

LEARNED: dict = {}
try:
    from stable_baselines3 import PPO as _PPO

    for _eid, _paths in (
        ("Pendulum-v1", (_M("pendulum_final", "best_model"),)),
        ("LunarLander-v3", (_M("lander_final", "best_model"),)),
        ("BipedalWalker-v3", (_M("walker_final", "best_model"),
                              _M("best3", "best_model"))),
        ("BipedalWalkerHardcore-v3", (_M("hardcore_final", "best_model"),)),
    ):
        for _p in _paths:
            try:
                LEARNED[_eid] = _PPO.load(_p)
                break
            except Exception:  # noqa: BLE001 - missing checkpoint is fine
                pass
except ImportError:  # stable-baselines3 not installed: rule laws only
    pass


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


# ---------------------------------------------------------------------------
# CarRacing-v3: pure-pursuit driving law over the Box2D world. The obs is
# pixels, but the track centerline and car pose come from env.unwrapped, so
# the driving state is fully observable. Driving feature vector (what the
# decision engines see instead of pixels):
#   [speed, safe_speed, heading_error_rad, lateral_offset, off_road]
# ---------------------------------------------------------------------------

# Pure-pursuit gains (best of the tuning sweep; seed scores 461.1 / 547.3).
# A full v3 lap in 1000 steps is physically impossible, so "passing" here
# = a strong on-road lap segment.
CAR = dict(look=0.40, lmin=5, lmax=18, kpp=22, jrec=10,
           vmax=55, alat=14, brk=25, kb=0.22, kg=0.6, gmax=0.9, off=8.0)
CAR_PASS = 300.0  # no official solve score; ~2/3 of the law's own segment


def _car_wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _car_track(env):
    """(centerline points, unit segment directions, mean segment length)."""
    track = env.unwrapped.track
    cl = np.array([[t[2], t[3]] for t in track], dtype=float)
    d = np.diff(cl, axis=0, append=cl[:1])
    n = np.linalg.norm(d, axis=1, keepdims=True)
    sd = d / np.maximum(n, 1e-6)
    sl = float(np.linalg.norm(d, axis=1).mean())
    return cl, sd, sl


def car_context(env) -> dict:
    """Pure-pursuit context for the car right now (the passing law's view)."""
    cl, sd, sl = _car_track(env)
    car = env.unwrapped.car
    pos = np.array([car.hull.position[0], car.hull.position[1]])
    ang = car.hull.angle
    fang = ang + math.pi / 2.0            # forward = hull local +Y
    vel = car.hull.linearVelocity
    speed = math.hypot(vel[0], vel[1])
    d = np.linalg.norm(cl - pos, axis=1)
    i = int(np.argmin(d))
    n = len(cl)
    p = CAR
    # steering alpha: angle to the speed-scaled lookahead point (pure pursuit)
    L = int(np.clip(speed * p["look"], p["lmin"], p["lmax"]))
    dx, dy = cl[(i + L) % n] - pos
    alpha = _car_wrap(math.atan2(dy, dx) - fang)
    kappa = 2.0 * math.sin(alpha) / max(math.hypot(dx, dy), 1.0)
    steer = float(np.clip(-p["kpp"] * kappa, -1, 1))
    # braking-aware corner-speed envelope over sliding windows; the window
    # bend is the SUM of per-segment direction changes, so S-bends cannot
    # cancel out and hide from the envelope (same math as _tune_car.py)
    hd = np.arctan2(sd[:, 1], sd[:, 0])
    turn = np.abs(_car_wrap(np.diff(hd, append=hd[:1])))
    total = float(turn.sum())
    cs = np.concatenate([[0.0], np.cumsum(turn)])
    cse = np.concatenate([cs, cs[1:] + total])   # wrap the closed loop
    v_t = p["vmax"]
    W, B = 6, p["brk"]
    for k in range(0, 48, 6):
        a = float(cse[i + k + W] - cse[i + k])
        if a > 0.05:
            R = sl * W / a
            vc = math.sqrt(p["alat"] * R)
            v_t = min(v_t, math.sqrt(vc * vc + 2.0 * B * sl * k))
    to = cl[i] - pos
    lat = float(-math.sin(ang) * to[1] - math.cos(ang) * to[0]) \
        / max(d[i], 0.5)
    return dict(pos=pos, fang=fang, ang=ang, speed=speed, alpha=alpha,
                steer=steer, lat=lat, v_t=v_t, off=bool(d[i] > p["off"]),
                i=i, cl=cl, sd=sd, sl=sl, dist=d[i])


def car_features(env) -> np.ndarray:
    c = car_context(env)
    return np.array([c["speed"], c["v_t"], c["alpha"], c["lat"],
                     1.0 if c["off"] else 0.0])


def policy_car(env) -> np.ndarray:
    """The passing pure-pursuit law: [steer, gas, brake]."""
    c = car_context(env)
    p = CAR
    if c["off"]:
        # recovery: pure pursuit to a point ahead on the road; loop toward
        # its side at full lock when it falls behind
        cl = c["cl"]
        j2 = (c["i"] + p["jrec"]) % len(cl)
        dx, dy = cl[j2] - c["pos"]
        alpha = _car_wrap(math.atan2(dy, dx) - c["fang"])
        if abs(alpha) > 1.4:
            steer = -math.copysign(1.0, alpha)
        else:
            kappa = 2.0 * math.sin(alpha) / max(math.hypot(dx, dy), 1.0)
            steer = float(np.clip(-p["kpp"] * kappa, -1, 1))
        gas = 0.7 if c["speed"] < 10 else 0.3
        return np.array([steer, gas, 0.0], dtype=np.float32)
    gas = float(np.clip((c["v_t"] - c["speed"]) * p["kg"], 0.0, p["gmax"]))
    brake = float(np.clip((c["speed"] - c["v_t"]) * p["kb"], 0.0, 1.0))
    return np.array([c["steer"], gas, brake], dtype=np.float32)


def car_jev_action(env, answers) -> np.ndarray:
    """Jev drives: the typed steer/throttle answers choose the maneuver,
    the passing law supplies the magnitude (same pattern as Pendulum)."""
    c = car_context(env)
    p = CAR
    steer_ans = answers.get("steer")
    val = steer_ans.value if steer_ans is not None else None
    # the law's steering demand: positive alpha/kappa = target to the left
    mag = float(np.clip(abs(c["steer"]), 0.0, 1.0))
    if val == "left":
        steer = -mag if mag > 0.05 else 0.0
    elif val == "right":
        steer = mag if mag > 0.05 else 0.0
    else:
        steer = 0.0
    thr_ans = answers.get("throttle")
    tval = thr_ans.value if thr_ans is not None else None
    if tval == "accelerate":
        gas = float(np.clip((c["v_t"] - c["speed"]) * p["kg"], 0.15,
                            p["gmax"]))
        brake = 0.0
    elif tval == "brake":
        gas = 0.0
        brake = float(np.clip((c["speed"] - c["v_t"]) * p["kb"], 0.15, 1.0))
    else:                                  # coast: hold speed gently
        gas = float(np.clip((c["v_t"] - c["speed"]) * p["kg"], 0.0, 0.2))
        brake = 0.0
    return np.array([steer, gas, brake], dtype=np.float32)


def list_envs() -> list[str]:
    # gymnasium 1.3 removed render_modes from EnvSpec; registry only lists
    # envs whose packages import cleanly, so family matching is enough.
    out = [eid for eid in gym.registry if eid.startswith(FAMILIES)]
    # Hardcore stays hidden until its learned policy passes; running it
    # untrained is a guaranteed failure. hardcore_final is only created
    # after a checkpoint passes the 300-seed sweep.
    hardcore_ready = os.path.isfile(
        _M("hardcore_final", "best_model.zip"))
    out = [eid for eid in out
           if not eid.startswith("BipedalWalkerHardcore") or hardcore_ready]
    # Continuous lander has no passing policy and the lander law is tuned
    # for the discrete action set: keep it out of the dropdown.
    out = [eid for eid in out if not eid.startswith("LunarLanderContinuous")]
    # The CarRacing law + training target v3 (normalized reward scale).
    out = [eid for eid in out
           if not eid.startswith("CarRacing") or eid == "CarRacing-v3"]
    return sorted(set(out)) or ["CartPole-v1"]


def frame_b64(env) -> str | None:
    frame = env.render()
    if frame is None:
        return None
    img = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=72)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def policy_pendulum_jev(obs, instability: float) -> float:
    """Jev drives: its typed instability score gates pump intensity, the PD
    catch region always catches. Works on both scales (local 0..4 level or
    laya's continuous score)."""
    c, s, w = (float(v) for v in obs[:3])
    th = math.atan2(s, c)
    energy = 0.5 * w * w + 15.0 * (1.0 + c)
    if abs(th) < 0.7 and abs(w) < 2.0:    # catch region: PD always holds
        return float(np.clip(-16.0 * th - 4.0 * w, -2.0, 2.0))
    pump = 0.1 * (30.0 - energy) * w      # swing-up by energy pumping
    if instability >= 2.5:                # Jev: unstable -> full pump
        return float(np.clip(pump, -2.0, 2.0))
    return float(np.clip(0.3 * pump, -2.0, 2.0))  # Jev: calm -> damped pump


def choose_action(env, env_id, state, answers, jev_drives: bool = False):
    choice = answers.get("direction") or answers.get("pump")
    to_right = choice.value == "right" or choice.value == "pos" \
        if choice is not None else True
    obs = np.asarray(state, dtype=float).ravel()
    act = env.action_space
    if jev_drives:
        if env_id.startswith("BipedalWalker"):
            learned = LEARNED.get(env_id)  # 6-D gait: only the learned motor
            if learned is not None:
                a, _ = learned.predict(np.asarray(state, dtype=np.float32),
                                       deterministic=True)
                a = np.asarray(a)
                return int(a.item()) if a.ndim == 0 else a
        if env_id.startswith("CarRacing"):
            return car_jev_action(env, answers)
        if isinstance(act, Discrete):
            if len(state) == 4 and act.n == 2:
                return int(cartpole_decide(state, answers))  # Jev direction
            if env_id.startswith("MountainCar") and obs.size == 2:
                return 2 if to_right else 0          # pump by Jev direction
            if env_id.startswith("Acrobot"):
                return 2 if to_right else 0          # pump by Jev direction
            if env_id.startswith("LunarLander"):
                side = answers.get("side_engine")
                x, y, vx, vy, ang, vang, leg1, leg2 = (
                    float(v) for v in obs[:8])
                if leg1 or leg2:                     # grounded: cut engines
                    return 0
                if side is not None and side.value in ("left-engine",
                                                       "right-engine"):
                    return 1 if side.value == "left-engine" else 3
                need = 0.3 * math.sqrt(max(y, 0.05))
                if vy < -max(need, 0.05):
                    return 2                         # main engine
                return 0
            return discrete_action(env_id, int(act.n), to_right)
        if isinstance(act, Box):
            if env_id.startswith("Pendulum"):
                inst = float(answers["instability"].value or 0.0)
                return np.array([policy_pendulum_jev(obs, inst)],
                                dtype=act.dtype)
            if env_id.startswith("MountainCar"):
                return np.array([1.0 if to_right else -1.0], dtype=act.dtype)
            u = 1.0 if to_right else -1.0
            return np.clip(u, act.low, act.high).astype(act.dtype).reshape(
                act.shape)
    learned = LEARNED.get(env_id)
    if learned is not None:
        a, _ = learned.predict(np.asarray(state, dtype=np.float32),
                               deterministic=True)
        a = np.asarray(a)
        return int(a.item()) if a.ndim == 0 else a
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
        if env_id.startswith("CarRacing"):
            return policy_car(env)               # pure-pursuit driving law
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
    if env_id.startswith("CarRacing"):
        s, g, b = (float(v) for v in np.asarray(action).ravel()[:3])
        ped = f"gas {g:.2f}" if g >= b else f"brake {b:.2f}"
        return f"steer {s:+.2f}, {ped}"
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
    for a in answers.values():
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
        self.jev_drives = bool(cfg.get("jev_drives", True))
        self.engines: dict[str, int] = {}
        self.ask_calls = 0
        self.latencies: list[float] = []
        self.trail: list[dict] = []

        self._ask()
        self._apply(state=self._disp_state())

    def _disp_state(self):
        """What the UI/trail should show as 'the state': raw obs, except
        CarRacing where pixels are replaced by the driving feature vector."""
        if self.env_id.startswith("CarRacing"):
            return car_features(self.env)
        return self.state

    def _ask(self) -> None:
        if self.answers is None or self.steps % self.decimate == 0:
            t0 = time.perf_counter()
            ask_state = self._disp_state()
            self.answers = self.engine.ask(
                self.env_id, ask_state, questions_for(self.env_id))
            self.ask_calls += 1
            self.latencies.append(time.perf_counter() - t0)

    def _apply(self, action=None, state=None) -> None:
        primary = (self.answers.get("direction")
                   or self.answers.get("pump")
                   or self.answers.get("side_engine")
                   or next(iter(self.answers.values())))
        self.engines[primary.engine] = self.engines.get(primary.engine, 0) + 1
        self.trail.append({
            "step": self.steps + 1,
            "answers": answers_json(self.answers),
            "engine": primary.engine,
            "conf": round(float(primary.confidence), 3),
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
            pre_state = self._disp_state()
            action = choose_action(self.env, self.env_id, self.state,
                                   self.answers, self.jev_drives)
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
            if eid.startswith("CarRacing"):
                return True, "completed the lap"
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
        if eid.startswith("CarRacing"):
            ok = self.reward >= CAR_PASS
            return ok, (f"strong lap segment, scored {self.reward:.0f}"
                        if ok else
                        f"time up, scored {self.reward:.0f} "
                        f"(bar {CAR_PASS:.0f})")
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
            "policy": "JEV" if self.jev_drives else
                      ("PPO" if self.env_id in LEARNED else "rule"),
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
            "engine_calls": self.ask_calls,
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
    enc_cfg = _read_cfg(_M("laya", "encoder", "config.json"))
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
    head_cfg = _read_cfg(_M("laya", "typed-decisions", "encoder",
                            "config.json"))
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

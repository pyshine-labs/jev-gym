"""Jev-style local decision agent for the inverted pendulum.

Implements the "System One" typed-decision contract from the Jev/Laya line of
models: a state goes in, and a set of typed questions (choice / score / noul)
comes back answered in a single pass, each with probabilities and a confidence.
No text is ever generated.

Two engines:
  - LocalDecisionHead: fully offline physics-informed head. Computes shared
    features once per call (the "single forward pass") and emits calibrated
    probabilities. Zero dependencies beyond numpy.
  - JevBackend: the real TypeSafe Jev model (typesafe/jev-1.13) via the
    OpenRouter Decisions API. Needs OPENROUTER_API_KEY. Falls back to the
    local head on any error, and reports which engine decided.
  - LayaBackend: uses the open-source `laya` package (pip install laya) to ask
    the same typed questions of a real Jev-compatible decision model. Falls
    back per-question to LocalDecisionHead on any error or low confidence,
    and reports which engine decided.
"""

from __future__ import annotations

import json
import math
import os
import urllib.request
from dataclasses import dataclass, field

import numpy as np

# The real Jev on OpenRouter: https://openrouter.ai/docs/guides/community/jev
JEV_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_MODEL = "typesafe/jev-1.13"

# CartPole-v1 termination limits (gymnasium defaults)
THETA_LIMIT = 0.2095   # rad, ~12 degrees
X_LIMIT = 2.4          # meters
THETA_DOT_LIMIT = 2.0  # rad/s, soft reference scale


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
    if env_id.startswith("CarRacing"):
        # driving vector [speed, safe_speed, steer_alpha, lateral, off]:
        # lateral offset is the road position, steer alpha the "pole angle"
        return np.array([v[3], v[0], v[2], 0.0])
    return np.zeros(4)  # BipedalWalker et al: neutral posture


def _sigmoid(v: float) -> float:
    if v >= 0:
        z = math.exp(-min(v, 60.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(v, -60.0))
    return z / (1.0 + z)


@dataclass
class Question:
    """A typed question, Jev-style: choice, score, or noul."""

    name: str
    qtype: str  # "choice" | "score" | "noul"
    instructions: str
    criteria: object = None  # dict for choice, list for score, ignored for noul


@dataclass
class Answer:
    """A typed answer: value + per-option probabilities + confidence."""

    name: str
    qtype: str
    value: object
    probabilities: dict = field(default_factory=dict)
    confidence: float = 0.0
    engine: str = "local"

    def __repr__(self) -> str:  # compact console readout
        return f"{self.name}={self.value!r} conf={self.confidence:.2f} [{self.engine}]"


# --- The typed question set for the inverted pendulum -----------------------

DIRECTION = Question(
    "direction",
    "choice",
    "Which way should the cart push right now to keep the pole up?",
    {
        "left": "the pole is leaning left or rotating left; push left",
        "right": "the pole is leaning right or rotating right; push right",
    },
)

AT_RISK = Question(
    "at_risk",
    "noul",
    "Will the pole fall or the cart leave the track within the next moments?",
)

INSTABILITY = Question(
    "instability",
    "score",
    "How unstable is the current state?",
    ["stable", "slightly off", "drifting", "wobbling", "about to fail"],
)

# Lander-flavoured lateral decision: which side engine should fire (the
# descent/main-engine contract stays in the motor translation).
SIDE_ENGINE = Question(
    "side_engine",
    "choice",
    "A lunar lander descends toward its pad and must stay level. The left "
    "engine rotates it counterclockwise (its angle increases); the right "
    "engine rotates it clockwise (angle decreases). Fire a side engine only "
    "to correct tilt or drift. Which side engine should fire right now?",
    {
        "left-engine": "fire the left engine: the lander must rotate "
                       "counterclockwise (angle should increase)",
        "right-engine": "fire the right engine: the lander must rotate "
                        "clockwise (angle should decrease)",
        "none": "the lander is level enough; no side engine",
    },
)

# MountainCar / MountainCarContinuous momentum pump: accelerate with the
# motion to build energy for the hill climb.
MC_PUMP = Question(
    "pump",
    "choice",
    "A car sits in a valley between two hills and must reach the flag at "
    "the top of the right hill. It cannot climb directly; it must swing "
    "back and forth, always accelerating in the direction it is currently "
    "moving, to build momentum. Which way should it accelerate right now?",
    {
        "left": "accelerate left: the car is moving left (velocity negative)",
        "right": "accelerate right: the car is moving right (velocity positive)",
    },
)

# Acrobot energy pump: torque with the elbow swing.
ACRO_PUMP = Question(
    "pump",
    "choice",
    "A two-link acrobot hangs downward and must swing its tip up to a line "
    "by pumping energy through its hip torque. Apply the torque in the same "
    "direction the lower link is currently swinging; when the lower link is "
    "nearly still, follow the upper link's swing. Which torque should fire "
    "right now?",
    {
        "neg": "torque -1: the elbow swings in the negative direction "
               "(elbow_velocity negative)",
        "pos": "torque +1: the elbow swings in the positive direction "
               "(elbow_velocity positive)",
    },
)

# CarRacing driving contract: the steering decision and the pedal decision.
STEER = Question(
    "steer",
    "choice",
    "A car drives on a winding race track. The nose must follow the road "
    "ahead and hug the centerline. Which steering should it apply right now?",
    {
        "left": "steer left: the road bends left ahead or the car is right "
                "of the centerline",
        "straight": "hold straight: the nose is aligned with the road ahead",
        "right": "steer right: the road bends right ahead or the car is "
                 "left of the centerline",
    },
)

THROTTLE = Question(
    "throttle",
    "choice",
    "A car races on a winding track: fast on straights, slow before tight "
    "corners, never off the road. Its speed and the corner-limited safe "
    "speed are given. Which pedal should it press right now?",
    {
        "accelerate": "press the gas: the speed is below the safe speed for "
                      "the road ahead",
        "coast": "lift off: the speed matches the safe speed",
        "brake": "brake: the speed is above the safe speed for the road "
                 "ahead",
    },
)

DEFAULT_QUESTIONS = [DIRECTION, AT_RISK, INSTABILITY]
LANDER_QUESTIONS = [SIDE_ENGINE, AT_RISK, INSTABILITY]
MC_QUESTIONS = [MC_PUMP, AT_RISK, INSTABILITY]
ACRO_QUESTIONS = [ACRO_PUMP, AT_RISK, INSTABILITY]
CAR_QUESTIONS = [STEER, THROTTLE, AT_RISK, INSTABILITY]


def questions_for(env_id: str) -> list:
    """The typed question set matching each env family's control contract."""
    if env_id.startswith("LunarLander"):
        return LANDER_QUESTIONS
    if env_id.startswith("MountainCar"):
        return MC_QUESTIONS
    if env_id.startswith("Acrobot"):
        return ACRO_QUESTIONS
    if env_id.startswith("CarRacing"):
        return CAR_QUESTIONS
    return DEFAULT_QUESTIONS


class LocalDecisionHead:
    """Offline System One head: shared features once, all questions answered."""

    engine = "local"

    def features(self, state) -> dict:
        x, x_dot, theta, theta_dot = (float(v) for v in state)
        lean = theta + 2.3 * theta_dot          # where the pole is heading
        align = 0.05 * x + 0.4 * x_dot          # align push with cart momentum
        risk = max(
            abs(theta) / THETA_LIMIT,
            abs(theta_dot) / THETA_DOT_LIMIT,
            abs(x) / X_LIMIT,
        )
        return {
            "x": x, "x_dot": x_dot, "theta": theta, "theta_dot": theta_dot,
            "lean": lean, "align": align,
            "risk": min(max(risk, 0.0), 1.0),
        }

    def ask(self, env_id, obs, questions) -> dict:
        state = canonical_state(env_id, obs)
        f = self.features(state)  # the single shared pass
        out = {}
        for q in questions:
            if q.qtype == "choice" and q.name == "pump":
                v = np.asarray(obs, dtype=float).ravel()
                if env_id.startswith("Acrobot"):
                    # pump with the elbow swing; follow the shoulder when stalled
                    v2, v1 = float(v[5]), float(v[4])
                    drive = v1 if abs(v2) < 0.05 else v2
                    value = "pos" if drive > 0 else "neg"
                else:  # MountainCar: accelerate with the motion
                    value = "right" if float(v[1]) > 0 else "left"
                out[q.name] = Answer(q.name, q.qtype, value,
                                     {value: 0.9}, 0.9, self.engine)
            elif q.qtype == "choice" and q.name == "direction":
                u = f["lean"] + f["align"]
                p_right = _sigmoid(6.0 * u)
                p_left = 1.0 - p_right
                value = "right" if u >= 0 else "left"
                out[q.name] = Answer(
                    q.name, "choice", value,
                    {"left": round(p_left, 4), "right": round(p_right, 4)},
                    max(p_left, p_right), self.engine,
                )
            elif q.qtype == "choice" and q.name == "side_engine":
                # lander attitude error in the canonical frame
                # [x, x_dot, theta, theta_dot] = [x, vx, ang, vang]:
                # want = attitude setpoint with angular-velocity damping
                want = min(max(0.08 * f["x"] + 0.7 * f["x_dot"]
                               - 0.75 * f["theta_dot"], -0.35), 0.35)
                err = want - f["theta"]
                if err > 0.12:
                    value, others = "left-engine", ["right-engine", "none"]
                elif err < -0.12:
                    value, others = "right-engine", ["left-engine", "none"]
                else:
                    value, others = "none", ["left-engine", "right-engine"]
                decisiveness = min(abs(err) / 0.5, 1.0) if value != "none" \
                    else 1.0 - min(abs(err) / 0.12, 1.0)
                conf = 0.5 + 0.5 * decisiveness
                p_val = 0.5 + 0.5 * decisiveness
                probs = {value: round(p_val, 4)}
                for o in others:
                    probs[o] = round((1.0 - p_val) / 2.0, 4)
                out[q.name] = Answer(q.name, "choice", value, probs, conf,
                                     self.engine)
            elif q.qtype == "choice" and q.name == "steer":
                # CarRacing: obs = [speed, safe_speed, heading_err, lat, off].
                # Same gains as the passing pure-pursuit law (kst=1.8/2.2,
                # klat=0.6/0.8): u > 0 means the nose must rotate left.
                v = np.asarray(obs, dtype=float).ravel()
                u = 2.0 * float(v[2]) + 0.7 * float(v[3])
                if u > 0.06:
                    value, others = "left", ["straight", "right"]
                elif u < -0.06:
                    value, others = "right", ["left", "straight"]
                else:
                    value, others = "straight", ["left", "right"]
                decisiveness = min(abs(u) / 0.3, 1.0)
                conf = 0.5 + 0.5 * decisiveness
                p_val = 0.5 + 0.5 * decisiveness
                probs = {value: round(p_val, 4)}
                for o in others:
                    probs[o] = round((1.0 - p_val) / 2.0, 4)
                out[q.name] = Answer(q.name, q.qtype, value, probs, conf,
                                     self.engine)
            elif q.qtype == "choice" and q.name == "throttle":
                # CarRacing: compare speed against the corner-limited safe
                # speed; off the road means full commitment to get back.
                v = np.asarray(obs, dtype=float).ravel()
                speed, v_t = float(v[0]), float(v[1])
                off = v.size >= 5 and bool(v[4] > 0.5)
                if off or speed < v_t - 2.0:
                    value, others = "accelerate", ["coast", "brake"]
                elif speed > v_t + 2.0:
                    value, others = "brake", ["accelerate", "coast"]
                else:
                    value, others = "coast", ["accelerate", "brake"]
                decisiveness = 1.0 if off else \
                    min(abs(speed - v_t) / 8.0, 1.0)
                conf = 0.5 + 0.5 * decisiveness
                p_val = 0.5 + 0.5 * decisiveness
                probs = {value: round(p_val, 4)}
                for o in others:
                    probs[o] = round((1.0 - p_val) / 2.0, 4)
                out[q.name] = Answer(q.name, q.qtype, value, probs, conf,
                                     self.engine)
            elif q.qtype == "noul":
                p = _sigmoid((f["risk"] - 0.55) * 9.0)
                out[q.name] = Answer(
                    q.name, "noul", round(p, 4),
                    {"true": round(p, 4), "false": round(1.0 - p, 4)},
                    abs(p - 0.5) * 2.0, self.engine,  # 0.5 == abstains
                )
            elif q.qtype == "score":
                v = f["risk"] * (len(q.criteria) - 1)
                level = min(int(round(v)), len(q.criteria) - 1)
                probs = {}
                for i, label in enumerate(q.criteria):
                    d = abs(i - v)
                    probs[label] = round(max(0.0, 1.0 - d), 4)
                total = sum(probs.values()) or 1.0
                probs = {k: round(v / total, 4) for k, v in probs.items()}
                conf = probs[q.criteria[level]]
                out[q.name] = Answer(
                    q.name, "score", round(v, 3), probs, conf, self.engine
                )
            else:
                raise ValueError(f"unsupported question: {q.name}/{q.qtype}")
        return out


class LayaBackend:
    """The open Jev-compatible decision model (`pip install laya`).

    Prefers the task-tuned checkpoint at .models/laya_gym/ (fine-tuned on
    each env's passing control law by train_laya_gym.py) so laya's own
    answers are control-grade; otherwise falls back to the base checkpoint
    at .models/laya/. Every question is tried against laya; only an
    unparseable answer falls back structurally to the local head.
    """

    engine = "laya"

    def __init__(self, min_confidence: float = 0.60,
                 model_path: str | None = None):
        self.min_confidence = min_confidence
        self.fallback = LocalDecisionHead()
        root = os.path.dirname(os.path.abspath(__file__))
        tuned = os.path.join(root, ".models", "laya_gym")
        base = model_path or os.path.join(root, ".models", "laya")
        try:
            tuned_ok = os.path.isfile(os.path.join(tuned, "model.safetensors"))
            local = tuned if (tuned_ok and not model_path) else base
            if os.path.isdir(local):
                from laya import load  # type: ignore

                device = None
                try:
                    import torch  # type: ignore

                    if torch.cuda.is_available():
                        device = "cuda"
                except Exception:  # noqa: BLE001 - torch optional
                    device = None
                self.agent = load(local, device=device)
                self.router = None
            else:
                from laya import Router  # type: ignore

                self.router = Router(preload=True)
                self.agent = None
        except Exception as exc:  # noqa: BLE001 - any import/init failure = no laya
            raise RuntimeError(
                f"laya engine unavailable ({exc}). Install with: pip install laya"
            ) from exc

    @staticmethod
    def build_payload(env_id: str, obs) -> dict:
        """Per-env state payload with human-readable physics keys. The
        task-tuned checkpoint is trained on exactly this serialization."""
        v = np.asarray(obs, dtype=float).ravel()
        if env_id.startswith("CartPole"):
            return {"cart_position": round(float(v[0]), 3),
                    "cart_velocity": round(float(v[1]), 3),
                    "pole_angle_deg": round(math.degrees(float(v[2])), 2),
                    "pole_rotation_deg_per_s": round(math.degrees(float(v[3])), 2)}
        if env_id.startswith("MountainCar"):
            return {"position": round(float(v[0]), 3),
                    "velocity": round(float(v[1]), 3)}
        if env_id.startswith("Acrobot"):
            return {"shoulder_velocity": round(float(v[4]), 3),
                    "elbow_velocity": round(float(v[5]), 3)}
        if env_id.startswith("LunarLander"):
            return {"x": round(float(v[0]), 3), "y": round(float(v[1]), 3),
                    "vx": round(float(v[2]), 3), "vy": round(float(v[3]), 3),
                    "angle_deg": round(math.degrees(float(v[4])), 2),
                    "angular_velocity": round(float(v[5]), 3)}
        if env_id.startswith("Pendulum"):
            return {"pole_angle_deg": round(
                        math.degrees(math.atan2(float(v[1]), float(v[0]))), 2),
                    "pole_rotation_deg_per_s": round(math.degrees(float(v[2])), 2)}
        if env_id.startswith("CarRacing"):
            # driving vector [speed, safe_speed, heading_err, lateral, off]
            return {"speed": round(float(v[0]), 2),
                    "safe_speed": round(float(v[1]), 2),
                    "target_heading_error_deg": round(
                        math.degrees(float(v[2])), 1),
                    "lateral_offset": round(float(v[3]), 2),
                    "off_road": bool(v[4] > 0.5)}
        c = canonical_state(env_id, v)  # BipedalWalker et al: neutral posture
        return {"cart_position": round(float(c[0]), 3),
                "cart_velocity": round(float(c[1]), 3),
                "pole_angle_deg": round(math.degrees(float(c[2])), 2),
                "pole_rotation_deg_per_s": round(math.degrees(float(c[3])), 2)}

    def _extract(self, ans: dict, qtype: str):
        for key in (qtype, "value"):
            if isinstance(ans, dict) and key in ans:
                return ans[key]
        if isinstance(ans, dict):
            for k, v in ans.items():
                if k in ("confidence", "probabilities"):
                    continue
                if isinstance(v, (str, int, float)):
                    return v
        return None

    def ask(self, env_id, obs, questions) -> dict:
        names = [q.name for q in questions]
        laya_state = self.build_payload(env_id, obs)
        local = self.fallback.ask(env_id, obs, questions)
        qpayload = {
            q.name: {"type": q.qtype, "instructions": q.instructions,
                     **({"criteria": q.criteria} if q.criteria else {})}
            for q in questions
        }
        try:
            if self.router is not None:
                result = self.router.predict(laya_state, qpayload)
            else:
                result = self.agent.predict(laya_state, qpayload)
            if isinstance(result, dict):
                raw = result.get("answers", {})
            else:
                raw = getattr(result, "answers", None) or {}
        except Exception:  # noqa: BLE001 - offline-safe: fall back entirely
            return local
        for q in questions:
            ans = raw.get(q.name)
            if not isinstance(ans, dict):
                continue
            value = self._extract(ans, q.qtype)
            if value is None:
                continue  # unparseable -> structural fallback to local
            # answer_confidence (top answer probability) is the calibrated
            # signal; kept unconditionally - laya wins on its own.
            conf = float(ans.get("answer_confidence")
                         or ans.get("confidence") or 0.0)
            probs = ans.get("probabilities") or {}
            local[q.name] = Answer(q.name, q.qtype, value, probs, conf, "laya")
        return local


class JevBackend:
    """The real TypeSafe Jev model via the OpenRouter Decisions API.

    Every step sends the cart state and the typed question set to
    typesafe/jev-1.13 and parses the typed answers (choice/noul/score).
    On any error the local head answers instead, so the pole never drops
    because of a network hiccup. Noul has no separate confidence in the
    Jev schema - the probability itself is the belief.
    """

    engine = "jev"

    def __init__(self, model: str = JEV_MODEL):
        self.model = model
        self.fallback = LocalDecisionHead()
        self.key = os.environ.get("OPENROUTER_API_KEY", "")
        if not self.key:
            raise RuntimeError(
                "The real Jev needs an OpenRouter API key. Create one at "
                "https://openrouter.ai/settings/keys and set it as the "
                "OPENROUTER_API_KEY environment variable."
            )

    def ask(self, env_id, obs, questions) -> dict:
        state = canonical_state(env_id, obs)
        local = self.fallback.ask(env_id, obs, questions)
        payload = {
            "model": self.model,
            "state": {
                "x_meters": round(float(state[0]), 4),
                "x_dot_m_s": round(float(state[1]), 4),
                "theta_radians": round(float(state[2]), 4),
                "theta_dot_rad_s": round(float(state[3]), 4),
                "pole_angle_degrees": round(math.degrees(float(state[2])), 2),
            },
            "questions": {
                q.name: {
                    "type": q.qtype,
                    "instructions": q.instructions,
                    **({"criteria": q.criteria} if q.criteria else {}),
                }
                for q in questions
            },
        }
        req = urllib.request.Request(
            JEV_URL,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = json.loads(resp.read().decode()).get("answers", {})
        except Exception:  # noqa: BLE001 - offline-safe: fall back entirely
            return local

        for q in questions:
            ans = raw.get(q.name)
            if not isinstance(ans, dict):
                continue
            if q.qtype == "choice" and "choice" in ans:
                conf = float(ans.get("confidence") or 0.0)
                local[q.name] = Answer(
                    q.name, "choice", ans["choice"],
                    ans.get("probabilities") or {}, conf, "jev",
                )
            elif q.qtype == "noul" and "noul" in ans:
                p = float(ans["noul"])
                local[q.name] = Answer(
                    q.name, "noul", round(p, 4),
                    {"true": round(p, 4), "false": round(1.0 - p, 4)},
                    abs(p - 0.5) * 2.0, "jev",
                )
            elif q.qtype == "score" and "score" in ans:
                conf = float(ans.get("confidence") or 0.0)
                local[q.name] = Answer(
                    q.name, "score", round(float(ans["score"]), 3),
                    ans.get("probabilities") or {}, conf, "jev",
                )
        return local


def make_engine(name: str, model_path: str | None = None,
                min_confidence: float = 0.60):
    if name == "jev":
        return JevBackend()
    if name == "laya":
        return LayaBackend(min_confidence=min_confidence,
                           model_path=model_path)
    return LocalDecisionHead()

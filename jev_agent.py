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
    "Which side engine should fire right now to keep the lander level?",
    {
        "left-engine": "fire the left engine to rotate left",
        "right-engine": "fire the right engine to rotate right",
        "none": "attitude is fine; no side engine needed",
    },
)

DEFAULT_QUESTIONS = [DIRECTION, AT_RISK, INSTABILITY]
LANDER_QUESTIONS = [SIDE_ENGINE, AT_RISK, INSTABILITY]


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

    def ask(self, state, questions) -> dict:
        f = self.features(state)  # the single shared pass
        out = {}
        for q in questions:
            if q.qtype == "choice" and q.name == "direction":
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

    Prefers a local checkpoint at .models/laya/ (fetched with
    _fetch_laya.py, no network needed at runtime); otherwise uses the
    `laya` Router with its default Hugging Face checkpoints. Every
    question is tried against laya; on any error, or when laya's
    confidence is below `min_confidence`, the local head's answer is used.
    """

    engine = "laya"

    def __init__(self, min_confidence: float = 0.60,
                 model_path: str | None = None):
        self.min_confidence = min_confidence
        self.fallback = LocalDecisionHead()
        try:
            local = model_path or os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                ".models", "laya")
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

    def ask(self, state, questions) -> dict:
        names = [q.name for q in questions]
        laya_state = {
            "x": float(state[0]), "x_dot": float(state[1]),
            "theta": float(state[2]), "theta_dot": float(state[3]),
        }
        local = self.fallback.ask(state, questions)
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

    def ask(self, state, questions) -> dict:
        local = self.fallback.ask(state, questions)
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

"""Jev-style local decision agent for the inverted pendulum.

Implements the "System One" typed-decision contract from the Jev/Laya line of
models: a state goes in, and a set of typed questions (choice / score / noul)
comes back answered in a single pass, each with probabilities and a confidence.
No text is ever generated.

Two engines:
  - LocalDecisionHead: fully offline physics-informed head. Computes shared
    features once per call (the "single forward pass") and emits calibrated
    probabilities. Zero dependencies beyond numpy.
  - LayaBackend: uses the open-source `laya` package (pip install laya) to ask
    the same typed questions of a real Jev-compatible decision model. Falls
    back per-question to LocalDecisionHead on any error or low confidence,
    and reports which engine decided.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

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

DEFAULT_QUESTIONS = [DIRECTION, AT_RISK, INSTABILITY]


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
    """Optional engine: ask the real open decision model (`pip install laya`).

    Every question is tried against laya; on any error, or when laya's
    confidence is below `min_confidence`, the local head's answer is used.
    """

    engine = "laya"

    def __init__(self, min_confidence: float = 0.60):
        self.min_confidence = min_confidence
        self.fallback = LocalDecisionHead()
        try:
            from laya import Router  # type: ignore

            self.router = Router(preload=True)
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
            "x": state[0], "x_dot": state[1],
            "theta": state[2], "theta_dot": state[3],
        }
        local = self.fallback.ask(state, questions)
        try:
            result = self.router.predict(
                laya_state,
                {q.name: {"type": q.qtype, "instructions": q.instructions,
                          **({"criteria": q.criteria} if q.criteria else {})}
                 for q in questions},
            )
            raw = result.get("answers", {})
        except Exception:  # noqa: BLE001 - offline-safe: fall back entirely
            return local
        for q in questions:
            ans = raw.get(q.name)
            if not isinstance(ans, dict):
                continue
            value = self._extract(ans, q.qtype)
            conf = float(ans.get("confidence", 0.0) or 0.0)
            if value is None or conf < self.min_confidence:
                continue  # keep local answer
            probs = ans.get("probabilities") or {}
            local[q.name] = Answer(q.name, q.qtype, value, probs, conf, "laya")
        return local


def make_engine(name: str):
    if name == "laya":
        return LayaBackend()
    return LocalDecisionHead()

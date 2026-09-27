"""Map typed Jev answers to a CartPole action (0 = push left, 1 = push right)."""

from __future__ import annotations

from jev_agent import X_LIMIT


def decide(state, answers) -> int:
    """Turn the typed answer set into a single action.

    The direction choice drives the action. A motor-level track-limit law
    takes over only when the cart is already near a track end: then we push
    back toward the center regardless of the lean (the direction answer
    keeps full charge everywhere else).
    """
    action = 1 if answers["direction"].value == "right" else 0

    x = float(state[0])
    if abs(x) > 0.8 * X_LIMIT:
        action = 0 if x > 0 else 1
    return action

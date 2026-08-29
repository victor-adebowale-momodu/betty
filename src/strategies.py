"""Transparent, odds-only betting strategies.

A strategy is a pure function mapping a single match's market information to a
bet decision (or no bet). Strategies use only the odds / implied probabilities;
they never use future information, external statistics, or machine learning.

Decision contract
-----------------
Each strategy returns either None (no bet) or a dict:

    {"outcome": "H"|"D"|"A",
     "odds": float,          # the decimal odds selected
     "prob": float,          # normalised implied probability of the outcome
     "why": str}             # short human-readable rationale

The win amount is derived by the backtester from `outcome` and `odds`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OUTCOMES = ("H", "D", "A")


# ---------------------------------------------------------------------------
# Strategy implementation helpers
# ---------------------------------------------------------------------------

def _row_dict(row: pd.Series) -> dict:
    return {
        "FTR": row["FTR"],
        "odds": {o: float(row[f"B365C{o}"]) for o in OUTCOMES},
        "p": {o: float(row[f"p_{o}_norm"]) for o in OUTCOMES},
        "p_raw": {o: float(row[f"p_{o}_raw"]) for o in OUTCOMES},
        "overround": float(row["overround"]),
    }


def _make_decision(info: dict, outcome: str, why: str):
    return {"outcome": outcome, "odds": info["odds"][outcome],
            "prob": info["p"][outcome], "why": why}


# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------

def strategy_favourite(info: dict, **params) -> dict | None:
    """Bet on the favourite: the outcome with the highest normalised
    probability. No threshold — it always bets on the most-likely outcome.
    An interpretable baseline that the whole market (and most bettors) follows.
    """
    outcome = max(OUTCOMES, key=lambda o: info["p"][o])
    return _make_decision(info, outcome, "favourite")


def strategy_favourite_odds_cap(info: dict, max_odds: float = 3.0) -> dict | None:
    """Bet on the favourite only when its decimal odds are no longer than
    `max_odds` (a strong favourite). The rationale: heavy favourites are priced
    most consistently by the market and leave less room for gross mispricing.
    """
    outcome = max(OUTCOMES, key=lambda o: info["p"][o])
    if info["odds"][outcome] > max_odds:
        return None
    return _make_decision(info, outcome, f"favourite with odds<= {max_odds:.2f}")


def strategy_prob_bucket(info: dict, threshold: float = 0.45) -> dict | None:
    """Probability-bucket strategy: bet on the most likely outcome only when its
    normalised probability is at least `threshold`. Targets contests where the
    market assigns a clear single-favourite probability mass.
    """
    outcome = max(OUTCOMES, key=lambda o: info["p"][o])
    if info["p"][outcome] < threshold:
        return None
    return _make_decision(info, outcome,
                          f"favourite prob>= {threshold:.2f}")


def strategy_low_overround(info: dict, max_overround: float = 0.10) -> dict | None:
    """Overround-related strategy: bet on the favourite only when the book is
    priced efficiently (overround <= `max_overround`). Explanation: a low
    overround implies a competitive market whose implied probabilities are
    typically better calibrated to realised frequencies.
    """
    if info["overround"] > max_overround:
        return None
    outcome = max(OUTCOMES, key=lambda o: info["p"][o])
    return _make_decision(info, outcome,
                          f"favourite when overround<= {max_overround:.2f}")


def strategy_draw_value(info: dict, threshold: float = 0.30) -> dict | None:
    """Market-shape strategy: bet on the draw when its normalised probability
    reaches `threshold`. Draws are structurally underpriced/mispriced in many
    leagues, so this probes market-shape mispricing in the middle outcome.
    """
    if info["p"]["D"] >= threshold:
        return _make_decision(info, "D",
                              f"draw prob>= {threshold:.2f}")
    return None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# name -> (callable, default params, human description)
STRATEGIES = {
    "favourite": {
        "func": strategy_favourite, "params": {},
        "desc": "Always bet the favourite (highest normalised-probability outcome).",
    },
    "favourite_strong": {
        "func": strategy_favourite_odds_cap, "params": {"max_odds": 2.0},
        "desc": "Bet the favourite only when odds <= 2.0 (strong favourite).",
    },
    "prob_bucket_45": {
        "func": strategy_prob_bucket, "params": {"threshold": 0.45},
        "desc": "Bet the favourite only when its normalised probability >= 0.45.",
    },
    "low_overround": {
        "func": strategy_low_overround, "params": {"max_overround": 0.10},
        "desc": "Bet the favourite only when book overround <= 0.10.",
    },
    "draw_value_30": {
        "func": strategy_draw_value, "params": {"threshold": 0.30},
        "desc": "Bet the draw when its normalised probability >= 0.30.",
    },
}


def get_strategy(name: str):
    """Return (func, params, desc) for a named strategy."""
    if name not in STRATEGIES:
        raise KeyError(f"Unknown strategy '{name}'. "
                       f"Available: {list(STRATEGIES)}")
    spec = STRATEGIES[name]
    return spec["func"], spec["params"], spec["desc"]


def evaluate_strategy(df: pd.DataFrame, name: str) -> pd.Series:
    """Apply a named strategy to every row; return a Series of decisions
    (None or decision dict)."""
    func, params, _ = get_strategy(name)
    decisions = df.apply(
        lambda row: func(_row_dict(row), **params), axis=1
    )
    return decisions

"""Market-probability mathematics.

Handles the conversion of decimal odds into implied probabilities, the
overround / book percentage, and normalisation back to a probability simplex.

Raw implied probabilities and normalised probabilities are kept distinct;
normalised values are descriptive, not a claim about "true" probabilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OUTCOMES = ("H", "D", "A")


# ---------------------------------------------------------------------------
# Per-odds helpers
# ---------------------------------------------------------------------------

def implied_probability(odds: float) -> float:
    """Decimal odds -> implied probability p = 1/odds."""
    if odds is None or odds <= 1:
        return np.nan
    return 1.0 / odds


def overround(prob_h: float, prob_d: float, prob_a: float) -> float:
    """overround = p_H + p_D + p_A - 1."""
    return prob_h + prob_d + prob_a - 1.0


def book_percentage(prob_h: float, prob_d: float, prob_a: float) -> float:
    """book% = p_H + p_D + p_A."""
    return prob_h + prob_d + prob_a


def normalise(prob_h: float, prob_d: float, prob_a: float):
    """Divide each implied probability by the book percentage."""
    total = prob_h + prob_d + prob_a
    if total <= 0 or not np.isfinite(total):
        return np.nan, np.nan, np.nan
    return prob_h / total, prob_d / total, prob_a / total


# ---------------------------------------------------------------------------
# DataFrame-level enrichment
# ---------------------------------------------------------------------------

def add_probability_columns(df: pd.DataFrame, odds_cols=None) -> pd.DataFrame:
    """Return a copy of `df` with implied and normalised probability columns.

    Parameters
    ----------
    df : DataFrame with H/D/A odds columns (named via odds_cols or defaults).
    odds_cols : dict with keys H/D/A mapping to odds column names. Defaults to
        the Bet365 closing odds columns.

    Adds:
      p_H_raw, p_D_raw, p_A_raw            : 1/odds
      overround                            : sum(p_raw) - 1
      book_pct                             : sum(p_raw)
      p_H_norm, p_D_norm, p_A_norm         : p_raw / book_pct
    """
    from .data import B365_CLOSING_ODDS

    if odds_cols is None:
        odds_cols = B365_CLOSING_ODDS
    out = df.copy()
    for o in OUTCOMES:
        odds = pd.to_numeric(out[odds_cols[o]], errors="coerce")
        prob = np.where(odds > 1, 1.0 / odds, np.nan)
        out[f"p_{o}_raw"] = prob
    raw = [out[f"p_{o}_raw"] for o in OUTCOMES]
    book = raw[0] + raw[1] + raw[2]
    out["book_pct"] = book
    out["overround"] = book - 1.0
    for i, o in enumerate(OUTCOMES):
        out[f"p_{o}_norm"] = raw[i] / book.where(book > 0)
    return out

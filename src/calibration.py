"""Calibration analysis.

Determines whether market-implied probabilities correspond to realised
frequencies: when the market assigns approximately X% to an outcome, how often
does that outcome actually occur?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OUTCOMES = ("H", "D", "A")


def probability_buckets(df: pd.DataFrame, outcome: str, n_buckets: int = 10,
                        min_count: int = 20) -> pd.DataFrame:
    """Bucket a normalised-probability column and compare implied vs observed.

    Parameters
    ----------
    df : enriched processed DataFrame.
    outcome : "H", "D", or "A".
    n_buckets : number of equal-width buckets over [0, 1].
    min_count : minimum matches per bucket to report (dropping tiny samples,
        which are unreliable).

    Returns a table with, per bucket:
      bucket range, n, mean implied (normalised) probability, realised
      frequency, and a binomial confidence interval for the frequency.
    """
    col = f"p_{outcome}_norm"
    df = df.copy()
    df["_hit"] = (df["FTR"] == outcome).astype(int)

    edges = np.linspace(0.0, 1.0, n_buckets + 1)
    labels = [f"{edges[i]:.2f}-{edges[i+1]:.2f}" for i in range(n_buckets)]
    df["_bucket"] = pd.cut(df[col], bins=edges, labels=labels,
                           include_lowest=True, right=False)

    rows = []
    for label in labels:
        sub = df[df["_bucket"] == label]
        if len(sub) < min_count:
            continue
        n = len(sub)
        mean_imp = float(sub[col].mean())
        freq = float(sub["_hit"].mean())
        lo, hi = _binomial_ci(n, int(sub["_hit"].sum()))
        rows.append({
            "bucket": label,
            "n": n,
            "implied": mean_imp,
            "observed": freq,
            "ci_low": lo,
            "ci_high": hi,
            "diff": freq - mean_imp,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def calibration_by_season(df: pd.DataFrame, outcome: str,
                          n_buckets: int = 10) -> pd.DataFrame:
    """Calibration split by season, using a coarser bucket count per season."""
    rows = []
    for season, g in df.groupby("season"):
        tbl = probability_buckets(g, outcome, n_buckets=n_buckets,
                                  min_count=10)
        if tbl.empty:
            continue
        tbl = tbl.copy()
        tbl["season"] = season
        rows.append(tbl)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def calibration_statistics(tbl: pd.DataFrame) -> dict:
    """Summarise calibration error across buckets (weighted by n)."""
    if tbl.empty:
        return {}
    n = tbl["n"].sum()
    mae = float((tbl["diff"].abs() * tbl["n"]).sum() / n)
    bias = float((tbl["diff"] * tbl["n"]).sum() / n)
    return {
        "buckets": int(len(tbl)),
        "total_matches": int(n),
        "mae": mae,
        "bias": bias,
    }


def _binomial_ci(n: int, k: int, alpha: float = 0.05):
    """Wilson score interval for a realised proportion k/n."""
    if n == 0:
        return np.nan, np.nan
    z = 1.96  # approx for alpha=0.05 two-sided
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half

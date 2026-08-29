"""Betting-performance summaries.

Transparent, interpretable metrics computed from a match-level betting ledger.
Everything here works on the ledger produced by `backtest.backtest_strategy`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def net_profit(ledger: pd.DataFrame) -> float:
    return float(ledger["profit"].sum())


def gross_return(ledger: pd.DataFrame) -> float:
    """Gross return = stake returned/earned on wins plus stake-back assuming
    losing bets lose the stake. Simpler interpretation: sum(stake) + sum(profit)."""
    return float(ledger["stake"].sum() + ledger["profit"].sum())


def total_stake(ledger: pd.DataFrame) -> float:
    return float(ledger["stake"].sum())


def roi(ledger: pd.DataFrame) -> float:
    stake = total_stake(ledger)
    if stake <= 0:
        return 0.0
    return net_profit(ledger) / stake


def win_rate(ledger: pd.DataFrame) -> float:
    if ledger.empty:
        return 0.0
    return float((ledger["win"] == 1).mean())


def average_odds(ledger: pd.DataFrame) -> float:
    if ledger.empty:
        return 0.0
    return float(ledger["odds"].mean())


def max_drawdown(ledger: pd.DataFrame) -> float:
    """Maximum peak-to-trough decline in cumulative profit (absolute £)."""
    if ledger.empty:
        return 0.0
    cum = ledger["cum_profit"].to_numpy()
    peak = np.maximum.accumulate(np.insert(cum, 0, 0.0))
    drawdown = peak[1:] - cum
    return float(drawdown.max())


def longest_losing_streak(ledger: pd.DataFrame) -> int:
    """Longest consecutive run of losing bets."""
    if ledger.empty:
        return 0
    streak = best = 0
    for w in ledger["win"]:
        if w == 0:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best


def summarize_ledger(ledger: pd.DataFrame) -> dict:
    """Compute the full metric set for a ledger."""
    n_bets = len(ledger)
    n_wins = int((ledger["win"] == 1).sum())
    n_losses = int((ledger["win"] == 0).sum())
    return {
        "bets": n_bets,
        "wins": n_wins,
        "losses": n_losses,
        "win_rate": win_rate(ledger),
        "total_stake": total_stake(ledger),
        "gross_return": gross_return(ledger),
        "net_profit": net_profit(ledger),
        "roi": roi(ledger),
        "average_odds": average_odds(ledger),
        "max_drawdown": max_drawdown(ledger),
        "longest_losing_streak": longest_losing_streak(ledger),
    }


def season_summary(ledger: pd.DataFrame) -> pd.DataFrame:
    """Per-season metrics as a DataFrame."""
    rows = []
    if ledger.empty:
        return pd.DataFrame()
    for season, g in ledger.groupby("season"):
        m = summarize_ledger(g)
        m["season"] = season
        rows.append(m)
    return pd.DataFrame(rows)

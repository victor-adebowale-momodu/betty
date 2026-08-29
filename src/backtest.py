"""Historical backtesting of betting strategies.

Simulates what would have happened if a strategy's bet decisions had been
followed with a fixed stake per qualifying bet. Produces a match-level ledger.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from . import strategies as S
from . import metrics as M


def backtest_strategy(df: pd.DataFrame, name: str,
                      stake: float = 1.0) -> pd.DataFrame:
    """Run a named strategy over an enriched processed DataFrame.

    Produces a ledger with one row per qualifying bet:

      date, season, home, away, FTR, outcome, odds, stake,
      win, profit, cum_profit, why

    Returns an empty DataFrame if no bets qualify.
    """
    from .strategies import evaluate_strategy

    decisions = evaluate_strategy(df, name)

    recs = []
    cum = 0.0
    for idx, dec in decisions.items():
        if dec is None:
            continue
        row = df.loc[idx]
        outcome = dec["outcome"]
        win = int(row["FTR"] == outcome)
        profit = stake * (dec["odds"] - 1.0) if win else -stake
        cum += profit
        recs.append({
            "date": row["Date"],
            "season": row["season"],
            "home": row["HomeTeam"],
            "away": row["AwayTeam"],
            "FTR": row["FTR"],
            "outcome": outcome,
            "odds": dec["odds"],
            "prob": dec.get("prob", np.nan),
            "overround": row["overround"],
            "stake": stake,
            "win": win,
            "profit": profit,
            "cum_profit": cum,
            "why": dec.get("why", ""),
        })
    return pd.DataFrame(recs)


def add_ledger_columns(df: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    """Attach per-match bet/no-bet + profit columns to `df` for the app's
    Match Explorer. Ledger must be produced by backtest_strategy on the same df.
    """
    out = df.copy()
    out["bet"] = False
    out["bet_outcome"] = ""
    out["bet_profit"] = 0.0
    out["bet_strategy_label"] = ""
    if ledger.empty:
        return out
    key = list(zip(ledger["date"], ledger["home"], ledger["away"]))
    key_set = {tuple(k) for k in key}
    profit_map = dict(zip(key, ledger["profit"]))
    outcome_map = dict(zip(key, ledger["outcome"]))
    # Match by (date, home, away) since key uses index of original df.
    for idx in out.index:
        row = out.loc[idx]
        k = (row["Date"], row["HomeTeam"], row["AwayTeam"])
        if k in key_set:
            out.loc[idx, "bet"] = True
            out.loc[idx, "bet_outcome"] = outcome_map.get(k, "")
            out.loc[idx, "bet_profit"] = profit_map.get(k, 0.0)
    return out


# Default chronological split for the primary data.
DEV_SEASONS = ["20/21", "21/22", "22/23"]
OOS_SEASONS = ["23/24", "24/25", "25/26"]


def dev_oos_summary(df: pd.DataFrame, name: str):
    """Backtest a strategy on the dev and OOS periods; return both metric dicts
    plus the two ledgers. Device/OOS split is purely chronological and fixed.
    """
    dev = df[df["season"].isin(DEV_SEASONS)]
    oos = df[df["season"].isin(OOS_SEASONS)]
    ledger_dev = backtest_strategy(dev, name)
    ledger_oos = backtest_strategy(oos, name)
    return {
        "dev": M.summarize_ledger(ledger_dev),
        "oos": M.summarize_ledger(ledger_oos),
        "ledger_dev": ledger_dev,
        "ledger_oos": ledger_oos,
        "desc": S.STRATEGIES[name]["desc"],
    }

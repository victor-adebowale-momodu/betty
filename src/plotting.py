"""Reusable Plotly visualisations for the research notebooks and Streamlit app.

Unified, clean styling. These functions take data (usually the enriched
processed DataFrame) and return Plotly figure objects without doing any
research calculation themselves — callers bring the numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_COLOR = "#1f77b4"
_COLOR2 = "#d62728"
_REF = "#888"


def _layout(fig: go.Figure, title: str | None = None, x: str | None = None,
            y: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title=x,
        yaxis_title=y,
        margin=dict(l=50, r=20, t=60, b=40),
        height=420,
    )
    return fig


def histogram(values: pd.Series, title: str = "Distribution",
              x: str = "value", nbins: int = 40,
              color: str = _COLOR) -> go.Figure:
    fig = go.Figure(go.Histogram(x=values.dropna(), nbinsx=nbins,
                                 marker_color=color, opacity=0.8))
    return _layout(fig, title=title, x=x, y="count")


def overround_by_season(df: pd.DataFrame) -> go.Figure:
    med = df.groupby("season")["overround"].median().sort_index()
    fig = go.Figure(go.Bar(x=med.index.astype(str), y=med.values,
                           marker_color=_COLOR))
    return _layout(fig, title="Median overround by season",
                   x="season", y="overround")


def calibration_curve(tbl: pd.DataFrame, title: str = "Calibration") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="perfect",
        line=dict(color=_REF, dash="dash")))
    if not tbl.empty:
        fig.add_trace(go.Scatter(
            x=tbl["implied"], y=tbl["observed"], mode="markers+lines",
            name="implied vs observed", marker=dict(color=_COLOR),
            error_y=dict(type="data", array=tbl["ci_high"] - tbl["observed"],
                         arrayminus=tbl["observed"] - tbl["ci_low"],
                         visible=True, color=_COLOR2)))
        fig.add_trace(go.Scatter(
            x=tbl["implied"], y=tbl["n"], mode="markers", name="sample count",
            yaxis="y2", marker=dict(color="#999", size=6)))
    fig.update_layout(
        template="plotly_white", title=title,
        xaxis_title="implied probability",
        yaxis=dict(title="observed frequency", range=[0, 1]),
        yaxis2=dict(title="n", overlaying="y", side="right", showgrid=False),
        height=420, margin=dict(l=50, r=60, t=60, b=40))
    return fig


def cumulative_pnl(ledger: pd.DataFrame, title: str = "Cumulative P&L") -> go.Figure:
    fig = go.Figure()
    if not ledger.empty:
        fig.add_trace(go.Scatter(
            x=ledger["date"], y=ledger["cum_profit"], mode="lines",
            name="cumulative profit", line=dict(color=_COLOR)))
        fig.add_hline(y=0, line_dash="dash", line_color=_REF)
    return _layout(fig, title=title, x="date", y="cumulative profit (£)")


def drawdown_curve(ledger: pd.DataFrame,
                   title: str = "Drawdown") -> go.Figure:
    fig = go.Figure()
    if not ledger.empty:
        cum = ledger["cum_profit"].to_numpy()
        running_max = np.maximum.accumulate(np.insert(cum, 0, 0.0))[1:]
        dd = cum - running_max
        fig.add_trace(go.Scatter(
            x=ledger["date"], y=dd, mode="lines", name="drawdown",
            fill="tozeroy", line=dict(color=_COLOR2)))
    return _layout(fig, title=title, x="date", y="drawdown (£)")


def season_roi(season_tbl: pd.DataFrame, title: str = "ROI by season") -> go.Figure:
    fig = go.Figure()
    if not season_tbl.empty:
        fig.add_trace(go.Bar(
            x=season_tbl["season"].astype(str),
            y=season_tbl["roi"] * 100,
            marker_color=np.where(season_tbl["roi"] >= 0, _COLOR, _COLOR2)))
        fig.add_hline(y=0, line_dash="dash", line_color=_REF)
    return _layout(fig, title=title, x="season", y="ROI (%)")


def returns_distribution(ledger: pd.DataFrame,
                         title: str = "Returns distribution") -> go.Figure:
    fig = go.Figure()
    if not ledger.empty:
        returns = np.where(ledger["win"] == 1,
                           ledger["odds"] - 1.0, -1.0)
        fig.add_trace(go.Histogram(
            x=returns * 100, nbinsx=50, marker_color=_COLOR, opacity=0.8))
    return _layout(fig, title=title, x="return (%)", y="count")


def market_shape(fig_title: str = "Market shape"):
    """Scaffolding for market-structure inspection (e.g. prob gap vs result).

    This is intentionally a thin helper; the notebook builds richer views,
    while the app keeps this layout. Returns a blank subplot figure.
    """
    return make_subplots(rows=1, cols=1, subplot_titles=[fig_title])

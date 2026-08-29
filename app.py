"""Betty — interactive research dashboard (Streamlit).

Presentation layer only. All research calculations live in `src/`; this file
only wires data into Plotly figures and Streamlit components.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import data as D
from src import probabilities as P
from src import calibration as C
from src import strategies as S
from src import backtest as B
from src import metrics as M
from src import plotting as plt

st.set_page_config(page_title="Betty · Bet365 La Liga 1X2", layout="wide")

# ---------------------------------------------------------------------------
# Load once
# ---------------------------------------------------------------------------
@st.cache_data
def load():
    primary = P.add_probability_columns(D.load_processed())
    return primary

primary = load()

st.title("Betty — Bet365 La Liga 1X2 Closing Odds")

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
st.header("Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Seasons (primary)", len(primary["season"].unique()))
c2.metric("Matches analysed", len(primary))
c3.metric("Avg overround", f"{primary['overround'].mean():.3f}")
c4.metric("Median overround", f"{primary['overround'].median():.3f}")

st.caption("Primary dataset = seasons with true Bet365 closing odds, excluding "
           "2018-19 (no closing-odds field) and 2019-20 (COVID-disrupted).")

# In-sample vs out-of-sample (chronological split).
oos_rows = []
for name in S.STRATEGIES:
    r = B.dev_oos_summary(primary, name)
    oos_rows.append({"strategy": name, "desc": S.STRATEGIES[name]["desc"],
                     "dev_roi": r["dev"]["roi"], "oos_roi": r["oos"]["roi"],
                     "oos_profit": r["oos"]["net_profit"], "oos_bets": r["oos"]["bets"]})
oos_tbl = pd.DataFrame(oos_rows).set_index("strategy")

st.subheader("Strategy performance — development vs out-of-sample (chronological split)")
st.dataframe(oos_tbl[["dev_roi", "oos_roi", "oos_bets", "oos_profit"]].
             rename(columns={
                 "dev_roi": "In-sample ROI (20/21–22/23)",
                 "oos_roi": "Out-of-sample ROI (23/24–25/26)",
                 "oos_bets": "OOS bets", "oos_profit": "OOS profit (£)"}).round(4))
st.caption("In-sample and out-of-sample are clearly separated. The high-volume "
           "favourite family shows a small (+1%) OOS ROI; `draw_value_30` does "
           "not persist out-of-sample. No strategy shows a large, economically "
           "meaningful edge.")

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
page = st.sidebar.radio("Section", [
    "Market", "Calibration", "Strategies", "Match Explorer"])

# ---------------------------------------------------------------------------
# Market
# ---------------------------------------------------------------------------
if page == "Market":
    st.header("Market structure")
    seasons = sorted(primary["season"].unique())
    sel = st.multiselect("Seasons", seasons, default=seasons)
    sub = primary[primary["season"].isin(sel)] if sel else primary

    cols = st.columns(2)
    with cols[0]:
        rec_col = st.selectbox("Odds distribution—outcome", ["H", "D", "A"])
        fig = plt.histogram(sub[f"B365C{rec_col}"], nbins=60,
                            title=f"Bet365 {rec_col} closing odds by season")
        st.plotly_chart(fig, use_container_width=True)
    with cols[1]:
        prob_col = st.selectbox("Normalised probability—outcome", ["H", "D", "A"],
                                key="market_prob")
        fig = plt.histogram(sub[f"p_{prob_col}_norm"], nbins=40,
                            title=f"Normalised implied p_{prob_col}")
        st.plotly_chart(fig, use_container_width=True)

    fig = plt.histogram(sub["overround"], nbins=40,
                        title="Overround (closing book) distribution",
                        x="overround")
    st.plotly_chart(fig, use_container_width=True)

    fig = plt.overround_by_season(sub)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
elif page == "Calibration":
    st.header("Calibration")
    st.write("When the market assigns approximately X% to an outcome, how often "
             "does it actually occur? No discrepancy here is called an edge.")
    col = st.selectbox("Outcome", ["H", "D", "A"])
    n_buckets = st.slider("Bucket count", 4, 20, 10)
    tbl = C.probability_buckets(primary, col, n_buckets=n_buckets)
    if tbl.empty:
        st.warning("Not enough observations per bucket.")
    else:
        fig = plt.calibration_curve(tbl, title=f"{col}: implied vs observed")
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Calibration table")
        st.dataframe(tbl.round(4))
        st.caption(f"MAE={C.calibration_statistics(tbl).get('mae', float('nan')):.4f}, "
                   f"bias={C.calibration_statistics(tbl).get('bias', float('nan')):.4f}")

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
elif page == "Strategies":
    st.header("Strategies")
    st.write("Fixed £1 stake per qualifying bet. In-sample and out-of-sample "
             "results are shown separately.")
    name = st.selectbox("Strategy", list(S.STRATEGIES))
    r = B.dev_oos_summary(primary, name)
    st.markdown(f"**Rule:** {r['desc']}")

    col = st.columns(2)[0]
    for tag in ("dev", "oos"):
        m = r[tag]
        st.subheader(f"{'In-sample' if tag=='dev' else 'Out-of-sample'}"
                     f" ({'20/21–22/23' if tag=='dev' else '23/24–25/26'})")
        kpi = st.columns(4)
        kpi[0].metric("Bets", m["bets"])
        kpi[1].metric("Win rate", f"{m['win_rate']:.3f}")
        kpi[2].metric("Avg odds", f"{m['average_odds']:.2f}")
        kpi[3].metric("ROI", f"{m['roi']:.4f}")
        kpi2 = st.columns(4)
        kpi2[0].metric("Profit (£)", f"{m['net_profit']:.2f}")
        kpi2[1].metric("Max drawdown (£)", f"{m['max_drawdown']:.2f}")
        kpi2[2].metric("Total stake (£)", f"{m['total_stake']:.2f}")
        kpi2[3].metric("Longest loss streak", m["longest_losing_streak"])
        fig = plt.cumulative_pnl(r[f"ledger_{tag}"],
                                 title=f"Cumulative P&L — {tag}")
        st.plotly_chart(fig, use_container_width=True)
        stbl = M.season_summary(r[f"ledger_{tag}"])
        st.dataframe(stbl[["season", "bets", "win_rate", "net_profit", "roi"]].round(4))

# ---------------------------------------------------------------------------
# Match Explorer
# ---------------------------------------------------------------------------
elif page == "Match Explorer":
    st.header("Match Explorer")
    st.write("Audit individual matches: what the strategy decided and what "
             "occurred.")
    name = st.selectbox("Strategy", list(S.STRATEGIES), key="explorer_strategy")
    # Attach bet info to every match using a full-primary ledger.
    ledger = B.backtest_strategy(primary, name)
    annotated = B.add_ledger_columns(primary, ledger)
    annotated["date"] = annotated["Date"].dt.date

    month_min = primary["Date"].min().date()
    month_max = primary["Date"].max().date()
    lo, hi = st.slider("Date range", min_value=month_min, max_value=month_max,
                       value=(month_min, month_max))
    mask = (annotated["date"] >= lo) & (annotated["date"] <= hi)
    view = annotated[mask]

    show_bets_only = st.checkbox("Only show bets", value=False)
    if show_bets_only:
        view = view[view["bet"]]

    cols = ["date", "season", "HomeTeam", "AwayTeam", "FTR",
            "B365CH", "B365CD", "B365CA",
            "p_H_norm", "p_D_norm", "p_A_norm", "overround",
            "bet", "bet_outcome", "bet_profit"]
    st.dataframe(view[cols].round(4), use_container_width=True,
                 height=520)

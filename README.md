# Betty

A small, readable quantitative research project investigating whether **Bet365's
La Liga 1X2 closing odds, used alone**, can support a profitable long-run
betting strategy.

Betty is deliberately simple: no machine learning, no large framework, no tests
wrapper. It is structured so a single researcher can understand every step.

## Research question

> Can information contained solely in Bet365's La Liga 1X2 closing odds be used
> to construct a strategy that generates persistent positive returns over the
> long run?

The market is 1X2 (H = home win, D = draw, A = away win). Strategies use
**odds information only** — no team stats, xG, injuries, league position, news,
weather, external data, or machine learning. The goal is not to prove betting is
profitable; it is to determine whether there is credible evidence of a
repeatable, economically meaningful **odds-only** edge.

## Data

Raw files in `data/raw/` are per-season CSVs in the standard
football-data.co.uk layout, covering **2018-19 .. 2025-26**. Each raw file is
left untouched.

Relevant fields:

| Field        | Meaning                              |
|--------------|--------------------------------------|
| `Date`       | Match date                           |
| `HomeTeam`   | Home team                            |
| `AwayTeam`   | Away team                            |
| `FTR`        | Full-time result: `H` / `D` / `A`    |
| `B365H/D/A`  | Bet365 1X2 odds                      |
| `B365CH/CD/CA` | Bet365 **closing** 1X2 odds        |

### Closing-odds ambiguity

The `B365CH/CD/CA` (Bet365 closing) columns are present in the 2019-20 through
2025-26 files but **not** in the 2018-19 file, which only has the regular
`B365H/D/A` fields. Rather than silently substituting another odds field, Betty
**excludes 2018-19 from the primary dataset**. The raw 2018-19 data remains
available for a later sensitivity analysis.

### Excluded seasons

| Season | Reason                                          | Status      |
|--------|-------------------------------------------------|-------------|
| 2018-19| No Bet365 closing-odds field                    | Excluded from primary |
| 2019-20| Disrupted by COVID-19                           | Excluded from primary |

**Primary dataset = 2020-21 .. 2025-26** (2,280 matches, 380 per season), all
carrying true Bet365 closing odds.

## 1X2 market

For each match the closing odds imply a probability for each of H, D, A.

## Implied probability

For decimal odds `o`: `p = 1 / o`, computed independently for H, D and A.

## Overround

```
book_pct    = p_H + p_D + p_A
overround   = book_pct - 1
```

The closing book typically carries a small overround (`~5.6%` median). A mature
market's implied probabilities must be **normalised** to sum to 1 before they
can be compared to realised outcome frequencies:

```
p_H_norm = p_H / book_pct,  etc.
```

Normalised values are a **descriptive re-scaling**, not a claim about "true"
probabilities.

## Calibration

`src/calibration.py` compares *what the price implied* with *what actually
occurred*. Matches are grouped into probability buckets per outcome; each bucket
reports the mean implied probability, the realised frequency, and a binomial
(Wilson) confidence interval. Calibration is examined separately for H, D and A,
and by season. A discrepancy between implied and observed is **not** treated as
an edge.

## Strategy methodology

`src/strategies.py` defines a small set of transparent, odds-only rules:

| Strategy          | Rule                                                        |
|-------------------|-------------------------------------------------------------|
| `favourite`       | Always bet the highest normalised-probability outcome       |
| `favourite_strong`| Favourite only when closing odds ≤ 2.0                      |
| `prob_bucket_45`  | Favourite only when its normalised probability ≥ 0.45       |
| `low_overround`   | Favourite only when the closing book overround ≤ 0.10       |
| `draw_value_30`   | Bet the draw when its normalised probability ≥ 0.30         |

No parameter mining, no ML, no future information. These are a handful of
interpretable rules with a clear reason for existing.

## Backtesting

`src/backtest.py` simulates following each strategy with a **fixed £1 stake**
per qualifying bet:

```
win   profit = odds - 1
lose  profit = -1
```

It returns a match-level ledger (date, season, teams, result, selected outcome,
odds, stake, win/loss, profit, cumulative profit). `src/metrics.py` summarises
bets, wins/losses, win rate, total stake, gross return, net profit, ROI, average
odds, max drawdown, longest losing streak, and season-level results.

## Out-of-sample methodology

The central criterion is **persistence**, not one strong historical result.

- **Chronological split only.** Matches are never randomly shuffled or split.
- Development (in-sample): **2020-21, 2021-22, 2022-23**.
- Hold-out (out-of-sample): **2023-24, 2024-25, 2025-26**.
- All strategies are **predefined and frozen** before the OOS period.

Notebook 05 also runs a **walk-forward** expansion of the development window and
reports the favourite strategy's ROI on each successive held-out season.

## Results (out-of-sample)

At £1 stakes, with a ~5.6% closing overround:

- **Favourite family** (favourite / favourite_strong / prob_bucket_45 /
  low_overround) — in-sample ROI ≈ **-1.5%** but out-of-sample ROI ≈ **+1.0%**.
  This is the opposite of overfitting (a simple, pre-defined rule holding up out
  of sample), yet the OOS profit is small (~£8–12 over ~1,100 bets) and is
  **not an economically meaningful edge**.
- **draw_value_30** — in-sample ROI ≈ **+1.9%** but out-of-sample ROI ≈
  **-5.4%**. This is the clearest non-persistence case: its in-sample value
  evaporates on unseen data.

**Bottom line:** none of the tested strategies demonstrates a large, persistent,
economically meaningful odds-only edge. The closing overround appears to absorb
most of the favourite-value signal available from Bet365's La Liga 1X2 closing
prices. The negative-to-marginal results are treated as valid findings, not
failures.

## Limitations

- Small sample: 2,280 matches / ~6 seasons; ROI point estimates are noisy.
- 2018-19 and 2019-20 are excluded from the primary analysis (no closing odds /
  COVID). The raw data remains for sensitivity work.
- Unit £1 stakes; real-world execution (fractional stakes, bookmaker limits,
  variance) is not modelled.
- Only Bet365's La Liga 1X2 closing prices are studied.

## Project layout

```
betty/
├── data/
│   ├── raw/            # original CSVs (read-only)
│   └── processed/      # laliga_processed.parquet (generated)
├── notebooks/          # 01..05 research notebooks
├── src/                # reusable library (data, probabilities, calibration,
│                       #   strategies, backtest, metrics, plotting, main)
├── app.py              # Streamlit dashboard
├── pyproject.toml      # project metadata + dependencies (managed by uv)
├── uv.lock             # locked dependency resolution (generated by uv sync)
└── README.md
```

## How to run

Dependencies are declared in `pyproject.toml` and managed with **uv**:

```bash
uv sync          # create the environment and install the project + dependencies
```

### CLI

```bash
uv run python -m src.main inspect   # inspect the raw dataset
uv run python -m src.main process   # build the processed primary dataset
uv run python -m src.main analyze   # run core market / calibration analysis
uv run python -m src.main backtest  # run the predefined strategies
uv run python -m src.main all       # full pipeline
```

The package also exposes a `betty` console script; either
`uv run betty inspect` or `python -m src.main inspect` works.

### Notebooks

```bash
uv run jupyter notebook notebooks/
```

Run `01_data_exploration.ipynb` through `05_out_of_sample.ipynb` in order.

### Streamlit dashboard

```bash
uv run streamlit run app.py
```

Sections: **Overview** (in-sample vs out-of-sample clearly separated), **Market**
(odds / probability / overround charts with season filtering), **Calibration**
(implied vs observed with confidence intervals), **Strategies** (per-strategy
dev/OOS metrics, P&L, drawdown), and **Match Explorer** (per-match audit table).
